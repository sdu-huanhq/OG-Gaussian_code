import os
import sys
import argparse
import yaml
import torch
import numpy as np
import pickle
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
from argparse import Namespace
import maskclip_onnx
from tqdm import tqdm
from loguru import logger

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from scene import Scene, GaussianModel, skip_feat_decoder
from gaussian_renderer import render
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, cosine_loss


class MaskCLIPFeaturizer(torch.nn.Module):
    def __init__(self, root_dir):
        super().__init__()
        self.model, _ = maskclip_onnx.clip.load("ViT-L/14@336px", download_root=root_dir)
        self.model.eval()
        self.patch_size = self.model.visual.patch_size
    def forward(self, img):
        b, _, h, w = img.shape
        features = self.model.get_patch_encodings(img).to(torch.float32)
        patch_h, patch_w = h // self.patch_size, w // self.patch_size
        return features.reshape(b, patch_h, patch_w, -1).permute(0, 3, 1, 2)

_DINO_MODEL_CACHE = None
_CLIP_MODEL_CACHE = None

def get_dino_model(device="cuda"):
    global _DINO_MODEL_CACHE
    if _DINO_MODEL_CACHE is None:
        try:
            model = torch.hub.load('facebookresearch/dinov3', 'dinov3_vits16plus', source='github', trust_repo=True)
            model.to(device)
            model.eval()
            _DINO_MODEL_CACHE = model
        except Exception as e:
            logger.error(f"Failed to load DINOv3: {e}")
            raise e
    return _DINO_MODEL_CACHE

def get_clip_model(weights_root):
    global _CLIP_MODEL_CACHE
    if _CLIP_MODEL_CACHE is None:
        try:
            clip_model = MaskCLIPFeaturizer(weights_root).cuda().eval()
            _CLIP_MODEL_CACHE = clip_model
        except Exception as e:
            logger.error(f"Failed to load DINOv3: {e}")
            raise e
    return _CLIP_MODEL_CACHE

class PipelineArgs(Namespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

def get_combined_args(full_config, source_path=None, model_path=None):
    m_args = full_config.get('model_params', {}).copy()
    o_args = full_config.get('optimization_params', {}).copy()
    p_args = full_config.get('pipeline_params', {}).copy()
    
    if source_path: m_args['source_path'] = source_path
    if model_path: m_args['model_path'] = model_path
    if 'quiet' not in m_args: m_args['quiet'] = False
    
    return PipelineArgs(**{**m_args, **o_args, **p_args})

def extract_features_memory(data, args, clip_model, dino, device):
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    raw_transform = T.Compose([T.ToTensor(), norm])
    dino_transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])])
    
    raw_img_H, raw_img_W = args.raw_img_resolution
    object_W = args.obj_feat_res
    final_W = args.final_feat_res
    final_H = raw_img_H * final_W // raw_img_W
    object_H = raw_img_H * object_W // raw_img_W
    
    scene_features = {}
    cameras_group = data['key_state_000']["camera_data"]
    
    for cam_key in cameras_group.keys():
        cam_data = cameras_group[cam_key]
        rgb_np = cam_data['rgb']
        mask_np = cam_data['objs_mask']
        
        pil_img = Image.fromarray(rgb_np)
        width, height = pil_img.size
        
        # DINO Processing
        ratio = args.dino_resolution / max(width, height)
        new_w, new_h = int(width * ratio), int(height * ratio)
        rgb_resized = np.array(pil_img.resize((new_w, new_h), Image.BILINEAR))
        rgb_tensor = dino_transform(rgb_resized)[:3].unsqueeze(0)
        
        _, _, H, W = rgb_tensor.shape
        target_H = H // dino.patch_size * dino.patch_size
        target_W = W // dino.patch_size * dino.patch_size
        rgb_tensor = F.interpolate(rgb_tensor, size=(target_H, target_W)).cuda()
        
        with torch.no_grad():
            features_dict = dino.forward_features(rgb_tensor)
            features = features_dict["x_norm_patchtokens"][0]
        
        features_hwc = features.reshape((target_H // dino.patch_size, target_W // dino.patch_size, -1)).cpu().numpy()
        features_chw = features_hwc.transpose((2, 0, 1))
        
        # CLIP Processing
        raw_input_image = raw_transform(pil_img)
        with torch.no_grad():
            whole_image_feature = clip_model(raw_input_image[None].cuda())[0]
            
        resized_clip_feat = F.interpolate(whole_image_feature.unsqueeze(0).float(), size=(object_H, object_W), mode='bilinear', align_corners=False)
        mask_tensor = torch.from_numpy(mask_np.astype(np.int64)).to(device).unsqueeze(0).unsqueeze(0)
        resized_mask = F.interpolate(mask_tensor.float(), size=(object_H, object_W), mode='nearest').long().squeeze()
        
        clip_feat_dim = whole_image_feature.shape[0]
        aggregated_feat_map = torch.zeros((clip_feat_dim, object_H, object_W), dtype=torch.float32, device=device)
        unique_ids = torch.unique(resized_mask)
        
        for uid in unique_ids:
            if uid == 0: continue
            binary_mask = (resized_mask == uid)
            if binary_mask.sum() == 0: continue
            masked_feat = resized_clip_feat[0, :, binary_mask]
            aggregated_feat_map[:, binary_mask] = masked_feat.mean(dim=1)[:, None]

        final_feat = F.interpolate(aggregated_feat_map.unsqueeze(0), size=(final_H, final_W), mode='bilinear', align_corners=False)[0]
        scene_features[cam_key] = {"dino": features_chw, "clip": final_feat.cpu().numpy()}
        
    return scene_features

def save_masks_for_segmentation(data, output_dir, action_id, anchor_id):
    action_mask_dir = os.path.join(output_dir, "action_masks")
    anchor_mask_dir = os.path.join(output_dir, "anchor_masks")
    os.makedirs(action_mask_dir, exist_ok=True)
    os.makedirs(anchor_mask_dir, exist_ok=True)
    cameras_group = data['key_state_000']["camera_data"]
    for cam_key, cam_data in cameras_group.items():
        mask_arr = cam_data['objs_mask']
        act_mask = np.zeros_like(mask_arr, dtype=np.uint8)
        act_mask[mask_arr == action_id] = 255
        Image.fromarray(act_mask).save(os.path.join(action_mask_dir, f"{cam_key}.png"))
        anc_mask = np.zeros_like(mask_arr, dtype=np.uint8)
        anc_mask[mask_arr == anchor_id] = 255
        Image.fromarray(anc_mask).save(os.path.join(anchor_mask_dir, f"{cam_key}.png"))

def run_reconstruction(full_config, pkl_data, scene_name, output_path, device_id):
    torch.cuda.set_device(device_id)
    device = "cuda"
    
    dino = get_dino_model(device)
    
    weights_root = full_config.get('weights_root', './weights')
    weights_root = os.path.abspath(weights_root)
    
    clip_model = get_clip_model(weights_root)

    args = get_combined_args(full_config, source_path=full_config.get('source_pkl_path', ''), model_path=output_path)
    
    scene_features = extract_features_memory(pkl_data, args, clip_model, dino, device)

    action_id = int(pkl_data['action_id'][0])
    anchor_id = int(pkl_data['anchor_id'][0])
    save_masks_for_segmentation(pkl_data, output_path, action_id, anchor_id)
    
    safe_state(args.quiet)
    gaussians = GaussianModel(args.sh_degree, args.distill_feature_dim)
    scene = Scene(args, gaussians, data=pkl_data, fx=414.5749, fy=414.5749)
    
    train_cameras = scene.getTrainCameras()
    for cam in train_cameras:
        if cam.image_name in scene_features:
            cam.dino_feat_chw = torch.from_numpy(scene_features[cam.image_name]["dino"]).cuda()
            cam.feat_chw = torch.from_numpy(scene_features[cam.image_name]["clip"]).cuda()

    gaussians.training_setup(args)
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    my_feat_decoder = skip_feat_decoder(args.distill_feature_dim, part_level=False).cuda()
    decoder_optimizer = torch.optim.Adam(my_feat_decoder.parameters(), lr=0.001)

    logger.info(f"-> Starting reconstruction for {scene_name} on GPU {device_id}...")
    ema_loss_for_log = 0.0
    first_iter = 0
    progress_bar = tqdm(range(first_iter, args.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, args.iterations + 1):
        gaussians.update_learning_rate(iteration)
        gaussians.update_feature_learning_rate(iteration)
        if iteration % 1000 == 0: gaussians.oneupSHdegree()

        viewpoint_cam = scene.getTrainCameras()[np.random.randint(0, len(train_cameras))]
        render_feature_flag = (iteration < args.update_features_until_iter)
        render_pkg = render(viewpoint_cam, gaussians, args, background, render_features=render_feature_flag)
        
        image = render_pkg["render"]
        gt_image = viewpoint_cam.original_image.cuda()
        loss = (1.0 - args.lambda_dssim) * l1_loss(image, gt_image)
        
        if iteration < args.update_features_until_iter:
            rendered_feat = render_pkg["render_feat"]
            gt_clip = viewpoint_cam.feat_chw
            gt_dino = viewpoint_cam.dino_feat_chw
            
            rendered_feat_bhwc = F.interpolate(rendered_feat.unsqueeze(0), size=gt_clip.shape[1:], mode="bilinear", align_corners=False)
            dino_pred, clip_pred = my_feat_decoder(rendered_feat_bhwc)
            dino_pred = F.interpolate(dino_pred, size=gt_dino.shape[1:], mode="bilinear", align_corners=False).squeeze(0)
            clip_pred = clip_pred.squeeze(0)
            
            ignore_mask = (torch.sum(gt_clip == 0, dim=0) == gt_clip.shape[0])
            clip_loss = cosine_loss(clip_pred, gt_clip, mask=~ignore_mask)
            dino_loss = cosine_loss(dino_pred, gt_dino)
            loss += 0.1 * (0.1 * clip_loss + dino_loss)

        loss.backward()
        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == args.iterations:
                progress_bar.close()
            if iteration in args.save_iterations:
                scene.save(iteration)
                torch.save(my_feat_decoder.state_dict(), os.path.join(scene.model_path, "feat_decoder.pth"))
            
            if iteration < args.densify_until_iter:
                gaussians.max_radii2D[render_pkg["visibility_filter"]] = torch.max(gaussians.max_radii2D[render_pkg["visibility_filter"]], render_pkg["radii"][render_pkg["visibility_filter"]])
                gaussians.add_densification_stats(render_pkg["viewspace_points"], render_pkg["visibility_filter"])
                if iteration > args.densify_from_iter and iteration % args.densification_interval == 0:
                    size_threshold = 20 if iteration > args.opacity_reset_interval else None
                    gaussians.densify_and_prune(args.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                if iteration % args.opacity_reset_interval == 0 or (args.white_background and iteration == args.densify_from_iter):
                    gaussians.reset_opacity()
            
            if iteration < args.iterations:
                if iteration < args.update_decoder_until_iter:
                    decoder_optimizer.step()
                    decoder_optimizer.zero_grad()
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

    logger.success(f"==>> Reconstruction finished: {output_path}")
    return output_path
