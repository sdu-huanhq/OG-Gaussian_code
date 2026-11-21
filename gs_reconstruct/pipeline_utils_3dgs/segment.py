import os
import sys
import yaml
import torch
import numpy as np
import cv2
import json
import math
import torch.nn.functional as F
from loguru import logger

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from scene import GaussianModel
from scene.cameras import Camera
from utils.graphics_utils import focal2fov
from flashsplat_rasterization import GaussianRasterizationSettings as FlashSplat_GaussianRasterizationSettings
from flashsplat_rasterization import GaussianRasterizer as FlashSplat_GaussianRasterizer
from plyfile import PlyData, PlyElement

def flashsplat_render(viewpoint_camera, pc, bg_color, gt_mask=None, used_mask=None, obj_num=2):
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda")
    
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    raster_settings = FlashSplat_GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx, tanfovy=tanfovy, bg=bg_color,
        scale_modifier=1.0, viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree, campos=viewpoint_camera.camera_center,
        prefiltered=False, debug=False, mask_grad=False, num_obj=obj_num
    )
    rasterizer = FlashSplat_GaussianRasterizer(raster_settings=raster_settings)
    
    means3D = pc.get_xyz
    opacity = pc.get_opacity
    shs = pc.get_features
    scales = pc.get_scaling
    rotations = pc.get_rotation
    
    if used_mask is not None:
        mask_bool = used_mask.bool()
        means3D = means3D[mask_bool]
        opacity = opacity[mask_bool]
        shs = shs[mask_bool]
        scales = scales[mask_bool]
        rotations = rotations[mask_bool]

    return rasterizer(
        gt_mask=gt_mask, unique_label=None, means3D=means3D, means2D=screenspace_points,
        shs=shs, colors_precomp=None, opacities=opacity, scales=scales, rotations=rotations, cov3D_precomp=None
    )[5]

def load_cameras(model_path, device="cuda"):
    json_path = os.path.join(model_path, "cameras.json")
    if not os.path.exists(json_path):
        return []
    with open(json_path, 'r') as f:
        cam_infos = json.load(f)
    cameras = []
    for cam_info in cam_infos:
        width, height = int(cam_info['width']), int(cam_info['height'])
        c2w = np.zeros((4,4))
        c2w[:3,:3] = np.array(cam_info['rotation'])
        c2w[:3,3] = np.array(cam_info['position'])
        c2w[3,3] = 1.0
        w2c = np.linalg.inv(c2w)
        R = w2c[:3,:3].transpose().astype(np.float32)
        T = w2c[:3,3].astype(np.float32)
        fovx = focal2fov(cam_info['fx'], width)
        fovy = focal2fov(cam_info['fy'], height)
        cam = Camera(colmap_id=cam_info['id'], R=R, T=T, FoVx=fovx, FoVy=fovy,
                     image=torch.zeros((3, height, width), device=device), 
                     gt_alpha_mask=None, image_name=cam_info['img_name'], uid=cam_info['id'], data_device=device)
        cameras.append(cam)
    return cameras

def save_ply_subset(gaussians, mask, path, distill_feats):
    if mask.sum() == 0:
        logger.warning(f"Empty mask for {path}, skipping.")
        return path, None

    xyz = gaussians._xyz[mask].detach().cpu().numpy()
    f_dc = gaussians._features_dc[mask].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = gaussians._features_rest[mask].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = gaussians._opacity[mask].detach().cpu().numpy()
    scale = gaussians._scaling[mask].detach().cpu().numpy()
    rotation = gaussians._rotation[mask].detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    for i in range(f_dc.shape[1]): l.append(f'f_dc_{i}')
    for i in range(f_rest.shape[1]): l.append(f'f_rest_{i}')
    l.append('opacity')
    for i in range(scale.shape[1]): l.append(f'scale_{i}')
    for i in range(rotation.shape[1]): l.append(f'rot_{i}')
    
    dtype_full = [(n, 'f4') for n in l]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attrs = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attrs))
    PlyData([PlyElement.describe(elements, 'vertex')]).write(path)
    
    feat_path = path.replace('.ply', '_distill_feat.pth')
    if distill_feats is not None and distill_feats.shape[0] == mask.shape[0]:
         torch.save(distill_feats[mask].cpu(), feat_path)
    return path, feat_path

def run_segmentation(cfg, model_path, device_id):
    torch.cuda.set_device(device_id)
    device = f"cuda:{device_id}"
    
    iteration = cfg.get('iteration', 3000)
    point_cloud_dir = os.path.join(model_path, "point_cloud", f"iteration_{iteration}")
    ply_path = os.path.join(point_cloud_dir, "point_cloud.ply")
    feat_path = os.path.join(point_cloud_dir, "point_cloud_distill_feat.pth")
    
    if not os.path.exists(ply_path):
        return None

    gaussians = GaussianModel(3, 32)
    gaussians.load_ply(ply_path)
    
    distill_feats = None
    if os.path.exists(feat_path):
        distill_feats = torch.load(feat_path, map_location=device)
    
    cameras = load_cameras(model_path, device=device)
    
    cameras_with_masks = []
    action_mask_dir = os.path.join(model_path, "action_masks")
    anchor_mask_dir = os.path.join(model_path, "anchor_masks")
    
    for cam in cameras:
        mask_name = f"{cam.image_name}.png"
        act_p = os.path.join(action_mask_dir, mask_name)
        anc_p = os.path.join(anchor_mask_dir, mask_name)
        
        if not os.path.exists(act_p) and not os.path.exists(anc_p): continue
        
        m_act = cv2.imread(act_p, 0)
        m_anc = cv2.imread(anc_p, 0)
        
        if m_act is None: m_act = np.zeros((cam.image_height, cam.image_width), dtype=np.uint8)
        if m_anc is None: m_anc = np.zeros((cam.image_height, cam.image_width), dtype=np.uint8)
        
        if m_act.shape != (cam.image_height, cam.image_width):
             m_act = cv2.resize(m_act, (cam.image_width, cam.image_height), interpolation=cv2.INTER_NEAREST)
        if m_anc.shape != (cam.image_height, cam.image_width):
             m_anc = cv2.resize(m_anc, (cam.image_width, cam.image_height), interpolation=cv2.INTER_NEAREST)

        combined = np.zeros_like(m_act, dtype=np.int64)
        combined[m_act > 127] = 1
        combined[m_anc > 127] = 2
        
        cam.objects = torch.from_numpy(combined).long().to(device)
        cameras_with_masks.append(cam)
    
    if not cameras_with_masks:
        return None

    num_objs = 3 
    all_counts = torch.zeros((num_objs+1, gaussians.get_xyz.shape[0]), device=device)
    bg_color = torch.tensor([0,0,0], dtype=torch.float32, device=device)
    
    for cam in cameras_with_masks:
        counts = flashsplat_render(cam, gaussians, bg_color, gt_mask=cam.objects.float(), obj_num=num_objs)
        all_counts += counts
        
    all_contrib_sum = all_counts.sum(dim=0)
    gamma = cfg.get('slackness', 0.7)
    labels = torch.zeros_like(all_counts).bool()
    
    mask_valid = all_contrib_sum > 0
    if mask_valid.any():
        for i in range(len(all_counts)):
            obj_contrib = torch.zeros((2, all_counts.shape[1]), device=device)
            obj_contrib[0, mask_valid] = all_contrib_sum[mask_valid] - all_counts[i, mask_valid]
            obj_contrib[1, mask_valid] = all_counts[i, mask_valid]
            
            obj_contrib = F.normalize(obj_contrib, dim=0)
            obj_contrib[0, :] += gamma
            labels[i] = torch.argmax(obj_contrib, dim=0)
        
    action_mask = labels[1]
    anchor_mask = labels[2]
    
    res_dir = os.path.join(model_path, "segmentation_results")
    os.makedirs(res_dir, exist_ok=True)
    
    act_ply, act_feat = save_ply_subset(gaussians, action_mask, os.path.join(res_dir, "action.ply"), distill_feats)
    anc_ply, anc_feat = save_ply_subset(gaussians, anchor_mask, os.path.join(res_dir, "anchor.ply"), distill_feats)
    
    return {"action": (act_ply, act_feat), "anchor": (anc_ply, anc_feat)}