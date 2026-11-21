#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
import torch.nn as nn, numpy as np, torch, math, h5py, cv2
from utils.graphics_utils import BasicPointCloud, getWorld2View2, getProjectionMatrix, fov2focal
from plyfile import PlyData, PlyElement
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.feat_decoder import feat_decoder, skip_feat_decoder
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from loguru import logger

class FileScene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "intrinsics.txt")):
            print("Found intrinsics.txt file, assuming Roger's custom plain format!")
            # TODO(roger): support eval
            scene_info = sceneLoadTypeCallbacks["Plain"](args.source_path)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
class H5Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FovX, FovY, image, gt_alpha_mask,
                 image_name, uid, trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda", 
                 gt_objects=None): 
        super(H5Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FovY = FovX
        self.FovX = FovY
        self.image_name = image_name

        self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        self.width = self.image_width
        self.height = self.image_height
        
        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
            self.mask = gt_alpha_mask.to(self.data_device)
            self.mask_valid_flag = True
        else:
            self.mask = None
            self.mask_valid_flag = False

        self.depth = None 
        self.depth_valid_mask = None

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=FovX, fovY=FovY).transpose(0, 1).cuda()
        
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        self.feat_chw = None      # CLIP features
        self.dino_feat_chw = None # DINO features
        self.part_feat_chw = None # Part-level features (if needed)
        
        if gt_objects is not None:
            self.objects = gt_objects.to(self.data_device)
        else:
            self.objects = None

class Scene:
    def __init__(self, args: ModelParams, gaussians: GaussianModel, shuffle=False, fx=414.5749, fy=414.5749, data=None):
        
        self.action_id = 0
        self.anchor_id = 0
        
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.white_background = args.white_background
        
        self.fx = fx
        self.fy = fy

        self.source_path = args.source_path 
        
        self.train_cameras = {}
        self.test_cameras = {}
        
        self.cam_list = []
        self.point_cloud_xyz = []
        self.point_cloud_rgb = []

        # ==========================================
        # ==========================================
        self._load_scene(data, args)

        if shuffle:
            random.shuffle(self.cam_list)

        if len(self.point_cloud_xyz) > 0:
            pcd_xyz = np.concatenate(self.point_cloud_xyz, axis=0)
            pcd_rgb = np.concatenate(self.point_cloud_rgb, axis=0)
        else:
            # Fallback
            print("Warning: No initial points found in H5 (no depth?), using random initialization.")
            pcd_xyz = np.random.random((10000, 3)) * 4 - 2
            pcd_rgb = np.random.random((10000, 3))

        self.cameras_extent = float(np.percentile(np.linalg.norm(pcd_xyz, axis=1), 90)) * 1.1
        # self.cameras_extent = 2
        logger.info(f"cameras_extent: {self.cameras_extent}")

        pcd = BasicPointCloud(points=pcd_xyz, colors=pcd_rgb, normals=np.zeros_like(pcd_xyz))

        self.train_cameras[1.0] = self.cam_list
        self.test_cameras[1.0] = []

        self.save_input_ply(os.path.join(self.model_path, "input.ply"), pcd)
        
        self.save_cameras_json(os.path.join(self.model_path, "cameras.json"), self.cam_list)
        
        # ==========================================
        # ==========================================
        self.gaussians.create_from_pcd(pcd, self.cameras_extent)

    def save_input_ply(self, path, pcd):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        xyz = pcd.points
        rgb = (pcd.colors * 255).astype(np.uint8)
        
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                 ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        
        elements = np.empty(xyz.shape[0], dtype=dtype)
        elements['x'] = xyz[:, 0]
        elements['y'] = xyz[:, 1]
        elements['z'] = xyz[:, 2]
        elements['red'] = rgb[:, 0]
        elements['green'] = rgb[:, 1]
        elements['blue'] = rgb[:, 2]
        
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        logger.success(f"==>> Saved input.ply to {path}")

    def save_cameras_json(self, path, cameras):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json_cams = []
        for id, cam in enumerate(cameras):
            json_cams.append(camera_to_JSON(id, cam))
        
        with open(path, 'w') as file:
            json.dump(json_cams, file)
        logger.success(f"==>> Saved cameras.json to {path}")
    
    def export_masks(self, output_root):
        action_dir = os.path.join(output_root, "action_masks")
        anchor_dir = os.path.join(output_root, "anchor_masks")
        os.makedirs(action_dir, exist_ok=True)
        os.makedirs(anchor_dir, exist_ok=True)
        
        print(f"Exporting masks to {output_root}...")
        
        for cam in self.cam_list:
            if not hasattr(cam, 'temp_objects_np') or cam.temp_objects_np is None:
                continue
                
            raw_mask = cam.temp_objects_np # numpy array (H, W)
            
            action_mask = np.zeros_like(raw_mask, dtype=np.uint8)
            action_mask[raw_mask == self.action_id] = 255
            
            # 2. Anchor Mask
            anchor_mask = np.zeros_like(raw_mask, dtype=np.uint8)
            anchor_mask[raw_mask == self.anchor_id] = 255
            
            img_name = f"{cam.image_name}.png"
            cv2.imwrite(os.path.join(action_dir, img_name), action_mask)
            cv2.imwrite(os.path.join(anchor_dir, img_name), anchor_mask)
            
            del cam.temp_objects_np
            
        print("Masks exported successfully.")
    
    def _load_scene(self, data, args):
        
        self.action_id = int(data['action_id'])
        self.anchor_id = int(data['anchor_id'])
        
        cam_group = data['key_state_000']['camera_data']
        camera_keys = sorted(list(cam_group.keys()))
        
        for idx, key in enumerate(camera_keys):
            cam_data = cam_group[key]
            
            rgb = cam_data['rgb'][()] # uint8 numpy
            
            if 'objs_mask' in cam_data:
                mask = cam_data['objs_mask'][()]
                gt_objects = mask.copy()
                mask_tensor = torch.from_numpy(mask).float()
                mask_tensor = (mask_tensor > 0).float().bool() # (H, W)
            else:
                mask_tensor = None

            image_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            image_tensor = mask_tensor.unsqueeze(0).float() * image_tensor if mask_tensor is not None else image_tensor
            
            H, W = image_tensor.shape[1], image_tensor.shape[2]

            c2w = cam_data['pose'][()]

            w2c = np.linalg.inv(c2w)
            
            R = w2c[:3, :3].transpose().astype(np.float32) 
            T = w2c[:3, 3].astype(np.float32)

            fovx = 2 * math.atan(W / (2 * self.fx))
            fovy = 2 * math.atan(H / (2 * self.fy))

            if 'depth' in cam_data:
                depth = cam_data['depth'][()]
                if idx % 2 == 0: 
                    depth = depth * (mask > 0)
                    self._project_depth_to_pcd(depth, c2w, W, H, rgb)

            cam = H5Camera(colmap_id=idx, R=R, T=T, FovX=fovx, FovY=fovy, 
                            image=image_tensor, gt_alpha_mask=mask_tensor,
                            image_name=key, uid=idx, data_device="cuda")
            
            cam.temp_objects_np = gt_objects
            
            if 'depth' in cam_data:
                depth_np = cam_data['depth'][()]
                cam.depth = torch.from_numpy(depth_np).cuda()
                cam.depth_valid_mask = (cam.depth > 0) & (cam.depth < 5.0)

            self.cam_list.append(cam)

    def _project_depth_to_pcd(self, depth, c2w, W, H, rgb):
        step = 1 
        ys, xs = np.indices(depth.shape)
        cx = W / 2.0
        cy = H / 2.0

        mask = (depth > 0.05) & (depth < 10.0)
        
        z = depth[mask][::step]
        x = (xs[mask][::step] - cx) * z / self.fx
        y = (ys[mask][::step] - cy) * z / self.fy
        
        xyz_cam = np.stack([x, y, z], axis=-1)
        
        # World = R * Cam + T -> Cam = R_inv * (World - T)
        xyz_world = (c2w[:3, :3] @ xyz_cam.T).T + c2w[:3, 3]
        
        rgb_vals = rgb[mask][::step] / 255.0
        
        self.point_cloud_xyz.append(xyz_world)
        self.point_cloud_rgb.append(rgb_vals)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]    

