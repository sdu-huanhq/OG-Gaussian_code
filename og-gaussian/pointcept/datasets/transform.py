"""
3D Point Cloud Augmentation

Inspirited by chrischoy/SpatioTemporalSegmentation

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import random
import numpy as np
import torch
import torch.nn.functional as F
from collections.abc import Sequence, Mapping
from typing import List
from scipy.spatial.transform import Rotation as R

from pointcept.utils.registry import Registry

TRANSFORMS = Registry("TRANSFORMS")


def rotate_quaternions(quats, Rot):
    r_trans = R.from_matrix(Rot)
    
    # Scipy expect: [x, y, z, w]
    # Input: [w, x, y, z]
    scipy_quats = quats[:, [1, 2, 3, 0]]
    r_old = R.from_quat(scipy_quats)
    
    r_new = r_trans * r_old
    
    new_scipy_quats = r_new.as_quat() # [x, y, z, w]
    new_quats = new_scipy_quats[:, [3, 0, 1, 2]] # -> [w, x, y, z]
    
    return new_quats.astype(np.float32)

def invert_SE3(mat):
    R = mat[:3, :3]
    t = mat[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv


@TRANSFORMS.register_module()
class CollectGS(object):
    def __init__(self, anc_keys, act_keys, gs_dim=59):
        self.anchor_keys = anc_keys
        self.action_keys = act_keys
        self.gs_dim = gs_dim

        assert not ('gs' in self.anchor_keys and 'coord' in self.anchor_keys), \
            f"Error: 'gs' and 'coord' cannot exist simultaneously in anchor_keys. Got: {self.anchor_keys}"
            
        assert not ('gs' in self.action_keys and 'coord' in self.action_keys), \
            f"Error: 'gs' and 'coord' cannot exist simultaneously in action_keys. Got: {self.action_keys}"

    def _collect_feats(self, data_dict, prefix, keys):
        feat_list = []
        
        core_source_key = f"{prefix}_feat"
        for k in keys:
            if k == 'coord':
                if core_source_key not in data_dict:
                    raise KeyError(f"Config '{k}' requires '{core_source_key}' in data_dict.")
                feat_list.append(data_dict[core_source_key][:, :3])
                
            elif k == 'gs':
                if core_source_key not in data_dict:
                    raise KeyError(f"Config '{k}' requires '{core_source_key}' in data_dict.")
                feat_list.append(data_dict[core_source_key][:, :self.gs_dim])
                
            else:
                target_key = f"{prefix}_{k}"
                if target_key not in data_dict:
                    raise KeyError(f"Key '{target_key}' expected by Collect config but not found.")
                feat_list.append(data_dict[target_key])
        
        if not feat_list:
            return None
            
        return np.concatenate(feat_list, axis=1)

    def __call__(self, data_dict):
        data = dict()

        data['anchor_coord'] = data_dict['anchor_feat'][:, :3]
        
        data['anchor_feat'] = self._collect_feats(data_dict, 'anchor', self.anchor_keys)
        
        data['anchor_offset'] = np.array([data['anchor_coord'].shape[0]], dtype=np.int32)
        
        if 'anchor_grid_coord' in data_dict:
            data['anchor_grid_coord'] = data_dict['anchor_grid_coord']
        if 'action_grid_coord' in data_dict:
            data['action_grid_coord'] = data_dict['action_grid_coord']
        # Pose
        data['anchor_pose'] = data_dict['anchor_init_pose']


        data['action_coord'] = data_dict['action_feat'][:, :3]
        
        data['action_feat'] = self._collect_feats(data_dict, 'action', self.action_keys)
        
        data['action_offset'] = np.array([data['action_coord'].shape[0]], dtype=np.int32)


        has_init = 'action_init_pose' in data_dict
        has_goal = 'action_goal_pose' in data_dict

        if has_init and has_goal:
            data['action_init_pose'] = data_dict['action_init_pose']
            data['action_goal_pose'] = data_dict['action_goal_pose']

        inv_init = invert_SE3(data_dict['action_init_pose'])
        data['action_transformation'] = data_dict['action_goal_pose'] @ inv_init

        data['action_goal_feat'] = data_dict['action_goal_feat']


        if 'skill_emb' in data_dict:
            data['skill_emb'] = data_dict['skill_emb']

        if 'shift' in data_dict:
            data['shift'] = data_dict['shift']
            
        if 'sample_name' in data_dict:
            data['sample_name'] = data_dict['sample_name']
            
        if 'action_gs_raw' in data_dict:
            data['action_gs_raw'] = data_dict['action_gs_raw'][:, :self.gs_dim]
        if 'anchor_gs_raw' in data_dict:
            data['anchor_gs_raw'] = data_dict['anchor_gs_raw'][:, :self.gs_dim]

        for key in ('normalization_center', 'normalization_scale'):
            if key in data_dict:
                data[key] = data_dict[key]
        
        if 'imagi_policy' in data_dict:
            data['imagi_policy'] = data_dict['imagi_policy']
        
        return data


@TRANSFORMS.register_module()
class GSNormalizeShared(object):
    """Shared geometry normalization and separate Euclidean encoder features."""

    def __init__(self, grid_resolution=384, is_activated=False):
        if is_activated:
            raise ValueError("GSNormalizeShared expects unactivated Gaussian attributes.")
        self.grid_resolution = grid_resolution

    @staticmethod
    def _encode(gs, log_rho):
        rotations = R.from_quat(gs[:, [4, 5, 6, 3]]).as_matrix()
        log_cov = np.einsum('nik,nk,njk->nij', rotations, 2 * gs[:, 7:10], rotations)
        # Symmetric half-vectorization: xx, yy, zz, xy, xz, yz.
        covariance_feat = log_cov[:, [0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]
        opacity = 0.5 * (np.tanh(0.5 * gs[:, 10:11]) + 1.0)
        return np.concatenate([
            gs[:, :3], covariance_feat, opacity, gs[:, 11:14],
            np.full((len(gs), 1), log_rho),
        ], axis=1).astype(np.float32)

    def __call__(self, data_dict):
        centers = np.concatenate([
            data_dict['action_feat'][:, :3], data_dict['anchor_feat'][:, :3],
        ]).astype(np.float64)
        center = centers.mean(axis=0)
        rho = max(np.sqrt(np.square(centers - center).sum(axis=1).mean()), 1e-8)
        log_rho = np.log(rho)
        data_dict['normalization_center'] = center[None].astype(np.float32)
        data_dict['normalization_scale'] = np.array([[rho]], dtype=np.float32)

        # The goal shares the initial pair's frame and never defines normalization.
        for key in ('action_feat', 'anchor_feat', 'action_goal_feat'):
            if key in data_dict:
                gs = data_dict[key].copy()
                gs[:, :3] = (gs[:, :3] - center) / rho
                gs[:, 7:10] -= log_rho
                data_dict[key] = gs

        normalized_centers = np.concatenate([
            data_dict['action_feat'][:, :3], data_dict['anchor_feat'][:, :3],
        ]).astype(np.float64)
        grid_origin = normalized_centers.min(axis=0)
        grid_extent = max(np.ptp(normalized_centers, axis=0).max(), 1e-8)
        for prefix in ('action', 'anchor'):
            gs = data_dict[f'{prefix}_feat']
            # "raw" denotes the unactivated attribute layout, now in the shared frame.
            data_dict[f'{prefix}_gs_raw'] = gs.copy()
            data_dict[f'{prefix}_encoding'] = self._encode(gs, log_rho)
            data_dict[f'{prefix}_grid_coord'] = np.floor(
                (gs[:, :3] - grid_origin) / grid_extent * (self.grid_resolution - 1)
            ).astype(np.int32)
        return data_dict


@TRANSFORMS.register_module()
class ToTensor(object):
    def __call__(self, data):
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, str):
            # note that str is also a kind of sequence, judgement should before sequence
            return data
        elif isinstance(data, int):
            return torch.LongTensor([data])
        elif isinstance(data, float):
            return torch.FloatTensor([data])
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, bool):
            return torch.from_numpy(data)
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.integer):
            return torch.from_numpy(data).long()
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.floating):
            return torch.from_numpy(data).float()
        elif isinstance(data, Mapping):
            result = {sub_key: self(item) for sub_key, item in data.items()}
            return result
        elif isinstance(data, Sequence):
            result = [self(item) for item in data]
            return result
        else:
            raise TypeError(f"type {type(data)} cannot be converted to tensor.")







#* =========================== tfeat transform (end) ===========================


@TRANSFORMS.register_module()
class CenterShift(object):
    def __init__(self, apply_z=True):
        self.apply_z = apply_z

    def __call__(self, data_dict):
        anchor_coord = data_dict['anchor_feat'][:, :3]
        action_coord = data_dict['action_feat'][:, :3]
        sx_min, sy_min, sz_min = anchor_coord.min(axis=0)
        ox_min, oy_min, oz_min = action_coord.min(axis=0)
        x_min, y_min, z_min = min(sx_min, ox_min), min(sy_min, oy_min), min(sz_min, oz_min)
        sx_max, sy_max, _ = anchor_coord.max(axis=0)
        ox_max, oy_max, _ = action_coord.max(axis=0)
        x_max, y_max = max(sx_max, ox_max), max(sy_max, oy_max)
        if self.apply_z:
            shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, z_min]
        else:
            shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, 0]
            
        data_dict['anchor_feat'][:, :3] -= shift
        data_dict['action_feat'][:, :3] -= shift
        
        if 'action_goal_feat' in data_dict:
            data_dict["action_goal_feat"][:, :3] -= shift
        data_dict["action_init_pose"][:3, 3] -= shift
        data_dict["action_goal_pose"][:3, 3] -= shift
        data_dict['anchor_init_pose'][:3, 3] -= shift
        
        if 'shift' in data_dict:
            data_dict['shift'] += np.array(shift)
        else:
            data_dict['shift'] = np.array(shift)
        
        
        return data_dict


@TRANSFORMS.register_module()
class RandomShift(object):
    def __init__(self, shift=((-0.2, 0.2), (-0.2, 0.2), (0, 0)), apply_target=None):
        self.shift = np.array(shift)
        self.apply_target = apply_target    # "scene"/"object"

    def __call__(self, data_dict):
        shift = np.random.uniform(self.shift[:, 0], self.shift[:, 1])
        if self.apply_target == 'action':
            if data_dict["move_link"]:
                return data_dict
            data_dict['action_feat'][:, :3] += shift
            data_dict["action_init_pose"][:3, 3] += shift

        
        elif self.apply_target == 'anchor':
            if data_dict["move_link"]:
                return data_dict
            data_dict['anchor_feat'][:, :3] += shift
            data_dict["anchor_init_pose"][:3, 3] += shift
            data_dict["action_goal_feat"][:, :3] += shift
            data_dict["action_goal_pose"][:3, 3] += shift
            
            
        else:
            raise NotImplementedError

        
        return data_dict


@TRANSFORMS.register_module()
class RandomDropout(object):
    def __init__(self, dropout_ratio: List, dropout_application_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.dropout_application_ratio = dropout_application_ratio

    def __call__(self, data_dict):
        if random.random() >= self.dropout_application_ratio:
            return data_dict
        
        ratio = np.random.uniform(self.dropout_ratio[0], self.dropout_ratio[1])
        
        n1 = data_dict["anchor_feat"].shape[0]
        keep_n1 = max(int(n1 * (1 - ratio)), 1)
        
        idx1 = np.random.choice(n1, keep_n1, replace=False)
        data_dict['anchor_feat'] = data_dict['anchor_feat'][idx1]
        data_dict['anchor_vfeat'] = data_dict['anchor_vfeat'][idx1]
        data_dict['anchor_tfeat'] = data_dict['anchor_tfeat'][idx1]

        n2 = data_dict["action_feat"].shape[0]
        keep_n2 = max(int(n2 * (1 - ratio)), 1)
        
        idx2 = np.random.choice(n2, keep_n2, replace=False)
        data_dict['action_feat'] = data_dict['action_feat'][idx2]
        data_dict['action_vfeat'] = data_dict['action_vfeat'][idx2]
        data_dict['action_tfeat'] = data_dict['action_tfeat'][idx2]
        data_dict["action_goal_feat"] = data_dict["action_goal_feat"][idx2]
        
            
        return data_dict
        
        
@TRANSFORMS.register_module()
class RandomRotate(object):
    def __init__(self, angle=None, center=None, axis="z", always_apply=False, p=0.5, apply_target=None):
        self.angle = [-1, 1] if angle is None else angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center
        self.apply_target = apply_target

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        angle = np.random.uniform(self.angle[0], self.angle[1]) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        if self.axis == "x":
            rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
        elif self.axis == "y":
            rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
        elif self.axis == "z":
            rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        else:
            raise NotImplementedError
        
        if self.apply_target == 'all':
            if data_dict["move_link"]:
                if self.center is None:
                    act_xyz = data_dict['action_feat'][:, :3]
                    anc_xyz = data_dict['anchor_feat'][:, :3]
                    act_quat = data_dict['action_feat'][:, 3:7]
                    anc_quat = data_dict['anchor_feat'][:, 3:7]
                    
                    if self.center is None:
                        min_xyz = np.minimum(act_xyz.min(axis=0), anc_xyz.min(axis=0))
                        max_xyz = np.maximum(act_xyz.max(axis=0), anc_xyz.max(axis=0))
                        center = (min_xyz + max_xyz) / 2.0
                    else:
                        center = self.center
                    act_xyz -= center
                    act_xyz[:] = act_xyz @ rot_t.T
                    act_xyz += center
                    act_quat[:] = rotate_quaternions(act_quat, rot_t)
                    
                    anc_xyz -= center
                    anc_xyz[:] = anc_xyz @ rot_t.T
                    anc_xyz += center
                    anc_quat[:] = rotate_quaternions(anc_quat, rot_t)
                    
                    act_init_pose = data_dict["action_init_pose"]
                    act_init_pose[:3, 3] -= center
                    mat_t = np.eye(4)
                    mat_t[:3, :3] = rot_t
                    act_init_pose[:] = mat_t @ act_init_pose
                    act_init_pose[:3, 3] += center
                    
                    act_goal_pose = data_dict["action_goal_pose"]
                    act_goal_pose[:3, 3] -= center
                    act_goal_pose[:] = mat_t @ act_goal_pose
                    act_goal_pose[:3, 3] += center
                    
                    act_goal = data_dict["action_goal_feat"]
                    act_goal[:, :3] -= center
                    act_goal[:, :3] = act_goal[:, :3] @ rot_t.T
                    act_goal[:, :3] += center
            return data_dict
        
        elif self.apply_target == 'action':
            if data_dict["move_link"]:
                return data_dict
            act_xyz = data_dict['action_feat'][:, :3]
            act_quat = data_dict['action_feat'][:, 3:7]
            if self.center is None:
                center = (act_xyz.min(axis=0) + act_xyz.max(axis=0)) / 2
            else:
                center = self.center
            act_xyz -= center
            act_xyz[:] = act_xyz @ rot_t.T
            act_xyz += center
            act_quat[:] = rotate_quaternions(act_quat, rot_t)
            init_pose = data_dict["action_init_pose"]
            init_pose[:3, 3] -= center
            
            mat_t = np.eye(4)
            mat_t[:3, :3] = rot_t
            
            init_pose[:] = mat_t @ init_pose
            init_pose[:3, 3] += center
            
            return data_dict
        
        elif self.apply_target == 'anchor':
            if data_dict["move_link"]:
                return data_dict
            anc_xyz = data_dict['anchor_feat'][:, :3]
            anc_quat = data_dict['anchor_feat'][:, 3:7]
            if self.center is None:
                center = (anc_xyz.min(axis=0) + anc_xyz.max(axis=0)) / 2
            else:
                center = self.center
            anc_xyz -= center
            anc_xyz[:] = anc_xyz @ rot_t.T
            anc_xyz += center
            anc_quat[:] = rotate_quaternions(anc_quat, rot_t)
            init_pose = data_dict["anchor_init_pose"]
            init_pose[:3, 3] -= center
            
            mat_t = np.eye(4)
            mat_t[:3, :3] = rot_t
            
            init_pose[:] = mat_t @ init_pose
            init_pose[:3, 3] += center
                
            
            goal_xyz = data_dict["action_goal_feat"][:, :3]
            goal_quat = data_dict["action_goal_feat"][:, 3:7]
            
            goal_xyz -= center                             
            goal_xyz[:] = np.dot(goal_xyz, np.transpose(rot_t))
            goal_xyz += center
            goal_quat[:] = rotate_quaternions(goal_quat, rot_t)
                
                
            action_goal_pose = data_dict["action_goal_pose"]
            action_goal_pose[:3, 3] -= center
            mat_t = np.eye(4)
            mat_t[:3, :3] = rot_t
            action_goal_pose[:] = np.dot(mat_t, action_goal_pose)
            action_goal_pose[:3, 3] += center
            return data_dict
        
        else:
            raise NotImplementedError


@TRANSFORMS.register_module()
class RandomScale(object):
    def __init__(self, scale=None, is_activated=False):
        self.scale = scale if scale is not None else [0.95, 1.05]
        self.is_activated = is_activated

    def __call__(self, data_dict):
        scale = np.random.uniform(
            self.scale[0], self.scale[1], 1
        )
        
        feat_keys = ['anchor_feat', 'action_feat', 'action_goal_feat']
        if self.is_activated:
            for key in feat_keys:
                data_dict[key][:, :3] *= scale
                data_dict[key][:, 7:10] *= scale 
        else:
            scale_log = np.log(scale)
            for key in feat_keys:
                data_dict[key][:, :3] *= scale
                data_dict[key][:, 7:10] += scale_log
        data_dict["anchor_init_pose"][:3, 3] *= scale
        data_dict["action_init_pose"][:3, 3] *= scale
        data_dict["action_goal_pose"][:3, 3] *= scale
            
        
        return data_dict


@TRANSFORMS.register_module()
class RandomFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data_dict):
        feat_keys = ["anchor_feat", "action_feat", "action_goal_feat"]
        pose_keys = ["anchor_init_pose", "action_init_pose", "action_goal_pose"]
        
        for axis_idx in [0, 1]:
            if np.random.rand() < self.p:
                for key in feat_keys:
                    data_dict[key][:, axis_idx] *= -1
                    
                    if axis_idx == 0: # Flip X -> invert Y and Z components of Quat
                        data_dict[key][:, [5, 6]] *= -1 
                    elif axis_idx == 1: # Flip Y -> invert X and Z components of Quat
                        data_dict[key][:, [4, 6]] *= -1

                flip_diag = [1, 1, 1, 1]
                flip_diag[axis_idx] = -1
                F = np.diag(flip_diag)

                for key in pose_keys:
                    data_dict[key] = F @ data_dict[key]
        
        return data_dict


@TRANSFORMS.register_module()
class RandomJitter(object):
    def __init__(self, sigma=0.01, clip=0.05):
        assert clip > 0
        self.sigma = sigma
        self.clip = clip

    def __call__(self, data_dict):
        num_anchor = data_dict["anchor_feat"].shape[0] 
        num_action = data_dict["action_feat"].shape[0]
        total_size = num_anchor + num_action
        jitter = np.clip(
            self.sigma * np.random.randn(total_size, 3),
            -self.clip,
            self.clip,
        )
        
        data_dict["anchor_feat"][:, :3] += jitter[:num_anchor]
        data_dict["action_feat"][:, :3] += jitter[num_anchor:]
        
        data_dict["action_goal_feat"][:, :3] += jitter[num_anchor:]

        
        return data_dict
    
    
@TRANSFORMS.register_module()
class GridSampleGS1(object):
    def __init__(
        self,
        min_grid_size=0.01,
        grid_size_increment=0.01,
        max_grid_count=1000,
        hash_type="fnv",
        mode="train"
    ):
        self.min_grid_size = min_grid_size
        self.max_grid_count = max_grid_count
        self.grid_size_increment = grid_size_increment
        self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec
        assert mode in ["train", "test"]
        self.mode = mode

    def __call__(self, data_dict):
        scene_count = np.arange(self.max_grid_count+1)
        grid_size = self.min_grid_size
        while len(scene_count) > self.max_grid_count:
            # scene coord
            scene_scaled_coord = data_dict["anchor_feat"][:, :3] / np.array(grid_size)
            scene_grid_coord = np.floor(scene_scaled_coord).astype(int)
            scene_min_coord = scene_grid_coord.min(0)
            scene_grid_coord -= scene_min_coord 
            scene_min_coord = scene_min_coord * np.array(grid_size)
            scene_key = self.hash(scene_grid_coord)
            scene_idx_sort = np.argsort(scene_key)
            scene_key_sort = scene_key[scene_idx_sort]
            
            _, scene_inverse, scene_count = np.unique(scene_key_sort, return_inverse=True, return_counts=True)
            grid_size += self.grid_size_increment
            
        if self.mode == "train":
            scene_idx_select = (
                np.cumsum(np.insert(scene_count, 0, 0)[0:-1])
                + np.random.randint(0, scene_count.max(), scene_count.size) % scene_count
            )
            scene_idx_unique = scene_idx_sort[scene_idx_select]
            data_dict["anchor_grid_coord"] = scene_grid_coord[scene_idx_unique]
            data_dict["anchor_feat"] = data_dict["anchor_feat"][scene_idx_unique]
            data_dict["anchor_vfeat"] = data_dict["anchor_vfeat"][scene_idx_unique]
            data_dict["anchor_tfeat"] = data_dict["anchor_tfeat"][scene_idx_unique]
            return data_dict
        
        elif self.mode == "test":
            # calculate mean
            original_anchor_feat = data_dict["anchor_feat"]
            scene_idx_select = (
                np.cumsum(np.insert(scene_count, 0, 0)[0:-1])
                + np.random.randint(0, scene_count.max(), scene_count.size) % scene_count
            )
            scene_idx_unique = scene_idx_sort[scene_idx_select]
            data_dict["anchor_grid_coord"] = scene_grid_coord[scene_idx_unique]
            keys = ["anchor_feat", "anchor_vfeat", "anchor_tfeat"]
            num_voxels = scene_count.shape[0]
            for k in keys:
                coord_sum = np.zeros((num_voxels, data_dict[k].shape[1]), dtype=np.float32)
                idx = np.empty(data_dict[k].shape[0], dtype=int)
                idx[scene_idx_sort] = scene_inverse
                np.add.at(coord_sum, idx, data_dict[k])
                data_dict[k] = coord_sum / scene_count[:, np.newaxis]

            # Preserve all existing means unless a quaternion cancels exactly to zero.
            mean_quaternions = data_dict["anchor_feat"][:, 3:7]
            zero_voxels = np.flatnonzero(np.all(mean_quaternions == 0, axis=1))
            if zero_voxels.size:
                voxel_offsets = np.concatenate(([0], np.cumsum(scene_count)))
                for voxel in zero_voxels:
                    source_indices = scene_idx_sort[voxel_offsets[voxel]:voxel_offsets[voxel + 1]]
                    quaternions = original_anchor_feat[source_indices, 3:7].astype(np.float64)
                    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
                    if not np.isfinite(norms).all() or np.any(norms == 0):
                        raise ValueError(
                            f"Cannot repair voxel {voxel}: original quaternions are invalid."
                        )
                    quaternions /= norms
                    reference = quaternions[0].copy()
                    quaternions[(quaternions @ reference) < 0] *= -1
                    mean = quaternions.mean(axis=0)
                    norm = np.linalg.norm(mean)
                    if not np.isfinite(norm) or norm == 0:
                        raise ValueError(f"Cannot repair zero quaternion in voxel {voxel}.")
                    mean_quaternions[voxel] = mean / norm
            return data_dict
        
        else:
            raise NotImplementedError

    @staticmethod
    def ravel_hash_vec(arr):
        """
        Ravel the coordinates after subtracting the min coordinates.
        """
        assert arr.ndim == 2
        arr = arr.copy()
        arr -= arr.min(0)
        arr = arr.astype(np.uint64, copy=False)
        arr_max = arr.max(0).astype(np.uint64) + 1

        keys = np.zeros(arr.shape[0], dtype=np.uint64)
        # Fortran style indexing
        for j in range(arr.shape[1] - 1):
            keys += arr[:, j]
            keys *= arr_max[j + 1]
        keys += arr[:, -1]
        return keys

    @staticmethod
    def fnv_hash_vec(arr):
        """
        FNV64-1A
        """
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(
            arr.shape[0], dtype=np.uint64
        )
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr


class Compose(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.transforms = []
        for t_cfg in self.cfg:
            self.transforms.append(TRANSFORMS.build(t_cfg))

    def __call__(self, data_dict):
        for t in self.transforms:
            data_dict = t(data_dict)
        return data_dict

SH_C0 = 0.28209479177387814
