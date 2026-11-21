import os, math
import torch.nn.functional as F
import sys
import argparse
import yaml
import torch
import numpy as np
import h5py
import pickle
from loguru import logger
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R
from loguru import logger

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

def transPts_T(pts, T):
    R = T[:3, :3]
    t = T[:3, 3]
    # (N, 3) x (3, 3) + (3,)
    return (pts @ R.T) + t

def transform_quaternions(quats, T):
    rot_mat = T[:3, :3]
    r_trans = R.from_matrix(rot_mat)
    
    # Scipy expect: [x, y, z, w]
    # Input: [w, x, y, z]
    scipy_quats = quats[:, [1, 2, 3, 0]]
    r_old = R.from_quat(scipy_quats)
    
    r_new = r_trans * r_old
    
    new_scipy_quats = r_new.as_quat() # [x, y, z, w]
    new_quats = new_scipy_quats[:, [3, 0, 1, 2]] # -> [w, x, y, z]
    
    return new_quats.astype(np.float32)

def load_ply_data(ply_path, feat_path):
    plydata = PlyData.read(ply_path)
    e = plydata.elements[0]
    
    xyz = np.stack([e['x'], e['y'], e['z']], axis=1)
    f_dc = np.stack([e['f_dc_0'], e['f_dc_1'], e['f_dc_2']], axis=1)
    
    rest_names = [n.name for n in e.properties if n.name.startswith('f_rest_')]
    rest_names.sort(key=lambda x: int(x.split('_')[-1]))
    f_rest = np.stack([e[n] for n in rest_names], axis=1)
    
    opacities = e['opacity'][..., None]
    scales = np.stack([e[f'scale_{i}'] for i in range(3)], axis=1)
    rots = np.stack([e[f'rot_{i}'] for i in range(4)], axis=1)
    
    distilled = torch.load(feat_path, map_location='cpu').numpy() if feat_path and os.path.exists(feat_path) else None
    if distilled is None:
        # Fallback if no distilled features
        distilled = np.zeros((xyz.shape[0], 32), dtype=np.float32)
    
    return xyz, f_dc, f_rest, opacities, scales, rots, distilled

def furthest_point_sample(xyz, npoint):
    B, N, C = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def scale_aware_sampling(xyz_np, min_points, max_points, device="cuda"):
    num_points = xyz_np.shape[0]
    
    if num_points > max_points:
        xyz = torch.from_numpy(xyz_np).float().to(device)
        center = xyz.mean(dim=0)
        xyz_centered = xyz - center
        
        fps_idx = furthest_point_sample(xyz_centered.unsqueeze(0), max_points).squeeze(0)
        return fps_idx.cpu().numpy()
        
    elif num_points < min_points:
        if num_points == 0:
            logger.error("num_points == 0")
        logger.warning(f"Point count {num_points} less than min_points {min_points}, applying repeat upsampling.")
        base_idx = np.arange(num_points)
        num_pad = min_points - num_points
        
        pad_idx = np.random.choice(base_idx, num_pad, replace=True)
        
        final_idx = np.concatenate([base_idx, pad_idx])
        return final_idx
        
    else:
        return np.arange(num_points)

def pose_7d_to_4x4(pose_7d):
    t = np.array(pose_7d[:3])
    q = np.array(pose_7d[3:])
    qx, qy, qz, qw = q[0], q[1], q[2], q[3]
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    mat = np.eye(4)
    mat[:3, :3] = R
    mat[:3, 3] = t
    return mat

def save_sampled_ply(path, xyz, f_dc, f_rest, opacity, scale, rot):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    for i in range(f_dc.shape[1]): l.append(f'f_dc_{i}')
    for i in range(f_rest.shape[1]): l.append(f'f_rest_{i}')
    l.append('opacity')
    for i in range(scale.shape[1]): l.append(f'scale_{i}')
    for i in range(rot.shape[1]): l.append(f'rot_{i}')

    dtype_full = [(n, 'f4') for n in l]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    normals = np.zeros_like(xyz)
    attrs = np.concatenate((xyz, normals, f_dc, f_rest, opacity, scale, rot), axis=1)
    elements[:] = list(map(tuple, attrs))
    PlyData([PlyElement.describe(elements, 'vertex')]).write(path)

def process_sampling_and_h5(cfg, seg_results, pkl_data, output_h5_path, device_id):
    torch.cuda.set_device(device_id)
    
    act_path, act_feat_path = seg_results['action']
    anc_path, anc_feat_path = seg_results['anchor']
    
    if not os.path.exists(act_path) or not os.path.exists(anc_path):
        return

    act_data = load_ply_data(act_path, act_feat_path)
    anc_data = load_ply_data(anc_path, anc_feat_path)
    
    sampling_cfg = cfg.get('sampling', {})
    act_min = sampling_cfg.get('action_min_points', 2048)
    act_max = sampling_cfg.get('action_max_points', 4096)
    anc_min = sampling_cfg.get('anchor_min_points', 1024)
    anc_max = sampling_cfg.get('anchor_max_points', 2048)
    act_idx = scale_aware_sampling(act_data[0], act_min, act_max, device="cuda")
    anc_idx = scale_aware_sampling(anc_data[0], anc_min, anc_max, device="cuda")
    
    def extract_subset(data_tuple, idx):
        return [d[idx] for d in data_tuple]
        
    s_act = extract_subset(act_data, act_idx)
    s_anc = extract_subset(anc_data, anc_idx)
    
    output_dir = os.path.dirname(output_h5_path)
    scene_name = os.path.splitext(os.path.basename(output_h5_path))[0]
    sampled_save_dir = cfg.get('intermediate_dir', output_dir)
    sampled_save_dir = os.path.join(sampled_save_dir, scene_name, "sampled_gs")
    os.makedirs(sampled_save_dir, exist_ok=True)
    
    act_ply_name = os.path.join(sampled_save_dir, f"action_sampled.ply")
    act_feat_name = os.path.join(sampled_save_dir, f"action_sampled_feat.pth")
    anc_ply_name = os.path.join(sampled_save_dir, f"anchor_sampled.ply")
    anc_feat_name = os.path.join(sampled_save_dir, f"anchor_sampled_feat.pth")
    
    save_sampled_ply(act_ply_name, s_act[0], s_act[1], s_act[2], s_act[3], s_act[4], s_act[5])
    torch.save(torch.from_numpy(s_act[6]), act_feat_name)
    
    save_sampled_ply(anc_ply_name, s_anc[0], s_anc[1], s_anc[2], s_anc[3], s_anc[4], s_anc[5])
    torch.save(torch.from_numpy(s_anc[6]), anc_feat_name)
    
    action_init_7d = pkl_data['key_state_000']['obj_pose']
    action_goal_7d = pkl_data['key_state_001']['obj_pose']
    anchor_name = str(pkl_data['anchor_name'])
    anchor_init_7d = pkl_data['key_state_000']['objs_pose'][anchor_name]
    
    act_init_pose = pose_7d_to_4x4(action_init_7d)
    act_goal_pose = pose_7d_to_4x4(action_goal_7d)
    anc_init_pose = pose_7d_to_4x4(anchor_init_7d)
    

    T_rel = act_goal_pose @ np.linalg.inv(act_init_pose)
    s_act_source = transPts_T(s_act[0], np.linalg.inv(act_init_pose))
    action_goal_xyz = transPts_T(s_act_source, act_goal_pose)
    action_goal_rot = transform_quaternions(s_act[5], T_rel)
    
    
    merged_ply_name = os.path.join(sampled_save_dir, f"merged_goal_state.ply")
    
    merged_xyz = np.concatenate([action_goal_xyz, s_anc[0]], axis=0)
    merged_f_dc = np.concatenate([s_act[1], s_anc[1]], axis=0)
    merged_f_rest = np.concatenate([s_act[2], s_anc[2]], axis=0)
    merged_opacity = np.concatenate([s_act[3], s_anc[3]], axis=0)
    merged_scale = np.concatenate([s_act[4], s_anc[4]], axis=0)
    merged_rot = np.concatenate([action_goal_rot, s_anc[5]], axis=0)
    
    save_sampled_ply(
        merged_ply_name, 
        merged_xyz, merged_f_dc, merged_f_rest, 
        merged_opacity, merged_scale, merged_rot
    )
    
    init_merged_xyz = np.concatenate([s_act[0], s_anc[0]], axis=0)
    init_merged_f_dc = np.concatenate([s_act[1], s_anc[1]], axis=0)
    init_merged_f_rest = np.concatenate([s_act[2], s_anc[2]], axis=0)
    init_merged_opacity = np.concatenate([s_act[3], s_anc[3]], axis=0)
    init_merged_scale = np.concatenate([s_act[4], s_anc[4]], axis=0)
    init_merged_rot = np.concatenate([s_act[5], s_anc[5]], axis=0)
    
    init_merged_ply_name = os.path.join(sampled_save_dir, f"merged_init_state.ply")
    
    save_sampled_ply(
        init_merged_ply_name, 
        init_merged_xyz, init_merged_f_dc, init_merged_f_rest, 
        init_merged_opacity, init_merged_scale, init_merged_rot
    )
    
    with h5py.File(output_h5_path, 'w') as f:
        f.create_dataset('skill_name', data=str(pkl_data.get('skill_name', 'unknown')))
        f.create_dataset('camera_intrinsic', data=str(pkl_data.get('camera_intrinsic', 'unknown')))
        
        # --- Action Group ---
        g_act = f.create_group('action')
        act_gs_feat = np.concatenate([
            s_act[0],  # xyz
            s_act[5],  # rot (pose)
            s_act[4],  # scale
            s_act[3],   # opacity
            s_act[1],  # f_dc
            s_act[2]  # f_rest
        ], axis=1).astype(np.float32)
        
        act_gs_goal_feat = np.concatenate([
            action_goal_xyz,  # xyz
            action_goal_rot,  # rot (pose)
            s_act[4],  # scale
            s_act[3],   # opacity
            s_act[1],  # f_dc
            s_act[2]  # f_rest
        ], axis=1).astype(np.float32)
        
        
        g_act.create_dataset('gs_feat', data=act_gs_feat)
        g_act.create_dataset('gs_goal_feat', data=act_gs_goal_feat)
        g_act.create_dataset('f_distilled', data=s_act[6].astype(np.float32))
        g_act.create_dataset('init_pose', data=act_init_pose.astype(np.float64))
        g_act.create_dataset('goal_pose', data=act_goal_pose.astype(np.float64))
        g_act.create_dataset('desc', data=str(pkl_data.get('action_name', 'action')))
        # --- Anchor Group ---
        g_anc = f.create_group('anchor')
        
        anc_gs_feat = np.concatenate([
            s_anc[0],  # xyz
            s_anc[5],  # rot (pose)
            s_anc[4],  # scale
            s_anc[3],   # opacity
            s_anc[1],  # f_dc
            s_anc[2]  # f_rest
        ], axis=1).astype(np.float32)
        
        g_anc.create_dataset('gs_feat', data=anc_gs_feat)
        g_anc.create_dataset('f_distilled', data=s_anc[6].astype(np.float32))
        g_anc.create_dataset('init_pose', data=anc_init_pose.astype(np.float64))
        g_anc.create_dataset('desc', data=str(pkl_data['anchor_desc']))
        
    logger.success(f"==>> H5 saved: {output_h5_path}")
    