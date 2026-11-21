from scipy.spatial.transform import Rotation as R
import numpy as np
import time
# import shapely.ops
import math
# import open3d as o3d
import torch
import scipy
import trimesh
# from diffusion_policy.so3diffusion.util import quat_to_rmat
from typing import Union, List



def rotate_about_world_xy_axis(q_center, angle_deg=90):
    theta = np.random.uniform(0, 2 * np.pi)
    axis = np.array([np.cos(theta), np.sin(theta), 0.0])

    r_rot = R.from_rotvec(np.deg2rad(angle_deg) * axis)

    r_center = R.from_quat(q_center)

    r_new = r_rot * r_center

    return r_new.as_quat()


def sample_axis_rotation_only(q_center, angle_range_deg=90):
    axis = np.random.choice(['x', 'y', 'z'])

    angle_deg = np.random.uniform(-angle_range_deg, angle_range_deg)
    angle_rad = np.deg2rad(angle_deg)

    if not isinstance(q_center, np.ndarray):
        q_center = np.array(q_center)

    r_delta = R.from_euler(axis, angle_rad)
    r_center = R.from_quat(q_center)

    r_sampled = r_delta * r_center

    return r_sampled.as_quat()



def quat_inverse(qua:np.ndarray):
    x, y, z, w = qua
    
    norm_squared = x**2 + y**2 + z**2 + w**2
    
    inverse = np.array([-x, -y, -z, w]) / norm_squared
    return inverse


def check_fin_obj_collision(obj_pcd, f_pos, radius):
    """


    return: 
    """
    mesh = trimesh.primitives.Sphere(radius, f_pos)
    # mesh.show()
    res = max(trimesh.proximity.signed_distance(mesh, obj_pcd)) > 0
    return res

def angle_3dvector(vec0, vec1):
    tem = vec0.dot(vec1)
    tep = np.sqrt(vec0.dot(vec0) * vec1.dot(vec1))
    angle = np.arccos(tem / tep)
    if np.isnan(angle):
        angle = np.arccos(tep / tem)
    return angle


def qua_diff(qua1, qua2):
    dot_product = np.sum(np.array(qua1) * np.array(qua2))
    dot_product = np.clip(dot_product, -1, 1)
    error = np.arccos(abs(dot_product))
    return error


def angle_diff(angle_1, angle_2):
    angle_1 = angle_1 % (2*np.pi)
    angle_2 = angle_2 % (2*np.pi)
    
    angle_min = min(angle_1, angle_2)
    angle_max = max(angle_1, angle_2)

    error = angle_max - angle_min
    if error <= np.pi:
        return error
    else:
        return angle_min + 2*np.pi - angle_max
    

def quaternion_to_rotation_matrix(q):
    """
    xyzw
    """
    return R.from_quat(q).as_matrix()

def quaternion_to_rotation_matrix_npBatch(qs):
    """
    xyzw
    qs: (n, 4)
    """
    rs = list()
    for q in qs:
        r = R.from_quat(q).as_matrix()
        rs.append(r) 
    return np.array(rs)

def PosRmat_to_TransMat(pos, rot):
    """

    input:
        pos: [ndarray, (3,), np.float]
        rot: [ndarray, (3,3), np.float]

    return:
        T: [ndarray, (4,4), np.float]
    """
    T = np.zeros((4, 4), dtype=float)
    T[:3, :3] = rot
    T[:, 3] = np.r_[np.array(pos), np.array([1,])]
    return T

def PosRmat_to_TransMat_batch(pos: torch.Tensor, rmat: torch.Tensor):
    """

    input:
        - pos: [B, ..., 3]

    return:
        T: (B, ..., 4, 4)
    """
    B = pos.shape[0]
    T = torch.zeros(rmat.shape[:-2] + (4, 4), device=pos.device)
    T[..., :3, :3] = rmat
    T[..., :3, 3] = pos
    T[..., 3, 3] = 1
    return T

def PosRmat_to_TransMat_npbatch(pos: np.ndarray, rmat: np.ndarray):
    """

    input:
        - pos: [B, ..., 3]

    return:
        T: (B, ..., 4, 4)
    """
    B = pos.shape[0]
    T = np.zeros(rmat.shape[:-2] + (4, 4))
    T[..., :3, :3] = rmat
    T[..., :3, 3] = pos
    T[..., 3, 3] = 1
    return T

def PosRmat_to_TransMat_batch_v1(pos: torch.Tensor, rmat: torch.Tensor):
    """

    input:
        - pos: [B, ..., 3]

    return:
        T: (B, ..., 4, 4)
    """
    B = pos.shape[0]
    T = torch.zeros((B, 4, 4), device=pos.device)
    T[:, :3, :3] = rmat
    T[:, :3, 3] = pos
    T[:, 3, 3] = 1
    return T


def PosQua_to_TransMat(pos, qua):
    """
    input:
        - pos: [ndarray, (3,), np.float]
        - qua: [ndarray, (4,), np.float]
    """
    return PosRmat_to_TransMat(pos, quaternion_to_rotation_matrix(qua))


# def PosQua_to_TransMat_batch(pos: torch.Tensor, qua: torch.Tensor):
#     """
#     input:
#     """
#     qua_wxyz = torch.concat((qua[:, 3][...,None], qua[:, :3]), dim=1)   # wxyz
#     return PosRmat_to_TransMat_batch(pos, quat_to_rmat(qua_wxyz)) # (B, 4, 4)


def PosQua_to_TransMat_npbatch(pos: np.ndarray, qua: np.ndarray):
    """

    input:
    
    return:
        - Tmat: (B, 4, 4)
    """
    rmats = list()
    for q in qua:
        rmats.append(quaternion_to_rotation_matrix(q))
    rmats = np.array(rmats)
    return PosRmat_to_TransMat_npbatch(pos, rmats) # (B, 4, 4)

def Qua_to_Euler(qua):
    """
    qua: (4,)
    return: euler (3,)
    """
    return R.from_quat(qua).as_euler('xyz', degrees=False)

def Euler_to_Qua(euler):
    """
    eulers: (3,)
    return: qua (4,)
    """
    return R.from_euler(seq='xyz', angles=euler, degrees=False).as_quat()

def Euler_to_Qua_npbatch(eulers):
    """
    eulers: (B, 3)
    return: qua (B, 4)
    """
    qs = list()
    for euler in eulers:
        qs.append(R.from_euler(seq='xyz', angles=euler, degrees=False).as_quat())
    return np.array(qs)


def PosEuler_to_TransMat(pos, euler):
    """

    input:
        pos: [ndarray, (3,), np.float]
        euler: [ndarray, (3,), np.float]

    return:
        T: [ndarray, (4,4), np.float]
    """
    return PosRmat_to_TransMat(
        pos, R.from_euler(seq='xyz', angles=euler, degrees=False).as_matrix())


def TransMat_to_PosQua(TMat):
    """

    input:
    
    return:
    """
    rmat = TMat[:3, :3]
    pos = TMat[:3, 3]
    qua = R.from_matrix(rmat).as_quat()
    return pos, qua


def TransMat_to_PosQua_npbatch(TMat):
    """

    input:
    
    return:
    """
    rmat = TMat[:, :3, :3]  # (B, 3, 3)
    pos = TMat[:, :3, 3]    # (B, 3)
    qua = list()
    for r in rmat:
        qua.append(R.from_matrix(r).as_quat())
    return pos, np.array(qua)


def transPts_tq(P_f1_pts, t_f2_f1, q_f2_f1):
    """

    args:

    return:
        - pts: (n, 3)
    """
    T_f2_f1 = PosRmat_to_TransMat(t_f2_f1, quaternion_to_rotation_matrix(q_f2_f1))
    one = np.ones((1, P_f1_pts.shape[0]))
    _P_f1_pts = np.concatenate((P_f1_pts.T, one), axis=0)   # (4,4)
    P_f2_f1 = np.matmul(T_f2_f1, _P_f1_pts)
    return P_f2_f1.T[:, :3]

def transPts_tq_npbatch(BP_f1_pts:np.ndarray, Bt_f2_f1:np.ndarray, Bq_f2_f1:np.ndarray=None):
    """

    args:

    return:
        - pts: (N, n, 3)
    """
    Bpts = list()
    for i in range(BP_f1_pts.shape[0]):
        if Bq_f2_f1 is None:
            q = (0, 0, 0, 1)
        else:
            q = Bq_f2_f1[i]
        pts = transPts_tq(BP_f1_pts[i], Bt_f2_f1[i], q)
        Bpts.append(pts)
    return np.array(Bpts)

def transPts_T(P_f1_pts, T_f2_f1) -> np.ndarray:
    """

    args:

    return:
        - pts: (n, 3)
    """
    one = np.ones((1, P_f1_pts.shape[0]))
    _P_f1_pts = np.concatenate((P_f1_pts.T, one), axis=0)   # (4,4)
    P_f2_f1 = np.matmul(T_f2_f1, _P_f1_pts)
    return P_f2_f1.T[:, :3]


def transVecs_T(V_f1_vecs: np.ndarray, T_f2_f1: np.ndarray) -> np.ndarray:
    """

    args:

    return:
    """
    R_f2_f1 = T_f2_f1[:3, :3]

    V_f2_vecs = (R_f2_f1 @ V_f1_vecs.T).T  # (n, 3)

    return V_f2_vecs




def transPts_T_npbatch(P_f1_pts: np.ndarray, T_f2_f1: np.ndarray):
    """

    args:

    return:
        - pts: (B, ..., n, 3)
    """
    one = np.ones(T_f2_f1.shape[:-2] + (1, P_f1_pts.shape[-2]))  # (B, 1, n)
    _P_f1_pts = np.concatenate((P_f1_pts.transpose(0, 2, 1), one), axis=1)  # (B, 4, n)
    P_f2_f1 = np.matmul(T_f2_f1, _P_f1_pts)
    return P_f2_f1.transpose(0, 2, 1)[..., :3]


def transPts_T_batch(P_f1_pts: torch.Tensor, T_f2_f1: torch.Tensor):
    """

    args:

    return:
        - pts: (B, ..., n, 3)
    """
    one = torch.ones(T_f2_f1.shape[:-2] + (1, P_f1_pts.shape[-2]), device=P_f1_pts.device)  # (B, ..., 1, n)
    _P_f1_pts = torch.concat((P_f1_pts.transpose(-1, -2), one), dim=-2)  # (B, ..., 4, n)
    P_f2_f1 = torch.matmul(T_f2_f1, _P_f1_pts)
    return P_f2_f1.transpose(-1, -2)[..., :3]


def transPts_T_batch_v1(P_f1_pts: torch.Tensor, T_f2_f1: torch.Tensor):
    """

    args:

    return:
        - pts: (B, n, 3)
    """
    B = P_f1_pts.shape[0]
    one = torch.ones((B, 1, P_f1_pts.shape[1]), device=P_f1_pts.device)
    _P_f1_pts = torch.concat((P_f1_pts.transpose(-1, -2), one), dim=1)  # (B, 4, n)
    P_f2_f1 = torch.matmul(T_f2_f1, _P_f1_pts)
    return P_f2_f1.transpose(-1, -2)[..., :3]


def transPt(P_f1_pt, T_f2_f1=None, t_f2_f1=None, q_f2_f1=None):
    """

    args:

    return:
        - pts: (3,)
    """
    if T_f2_f1 is not None:
        pass
    elif t_f2_f1 is not None and q_f2_f1 is not None:
        T_f2_f1 = PosRmat_to_TransMat(t_f2_f1, quaternion_to_rotation_matrix(q_f2_f1))
    else:
        raise ValueError

    _P_f1_pt = np.array([
        [P_f1_pt[0]], 
        [P_f1_pt[1]], 
        [P_f1_pt[2]],
        [1]
    ])
    P_f2_pt = np.matmul(T_f2_f1, _P_f1_pt) # (4, 1)
    return P_f2_pt[:3, 0]


# def transPt_tq_batch(P_f1_pt, t_f2_f1=None, q_f2_f1=None):
#     """

#     args:

#     return:
#         - pts: (B, 3)
#     """
#     # quaternion to rotation
#     qua = torch.concat((q_f2_f1[:, 3][...,None], q_f2_f1[:, :3]), dim=1)   # wxyz
#     T_f2_f1 = PosRmat_to_TransMat_batch(t_f2_f1, quat_to_rmat(qua)) # (B, 4, 4)

#     _P_f1_pt = torch.ones((P_f1_pt.shape[0], 4, 1), device=P_f1_pt.device)  # (B, 4, 1)
#     _P_f1_pt[:, :3] = P_f1_pt   # (B, 3, 1)
#     P_f2_pt = torch.matmul(T_f2_f1, _P_f1_pt)   # (B, 4, 1)
#     return P_f2_pt[:, :3, 0]

def transPt_T_batch(P_f1_pt, T_f2_f1):
    """

    args:

    return:
        - pts: (B, 3)
    """
    _P_f1_pt = torch.ones((P_f1_pt.shape[0], 4, 1), device=P_f1_pt.device)  # (B, 4, 1)
    _P_f1_pt[:, :3, 0] = P_f1_pt   # (B, 3, 1)
    P_f2_pt = torch.matmul(T_f2_f1, _P_f1_pt)   # (B, 4, 1)
    return P_f2_pt[:, :3, 0]


def transPt_T_npbatch(
        P_f1_pt: np.ndarray, 
        T_f2_f1: np.ndarray):
    """

    args:

    return:
        - pts: (B, 3)
    """
    _P_f1_pt = np.ones(P_f1_pt.shape[:-1]+(4, 1))  # (B, ..., 4, 1)
    _P_f1_pt[..., :3, 0] = P_f1_pt   # (B, ..., 3, 1)
    P_f2_pt = np.matmul(T_f2_f1, _P_f1_pt)   # (B, ..., 4, 1)
    return P_f2_pt[..., :3, 0]

# def TransMat_to_PosRmat(transMat):
#     """
#     """
#     pos = transMat[:3, 3]
#     rmat = transMat[:3, :3]
#     return transMat[:3, 3], transMat[:3, :3]


def create_point_cloud(im_rgb, im_dep, cameraInMatrix, workspace_mask=None):
    """
 Generate point cloud using depth image only.

        Input:
            im_rgb: [numpy.ndarray, (H,W,3), numpy.float32]
                rgb image
            im_dep: [numpy.ndarray, (H,W), numpy.float32]

        Output:
            cloud: [numpy.ndarray, (H,W,3)/(H*W,3), numpy.float32]
                generated cloud, (H,W,3) for organized=True, (H*W,3) for organized=False
    """
    xmap = np.arange(im_rgb.shape[1])
    ymap = np.arange(im_rgb.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = im_dep
    points_x = (xmap - cameraInMatrix[0][2]) * points_z / cameraInMatrix[0][0]
    points_y = (ymap - cameraInMatrix[1][2]) * points_z / cameraInMatrix[1][1]
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    if workspace_mask is not None:
        cloud = cloud[workspace_mask]
    cloud = cloud.reshape([-1, 3])

    return cloud

# def removeOutLier_pcl(pcl, nb_points=10, radius=0.02):
#     """
#     """
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(pcl)
#     return np.asarray(pcd.points)

def farthest_point_sample(point, npoint, color=None, feat:List[np.ndarray]=None):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    if color is not None and feat is None:
        color = color[centroids.astype(np.int32)]
        return point, color
    elif color is None and feat is not None:
        feat = [f[centroids.astype(np.int32)] for f in feat]
        return point, feat
    elif color is not None and feat is not None:
        color = color[centroids.astype(np.int32)]
        # feat = feat[centroids.astype(np.int32)]
        feat = [f[centroids.astype(np.int32)] for f in feat]
        return point, color, feat
    else:
        return point


def PosRad_to_Tmat(pos, rad):
    R = np.array([[np.cos(rad), -np.sin(rad), pos[0]],
                  [np.sin(rad),  np.cos(rad), pos[1]],
                  [0,0,1]])
    return R

def transPt2D(P_f1_pt, T_f2_f1):
    """
    P_f1_pt: (2,)
    T_f2_f1: (3, 3)
    """
    _P_f1_pt = np.array([
        [P_f1_pt[0]], 
        [P_f1_pt[1]], 
        [1]
    ])
    P_f2_pt = np.matmul(T_f2_f1, _P_f1_pt) # (3, 1)
    return P_f2_pt[:2, 0]