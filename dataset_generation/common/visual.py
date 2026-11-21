import numpy as np
from typing import Union
import open3d as o3d
import cv2
import matplotlib.cm as cm
import colorsys
import common.transformation as tf
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import numpy as np
from sklearn.decomposition import PCA
from pycpd import DeformableRegistration
import torch
import fcl
from scipy.spatial import ConvexHull


def check_pointcloud_collision(pc1, pc2):
    def convex_hull_mesh(points):
        hull = ConvexHull(points)
        return points[hull.vertices], hull.simplices

    def make_collision_object(points):
        verts, faces = convex_hull_mesh(points)
        geom = fcl.BVHModel()
        geom.beginModel(len(verts), len(faces))
        geom.addSubModel(verts, faces)
        geom.endModel()
        return fcl.CollisionObject(geom)

    obj1 = make_collision_object(pc1)
    obj2 = make_collision_object(pc2)
    req = fcl.CollisionRequest()
    res = fcl.CollisionResult()
    ret = fcl.collide(obj1, obj2, req, res)
    return ret > 0


def sample_points_in_sphere_torch(center, radius=0.02, num_points=1000, device="cpu"):
    center = center.to(device)

    dirs = torch.randn((num_points, 3), device=device)
    dirs = dirs / torch.norm(dirs, dim=1, keepdim=True)

    r = torch.rand(num_points, device=device) ** (1/3) * radius

    points = center + dirs * r.unsqueeze(1)  # (N, 3)

    return points

def sample_quaternions_around(q_center, max_angle_deg=30, num_samples=1000):
    q_center = np.array(q_center)
    assert q_center.shape == (4,), "q_center 应为 (x, y, z, w)"

    r_center = R.from_quat(q_center)
    
    sampled_quaternions = []

    for _ in range(num_samples):
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        angle_rad = np.deg2rad(np.random.uniform(-max_angle_deg, max_angle_deg))
        delta_r = R.from_rotvec(axis * angle_rad)

        r_sampled = delta_r * r_center
        q_sampled = r_sampled.as_quat()  # (x, y, z, w)
        sampled_quaternions.append(q_sampled)

    return np.stack(sampled_quaternions)


def sample_points_in_sphere(center, radius=0.02, num_points=1000):
    directions = np.random.normal(size=(num_points, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    distances = np.random.rand(num_points) ** (1/3) * radius

    points = center + directions * distances[:, None]
    return points


def generate_2d_spiral_with_constraints(origin=np.array([0.0, 0.0, 0.0]),
                                        point_spacing=0.05,
                                        max_radius=1.0,
                                        max_points=1000):
    points = []
    theta = 0.0
    b = point_spacing / (2 * np.pi)
    count = 0

    while count < max_points:
        r = b * theta
        if r > max_radius:
            break
        x = r * np.cos(theta) + origin[0]
        y = r * np.sin(theta) + origin[1]
        points.append([x, y, origin[2]])

        d_theta = point_spacing / np.sqrt(b**2 + r**2)
        theta += d_theta
        count += 1

    return np.array(points)


def generate_uniform_spherical_expansion(origin=np.array([0.0, 0.0, 0.0]),
                                         num_layers=5,
                                         points_per_layer=100,
                                         max_radius=1.0):
    def fibonacci_sphere(samples):
        points = []
        offset = 2.0 / samples
        increment = np.pi * (3. - np.sqrt(5))

        for i in range(samples):
            y = ((i * offset) - 1) + (offset / 2)
            r = np.sqrt(1 - y*y)
            phi = i * increment
            x = np.cos(phi) * r
            z = np.sin(phi) * r
            points.append([x, y, z])

        return np.array(points)

    unit_sphere_pts = fibonacci_sphere(points_per_layer)
    radii = np.linspace(max_radius / num_layers, max_radius, num_layers)

    all_points = [np.expand_dims(origin, axis=0),]
    for r in radii:
        layer_pts = unit_sphere_pts * r + origin
        all_points.append(layer_pts)

    return np.concatenate(all_points, axis=0)


def class_registration(pcd1, pcd2, w_xyz=1.0, w_feat=0.01):
    """
    pcd: {points, features}
    """
    def normalize_point_cloud(pc):
        center = pc.mean(axis=0)
        pc_centered = pc - center
        scale = np.abs(pc_centered).max(axis=0)
        scale[scale == 0] = 1.0
        norm_pc = pc_centered / scale
        return norm_pc, center, scale

    points1, c1, s1 = normalize_point_cloud(pcd1['points'])
    points2, c2, s2 = normalize_point_cloud(pcd2['points'])
    # source = points1
    # target = points2
    source = np.concatenate([points1*w_xyz, pcd1['features']*w_feat], axis=1)  # (N, 3+D)
    target = np.concatenate([points2*w_xyz, pcd2['features']*w_feat], axis=1)  # (M, 3+D)
    # reg = DeformableRegistration(X=target, Y=source, alpha=0.1, beta=2.0, max_iterations=100)
    reg = DeformableRegistration(X=target, Y=source)
    TY, _ = reg.register()
    return TY[:, :3] * s2 + c2



def rgbd_to_pointcloud(
        rgb, 
        depth, 
        cam_intrinsics, 
        cam_pose, 
        seg=None, 
        target_id=None, 
        valid_mask=None,
        feature_map=None):
    H, W = depth.shape

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    if valid_mask is not None:
        u, v, z = u[valid_mask], v[valid_mask], depth[valid_mask]
    elif seg is not None and target_id is not None:
        valid_mask = (seg == target_id) & (depth > 0)
        u, v, z = u[valid_mask], v[valid_mask], depth[valid_mask]
    else:
        u, v, z = u.flatten(), v.flatten(), depth.flatten()

    if isinstance(cam_intrinsics, torch.Tensor):
        cam_intrinsics = cam_intrinsics.numpy()

    x = (u - cam_intrinsics[0, 2]) * z / cam_intrinsics[0, 0]
    y = (v - cam_intrinsics[1, 2]) * z / cam_intrinsics[1, 1]
    points_camera = np.vstack((x, y, z, np.ones_like(z)))
    points_world = cam_pose @ points_camera  # (4, N)
    points_world = points_world[:3].T  # (N, 3)

    colors = rgb[v, u] / 255.0

    if feature_map is None:
        return points_world, colors
    else:
        return points_world, colors, feature_map[v, u]
    
    
def pcd_voxel_down_sample_with_color_and_features(points, colors, features, voxel_size):

    voxel_indices = np.floor(points / voxel_size)
    voxel_indices_tuple = [tuple(v) for v in voxel_indices]

    voxel_dict = {}
    for i, voxel in enumerate(voxel_indices_tuple):
        if voxel not in voxel_dict:
            voxel_dict[voxel] = {"points": [], "colors": [], "features": []}
        voxel_dict[voxel]["points"].append(points[i])
        voxel_dict[voxel]["colors"].append(colors[i])
        voxel_dict[voxel]["features"].append(features[i])

    down_points = []
    down_colors = []
    down_features = []
    for voxel in voxel_dict.values():
        voxel_points = np.array(voxel["points"])
        voxel_colors = np.array(voxel["colors"])
        voxel_features = np.array(voxel["features"])

        avg_point = np.mean(voxel_points, axis=0)
        avg_color = np.mean(voxel_colors, axis=0)
        avg_feature = np.mean(voxel_features, axis=0)

        down_points.append(avg_point)
        down_colors.append(avg_color)
        down_features.append(avg_feature)

    down_points = np.array(down_points)
    down_colors = np.array(down_colors)
    down_features = np.array(down_features)

    return down_points, down_colors, down_features




def visual_feature(features, foreground_thresh=0):
    """
    visual dino feature
    """
    # PCA
    patch_h, patch_w, c = features.shape
    features_pca = features.reshape(patch_h * patch_w, -1)
    pca = PCA(n_components=3)
    pca.fit(features_pca)
    pca_features = pca.transform(features_pca)
    pca_features[:, 0] = (pca_features[:, 0] - pca_features[:, 0].min()
                        ) / (pca_features[:, 0].max() - pca_features[:, 0].min())
    pca_features_fg = pca_features[:, 0] > foreground_thresh
    pca_features_bg = ~pca_features_fg

    b = np.where(pca_features_bg)

    pca.fit(features_pca[pca_features_fg])
    pca_features_rem = pca.transform(features_pca[pca_features_fg])
    for i in range(3):
        # transform using mean and std, I personally found this transformation gives a better visualization
        pca_features_rem[:, i] = (pca_features_rem[:, i] - pca_features_rem[:, i].mean()) / (pca_features_rem[:, i].std() ** 2) + 0.5

    pca_features_rgb = pca_features.copy()
    pca_features_rgb[pca_features_fg] = pca_features_rem
    pca_features_rgb[b] = 0

    pca_features_rgb = pca_features_rgb.reshape(patch_h, patch_w, 3)
    plt.imshow(pca_features_rgb)
    plt.show()
    plt.close()


def insert_points_between_cloud(cloud: np.ndarray, insert_num: int):
    
    new_cloud = []
    
    for i in range(len(cloud) - 1):
        p1 = cloud[i]
        p2 = cloud[i + 1]
        
        for j in range(insert_num + 1):
            new_point = p1 + (p2 - p1) * (j / (insert_num + 1))
            new_cloud.append(new_point)
    
    new_cloud.append(cloud[-1])
    
    return np.array(new_cloud)


def sample_points_in_bbox(center, quat, extent, num_points=1):
    if not isinstance(center, np.ndarray):
        center = np.array(center)
    if not isinstance(quat, np.ndarray):
        quat = np.array(quat)
    if not isinstance(extent, np.ndarray):
        extent = np.array(extent)

    local_points = (np.random.rand(num_points, 3) - 0.5) * extent
    
    rot_matrix = tf.quaternion_to_rotation_matrix(quat)
    
    world_points = local_points @ rot_matrix.T + center
    
    return world_points


def sample_points_on_rectangle_edges(corners, num_points_per_edge=40):
    if corners.shape != (4, 3):
        raise ValueError("角点坐标必须为形状 (4, 3) 的 numpy 数组")
    
    z_values = corners[:, 2]
    if not np.allclose(z_values, z_values[0]):
        raise ValueError("矩形必须平行于 xy 平面，即所有角点的 z 坐标应相等")
    
    edges = [
        (corners[0], corners[1]),
        (corners[1], corners[2]),
        (corners[2], corners[3]),
        (corners[3], corners[0])
    ]
    
    sampled_points = []
    for start, end in edges:
        edge_points = np.linspace(start, end, num_points_per_edge + 1, endpoint=False)[1:]
        sampled_points.append(edge_points)
    
    return np.vstack(sampled_points)


def compute_bbox_top_corners(bbox_center, bbox_orientation, bbox_extent):
    local_corners = np.array([
        [-bbox_extent[0] / 2, -bbox_extent[1] / 2, bbox_extent[2] / 2],
        [-bbox_extent[0] / 2, bbox_extent[1] / 2, bbox_extent[2] / 2],
        [bbox_extent[0] / 2, bbox_extent[1] / 2, bbox_extent[2] / 2],
        [bbox_extent[0] / 2, -bbox_extent[1] / 2, bbox_extent[2] / 2],
    ])
    
    rotation = R.from_quat(bbox_orientation)
    world_corners = rotation.apply(local_corners) + bbox_center
    
    return world_corners


def ray_box_intersection(box_size, ray_direction, delta=1e-3):
    ray_direction = ray_direction / np.linalg.norm(ray_direction)
    
    l1, l2, l3 = box_size
    a, b, c = ray_direction

    x_min, x_max = -l1 / 2, l1 / 2
    y_min, y_max = -l2 / 2, l2 / 2
    z_min, z_max = -l3 / 2, l3 / 2

    t_values = []

    if a != 0:
        t1 = x_min / a
        t2 = x_max / a
        t_values.append((t1, "x_min"))
        t_values.append((t2, "x_max"))

    if b != 0:
        t3 = y_min / b
        t4 = y_max / b
        t_values.append((t3, "y_min"))
        t_values.append((t4, "y_max"))

    if c != 0:
        t5 = z_min / c
        t6 = z_max / c
        t_values.append((t5, "z_min"))
        t_values.append((t6, "z_max"))

    intersection_points = []
    for t, face in t_values:
        if t < 0:
            continue
        x = t * a
        y = t * b
        z = t * c
        if x_min-delta <= x <= x_max+delta and y_min-delta <= y <= y_max+delta and z_min-delta <= z <= z_max+delta:
            intersection_points.append((x, y, z, t, face))

    if intersection_points:
        intersection_points.sort(key=lambda p: p[3])
        nearest_point = intersection_points[0]
        return nearest_point[:3]
    else:
        raise ValueError


def sample_points_on_sphere(
        interval: int, radius: float, center: Union[list, np.ndarray]=None) -> np.ndarray:
    """
    
    args:
    
    return:
    - points: (num_points, 3)
    """
    latitudes = np.arange(-90+interval, 90, interval)
    longitudes = np.arange(-180, 180, interval)
    
    points = []
    for lat in latitudes:
        for lon in longitudes:
            lat_rad = np.radians(lat)
            lon_rad = np.radians(lon)
            x = radius * np.cos(lat_rad) * np.cos(lon_rad)
            y = radius * np.cos(lat_rad) * np.sin(lon_rad)
            z = radius * np.sin(lat_rad)
            points.append(np.array([x, y, z]))
    
    if center is None:
        center = np.array([0, 0, 0])

    points.append(np.array([center[0], center[1], radius]))
    points.append(np.array([center[0], center[1], -radius]))

    points = np.array(points) + np.array(center)
    return points


def sample_small_quaternion(max_angle_deg=10):
    max_angle_rad = np.radians(max_angle_deg)

    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)

    angle = np.random.uniform(-max_angle_rad, max_angle_rad)

    rotation = R.from_rotvec(angle * axis)
    quaternion = rotation.as_quat()
    return quaternion


def get_pixels_num_with_value(image:np.ndarray, value):
    """
    image: image with 1 channel
    """
    res = np.zeros_like(image)
    res[image==value] = 1
    return np.sum(res)


def generate_camera_poses_hemispherical(num_views, radius, pos):
    indices = np.arange(0, num_views, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / num_views)
    theta = np.pi * (1 + 5**0.5) * indices

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    points = np.stack((x, y, z), axis=-1)
    points = points[points[:, 2] >= 0]

    poses = []
    for point in points:
        position = point

        direction = -position / np.linalg.norm(position)
        up = np.array([0, 0, 1])
        if np.allclose(direction, up) or np.allclose(direction, -up):
            right = np.array([1, 0, 0])
        else:
            right = np.cross(direction, up)
            right /= np.linalg.norm(right)
        camera_down = np.cross(direction, right)
        rotation_matrix = np.stack([right, -camera_down, -direction], axis=-1)

        orientation = R.from_matrix(rotation_matrix).as_quat()  # [x, y, z, w]

        poses.append((position+pos, orientation))

    return poses



def ellipsoid_surface_point_and_normal_v1(center, axes, unit_vectors):
    """

    args:
    return: 
    """
    x_c, y_c, z_c = center
    a, b, c = axes

    t = np.sqrt(1 / ((unit_vectors[:,0]**2 / a**2) + (unit_vectors[:,1]**2 / b**2) + (unit_vectors[:,2]**2 / c**2)))

    surface_point = np.zeros_like(unit_vectors)
    surface_point[:, 0] = x_c + unit_vectors[:,0]*t # (n,)
    surface_point[:, 1] = y_c + unit_vectors[:,1]*t # (n,)
    surface_point[:, 2] = z_c + unit_vectors[:,2]*t # (n,)

    surface_normal = np.zeros_like(unit_vectors)
    surface_normal[:, 0] = (surface_point[:, 0] - x_c) / a**2    # (n,)
    surface_normal[:, 1] = (surface_point[:, 1] - y_c) / b**2
    surface_normal[:, 2] = (surface_point[:, 2] - z_c) / c**2

    norm_l = np.linalg.norm(surface_normal, axis=1) # (n,)
    norm_l = np.expand_dims(norm_l, axis=1).repeat(3, axis=1)
    surface_normal = surface_normal / norm_l

    return surface_point, surface_normal


def ellipsoid_surface_points(center, rotation, axes, unit_vectors):
    """

    args:
        
    return: 
    """
    a, b, c = axes

    t_w_v = np.array(unit_vectors) + np.array(center)
    T_w_e = tf.PosQua_to_TransMat(center, rotation)
    unit_vectors = tf.transPts_T(t_w_v, np.linalg.inv(T_w_e))

    t = np.sqrt(1 / ((unit_vectors[:,0]**2 / a**2) + (unit_vectors[:,1]**2 / b**2) + (unit_vectors[:,2]**2 / c**2)))
    surface_point = np.zeros_like(unit_vectors)
    surface_point[:, 0] = unit_vectors[:,0]*t # (n,)
    surface_point[:, 1] = unit_vectors[:,1]*t # (n,)
    surface_point[:, 2] = unit_vectors[:,2]*t # (n,)

    # t_w_p = T_w_e * t_e_p
    surface_point = tf.transPts_T(surface_point, T_w_e)

    return surface_point


def minimum_circumscribed_ellipsoid(bbox_min, bbox_max):
    """
    return:
    """
    center, radii = maximum_inscribed_ellipsoid(bbox_min, bbox_max)

    surface_unit = bbox_max-center
    surface_unit /= np.linalg.norm(surface_unit)
    surface_point = ellipsoid_surface_points(center, (0,0,0,1), radii, np.array([surface_unit,]))
    scale = np.linalg.norm(bbox_max-center) / np.linalg.norm(surface_point[0]-center)
    print('scale =', scale)
    return center, radii*scale


def maximum_inscribed_ellipsoid(bbox_min, bbox_max):
    """
    return:
    """
    bbox_min = np.array(bbox_min)
    bbox_max = np.array(bbox_max)

    center = (bbox_min + bbox_max) / 2.
    radii = (bbox_max - bbox_min) / 2.

    return center, radii


def generate_points_on_cuboid_surface(size:tuple, num:int, pos=None, quat=None):
    """

    args:
    
    return:
        - points: (num_points, 3)
    """
    a, b, c = size
    areas = [4*a*b, 4*a*b, 4*a*c, 4*a*c, 4*b*c, 4*b*c]
    total_area = sum(areas)
    
    points_per_face = [int(num * area / total_area) for area in areas]
    
    while sum(points_per_face) < num:
        points_per_face[np.argmax(areas)] += 1

    points = []
    
    x = np.random.uniform(-a, a, points_per_face[0])
    y = np.random.uniform(-b, b, points_per_face[0])
    z = np.full(points_per_face[0], c)
    points.extend(np.column_stack((x, y, z)))
    
    x = np.random.uniform(-a, a, points_per_face[1])
    y = np.random.uniform(-b, b, points_per_face[1])
    z = np.full(points_per_face[1], -c)
    points.extend(np.column_stack((x, y, z)))

    x = np.random.uniform(-a, a, points_per_face[2])
    y = np.full(points_per_face[2], -b)
    z = np.random.uniform(-c, c, points_per_face[2])
    points.extend(np.column_stack((x, y, z)))

    x = np.random.uniform(-a, a, points_per_face[3])
    y = np.full(points_per_face[3], b)
    z = np.random.uniform(-c, c, points_per_face[3])
    points.extend(np.column_stack((x, y, z)))

    x = np.full(points_per_face[4], a)
    y = np.random.uniform(-b, b, points_per_face[4])
    z = np.random.uniform(-c, c, points_per_face[4])
    points.extend(np.column_stack((x, y, z)))

    x = np.full(points_per_face[5], -a)
    y = np.random.uniform(-b, b, points_per_face[5])
    z = np.random.uniform(-c, c, points_per_face[5])
    points.extend(np.column_stack((x, y, z)))

    if pos is not None and quat is not None:
        return tf.transPts_tq(points, pos, quat)
    
    return np.array(points)


def uniform_sample_cube_surface_points(size:tuple, num:int, pos=None, quat=None):
    """

    args:
    
    return:
        - points: (num, 3)
    """
    l1, l2, l3 = size

    size_all = l1*l2*2 + l2*l3*2 + l1*l3*2
    num_up = int((l1*l2 / size_all) * num)
    num_left = int((l1*l3 / size_all) * num)
    num_forward = int((l2*l3 / size_all) * num)

    points_1 = np.random.rand(num_up, 3)
    points_1[:, 0] = points_1[:, 0]*2*l1 - l1
    points_1[:, 1] = points_1[:, 1]*2*l2 - l2
    points_1[:, 2] = points_1[:, 2] * 0 + l3
    points_2 = np.random.rand(num_up, 3)
    points_2[:, 0] = points_2[:, 0]*2*l1 - l1
    points_2[:, 1] = points_2[:, 1]*2*l2 - l2
    points_2[:, 2] = points_2[:, 2] * 0 - l3
    points_3 = np.random.rand(num_left, 3)
    points_3[:, 0] = points_3[:, 0]*2*l1 - l1
    points_3[:, 1] = points_3[:, 1] * 0 - l2
    points_3[:, 2] = points_3[:, 2]*2*l3 - l3
    points_4 = np.random.rand(num_left, 3)
    points_4[:, 0] = points_4[:, 0]*2*l1 - l1
    points_4[:, 1] = points_4[:, 1] * 0 + l2
    points_4[:, 2] = points_4[:, 2]*2*l3 - l3
    points_5 = np.random.rand(num_forward, 3)
    points_5[:, 0] = points_5[:, 0] * 0 + l1
    points_5[:, 1] = points_5[:, 1]*2*l2 - l2
    points_5[:, 2] = points_5[:, 2]*2*l3 - l3
    points_6 = np.random.rand(num_forward, 3)
    points_6[:, 0] = points_6[:, 0] * 0 - l1
    points_6[:, 1] = points_6[:, 1]*2*l2 - l2
    points_6[:, 2] = points_6[:, 2]*2*l3 - l3

    points = np.concatenate([points_2, points_1, points_3, points_4, points_5, points_6], axis=0)
    if points.shape[0] < num:
        points_append = np.zeros((num-points.shape[0], 3))
        for i in range(points_append.shape[0]):
            points_append[i, :] = points[-1, :]
        points = np.concatenate([points, points_append], axis=0)
    
    if pos is not None and quat is not None:
        return tf.transPts_tq(points, pos, quat)
    return points


def create_cube_pcl(size, pos, quat, num, usedForNormal=False):
    """

    args:
    """
    l1, l2, l3 = size
    edge_radio = 0.5
    face_radio = 1-edge_radio
    len_edge = 2*l1*4 + 2*l2*4 + 2*l3*4
    num_edgeX = int((2*l1 / len_edge) * (num*edge_radio))
    num_edgeY = int((2*l2 / len_edge) * (num*edge_radio))
    num_edgeZ = int((2*l3 / len_edge) * (num*edge_radio))

    size_all = l1*l2*2 + l2*l3*2 + l1*l3*2
    num_up = int((l1*l2 / size_all) * (num*face_radio))
    num_left = int((l1*l3 / size_all) * (num*face_radio))
    num_forward = int((l2*l3 / size_all) * (num*face_radio))

    points_edge_x = np.random.rand(num_edgeX)*2*l1 - l1
    points_edge_x1 = np.ones((num_edgeX, 3)) * np.array([0, -l2, l3])
    points_edge_x2 = np.ones((num_edgeX, 3)) * np.array([0, l2, l3])
    points_edge_x3 = np.ones((num_edgeX, 3)) * np.array([0, l2, -l3])
    points_edge_x4 = np.ones((num_edgeX, 3)) * np.array([0, -l2, -l3])
    points_edge_x1[:, 0] = points_edge_x
    points_edge_x2[:, 0] = points_edge_x
    points_edge_x3[:, 0] = points_edge_x
    points_edge_x4[:, 0] = points_edge_x
    points_edge_xs = np.concatenate([points_edge_x1, points_edge_x2, points_edge_x3, points_edge_x4], axis=0)
    points_edge_y = np.random.rand(num_edgeY)*2*l2 - l2
    points_edge_y1 = np.ones((num_edgeY, 3)) * np.array([l1, 0, l3])
    points_edge_y2 = np.ones((num_edgeY, 3)) * np.array([-l1, 0, l3])
    points_edge_y3 = np.ones((num_edgeY, 3)) * np.array([-l1, 0, -l3])
    points_edge_y4 = np.ones((num_edgeY, 3)) * np.array([l1, 0, -l3])
    points_edge_y1[:, 1] = points_edge_y
    points_edge_y2[:, 1] = points_edge_y
    points_edge_y3[:, 1] = points_edge_y
    points_edge_y4[:, 1] = points_edge_y
    points_edge_ys = np.concatenate([points_edge_y1, points_edge_y2, points_edge_y3, points_edge_y4], axis=0)
    points_edge_z = np.random.rand(num_edgeZ)*2*l3 - l3
    points_edge_z1 = np.ones((num_edgeZ, 3)) * np.array([l1, l2, 0])
    points_edge_z2 = np.ones((num_edgeZ, 3)) * np.array([-l1, l2, 0])
    points_edge_z3 = np.ones((num_edgeZ, 3)) * np.array([-l1, -l2, 0])
    points_edge_z4 = np.ones((num_edgeZ, 3)) * np.array([l1, -l2, 0])
    points_edge_z1[:, 2] = points_edge_z
    points_edge_z2[:, 2] = points_edge_z
    points_edge_z3[:, 2] = points_edge_z
    points_edge_z4[:, 2] = points_edge_z
    points_edge_zs = np.concatenate([points_edge_z1, points_edge_z2, points_edge_z3, points_edge_z4], axis=0)
    points = np.concatenate([points_edge_xs, points_edge_ys, points_edge_zs], axis=0)
    num_edge = points.shape[0]
    
    l = 0.008
    if usedForNormal:
        l1 -= l
        l2 -= l
    points_1 = np.random.rand(num_up, 3)
    points_1[:, 0] = points_1[:, 0]*2*l1 - l1
    points_1[:, 1] = points_1[:, 1]*2*l2 - l2
    points_1[:, 2] = points_1[:, 2] * 0 + l3
    points_2 = np.random.rand(num_up, 3)
    points_2[:, 0] = points_2[:, 0]*2*l1 - l1
    points_2[:, 1] = points_2[:, 1]*2*l2 - l2
    points_2[:, 2] = points_2[:, 2] * 0 - l3
    if usedForNormal:
        l1 += l
        l2 += l

        l1 -= l
        l3 -= l
    points_3 = np.random.rand(num_left, 3)
    points_3[:, 0] = points_3[:, 0]*2*l1 - l1
    points_3[:, 1] = points_3[:, 1] * 0 - l2
    points_3[:, 2] = points_3[:, 2]*2*l3 - l3
    points_4 = np.random.rand(num_left, 3)
    points_4[:, 0] = points_4[:, 0]*2*l1 - l1
    points_4[:, 1] = points_4[:, 1] * 0 + l2
    points_4[:, 2] = points_4[:, 2]*2*l3 - l3
    if usedForNormal:
        l1 += l
        l3 += l

        l2 -= l
        l3 -= l
    points_5 = np.random.rand(num_forward, 3)
    points_5[:, 0] = points_5[:, 0] * 0 + l1
    points_5[:, 1] = points_5[:, 1]*2*l2 - l2
    points_5[:, 2] = points_5[:, 2]*2*l3 - l3
    points_6 = np.random.rand(num_forward, 3)
    points_6[:, 0] = points_6[:, 0] * 0 - l1
    points_6[:, 1] = points_6[:, 1]*2*l2 - l2
    points_6[:, 2] = points_6[:, 2]*2*l3 - l3
    if usedForNormal:
        l2 += l
        l3 += l

    num_edge += points_2.shape[0]
    points = np.concatenate([points, points_2, points_1, points_3, points_4, points_5, points_6], axis=0)
    if points.shape[0] < num:
        points_append = np.zeros((num-points.shape[0], 3))
        for i in range(points_append.shape[0]):
            points_append[i, :] = points[-1, :]
        points = np.concatenate([points, points_append], axis=0)
    return tf.transPts_tq(points, pos, quat)


def randomize_colors(N, bright=True):
    """
    Modified from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py#L59
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.5
    hsv = [(1.0 * i / N, 1, brightness) for i in range(N)]
    colors = np.array(list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv)))
    rstate = np.random.RandomState(seed=20)
    np.random.shuffle(colors)
    return colors


def segmentation_to_rgb(seg_im, random_colors=False):
    """
    Helper function to visualize segmentations as RGB frames.
    NOTE: assumes that geom IDs go up to 255 at most - if not,
    multiple geoms might be assigned to the same color.
    """
    # ensure all values lie within [0, 255]
    seg_im = np.mod(seg_im, 256)

    if random_colors:
        colors = randomize_colors(N=256, bright=True)
        return (255.0 * colors[seg_im]).astype(np.uint8)
    else:
        # deterministic shuffling of values to map each geom ID to a random int in [0, 255]
        rstate = np.random.RandomState(seed=8)
        inds = np.arange(256)
        rstate.shuffle(inds)

        # use @inds to map each geom ID to a color
        return (255.0 * cm.rainbow(inds[seg_im], 3)).astype(np.uint8)[..., :3]
    

def draw_pcl(pcl, colors=None, coordinate_frame_size=0.1):
    """
    args:
        - pcl: (n, 3) points location
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcl)
    if colors is not None:
        assert colors.shape[1] >= 3
        if colors.shape[1] > 3:
            pca = PCA(n_components=3, random_state=42)
            colors = pca.fit_transform(colors)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=coordinate_frame_size, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, coordinate_frame], window_name="pcd")


def depth2Gray(im_depth):
    x_max = np.max(im_depth)
    x_min = np.min(im_depth)
    # if x_max == x_min:
    #     raise EOFError
    
    k = 255 / (x_max - x_min)
    b = 255 - k * x_max
    return (im_depth * k + b).astype(np.uint8)

def depth2Gray3(im_depth):
    """
    (h, w, 3)
    """
    x_max = np.max(im_depth)
    x_min = np.min(im_depth)
    if x_max == x_min:
        print('图像渲染出错 ...')
        raise EOFError
    
    k = 255 / (x_max - x_min)
    b = 255 - k * x_max

    ret = (im_depth * k + b).astype(np.uint8)
    ret = np.expand_dims(ret, 2).repeat(3, axis=2)
    return ret

def depth2RGB(im_depth):
    """
    (h, w, 3)
    """
    im_depth = depth2Gray(im_depth)
    im_color = cv2.applyColorMap(im_depth, cv2.COLORMAP_JET)
    return im_color


def show_rgb_seg_dep(camera_name, img_rgb, img_seg, img_dep):
    import matplotlib.pyplot as plt
    # from diffusion_policy.common.visual import depth2RGB

    plt.figure(figsize=(10,5))
    plt.suptitle(camera_name)
    plt.subplot(2,2,1), plt.title('RGB')
    plt.imshow(img_rgb)
    plt.subplot(2,2,2), plt.title('Seg')
    plt.imshow(segmentation_to_rgb(img_seg))
    plt.subplot(2,2,3), plt.title('Dep')
    plt.imshow(depth2RGB(img_dep))
    plt.show()

def show_pcd(scene_pcd):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # scene
    ax.scatter(*tuple(scene_pcd.transpose(1, 0)))
    # show
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # plt.savefig('savefig_example.png')
    plt.show()



def visual_pcd(pcd):
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(projection='3d')
    
    # visual scene
    ax.scatter(*tuple(pcd.transpose(1, 0)), color=Color.color('black'))

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # ax.set_zlim(0.7, 1.05)
    # plt.xticks(np.arange(-0.4, 0.4, 0.05))
    # plt.yticks(np.arange(-0.4, 0.4, 0.05))
    plt.show()



class Color:
    scale = 255.
    base_color = {
        'r': [255, 0, 0],
        'red': [255, 0, 0],
        'g': [0, 255, 0],
        'green': [0, 255, 0],
        'b': [0, 0, 255],
        'blue': [0, 0, 255],
        # 'b': [0, 0, 0],
        'black': [0, 0, 0]
    }
    
    # def __init__(self) -> None:
        # pass 
    
    @classmethod
    def color(self, c:str) -> np.ndarray:
        """
        return: rgb
        """
        assert c in self.base_color.keys()
        return np.array(self.base_color[c])

    @classmethod
    def gradient_colors(
            self, 
            num: int,
            start_c: Union[str, list, np.ndarray], 
            end_c: Union[str, list, np.ndarray], 
            ) -> np.ndarray:
        """

        args:
                - str: r, g, b, red, green, blue
                - list: rgb
                - np.ndarray: one-dim rgb
        
        return:
        """
        assert num > 1

        if isinstance(start_c, str):
            assert start_c in self.base_color.keys()
            start_color = np.array(self.base_color[start_c])
        else:
            start_color = np.array(start_c)

        if isinstance(end_c, str):
            assert end_c in self.base_color.keys()
            end_color = np.array(self.base_color[end_c])
        else:
            end_color = np.array(end_c)

        step = (end_color - start_color) / (num-1)  # (3,)
        step_num = np.arange(num)[:, np.newaxis]   # (num, 1)
        steps = np.expand_dims(step, axis=0).repeat(num, axis=0)    # (num, 3)
        colors = steps * step_num + start_color
        colors[-1] = end_color
        return colors / self.scale
    


if __name__ == '__main__':
    import time
    t1 = time.time()
    pts = sample_points_on_sphere(10, 1)
    print('耗时 ', time.time()-t1)
    print('pts.shape =', pts.shape)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, coordinate_frame],window_name="pcd")