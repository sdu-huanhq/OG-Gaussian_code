
import torch, torch.nn as nn, torch.nn.functional as F
from gsplat import rasterization

from pointcept.models.utils.structure import Point
from pointcept.models.builder import MODELS, build_model
from pointcept.datasets.error import Error
from pointcept.datasets.builder import build_error
from pointcept.utils.Lie import exp_so3, log_SO3, bracket_so3
import math
from typing import Tuple, Literal


def quat2mat(quat):
    """Convert quaternion (w, x, y, z) to rotation matrix."""
    # Ensure quat is normalized
    quat = F.normalize(quat, dim=-1)
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    
    out = torch.empty(quat.shape[:-1] + (3, 3), device=quat.device, dtype=quat.dtype)
    out[..., 0, 0] = 1 - 2 * (yy + zz)
    out[..., 0, 1] = 2 * (xy - wz)
    out[..., 0, 2] = 2 * (xz + wy)
    out[..., 1, 0] = 2 * (xy + wz)
    out[..., 1, 1] = 1 - 2 * (xx + zz)
    out[..., 1, 2] = 2 * (yz - wx)
    out[..., 2, 0] = 2 * (xz - wy)
    out[..., 2, 1] = 2 * (yz + wx)
    out[..., 2, 2] = 1 - 2 * (xx + yy)
    return out


def rotmat_to_quat_wxyz(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Convert rotation matrices to quaternions in WXYZ order.
    R: (..., 3, 3)
    return: (..., 4)
    """
    assert R.shape[-2:] == (3, 3)

    m00 = R[..., 0, 0]
    m11 = R[..., 1, 1]
    m22 = R[..., 2, 2]
    trace = m00 + m11 + m22

    q = torch.zeros(*R.shape[:-2], 4, device=R.device, dtype=R.dtype)

    cond = trace > 0
    if cond.any():
        t = torch.sqrt(torch.clamp(trace[cond] + 1.0, min=eps)) * 2.0
        q[cond, 0] = 0.25 * t
        q[cond, 1] = (R[cond, 2, 1] - R[cond, 1, 2]) / t
        q[cond, 2] = (R[cond, 0, 2] - R[cond, 2, 0]) / t
        q[cond, 3] = (R[cond, 1, 0] - R[cond, 0, 1]) / t

    cond1 = (~cond) & (m00 > m11) & (m00 > m22)
    if cond1.any():
        t = torch.sqrt(torch.clamp(1.0 + m00[cond1] - m11[cond1] - m22[cond1], min=eps)) * 2.0
        q[cond1, 0] = (R[cond1, 2, 1] - R[cond1, 1, 2]) / t
        q[cond1, 1] = 0.25 * t
        q[cond1, 2] = (R[cond1, 0, 1] + R[cond1, 1, 0]) / t
        q[cond1, 3] = (R[cond1, 0, 2] + R[cond1, 2, 0]) / t

    cond2 = (~cond) & (~cond1) & (m11 > m22)
    if cond2.any():
        t = torch.sqrt(torch.clamp(1.0 + m11[cond2] - m00[cond2] - m22[cond2], min=eps)) * 2.0
        q[cond2, 0] = (R[cond2, 0, 2] - R[cond2, 2, 0]) / t
        q[cond2, 1] = (R[cond2, 0, 1] + R[cond2, 1, 0]) / t
        q[cond2, 2] = 0.25 * t
        q[cond2, 3] = (R[cond2, 1, 2] + R[cond2, 2, 1]) / t

    cond3 = (~cond) & (~cond1) & (~cond2)
    if cond3.any():
        t = torch.sqrt(torch.clamp(1.0 + m22[cond3] - m00[cond3] - m11[cond3], min=eps)) * 2.0
        q[cond3, 0] = (R[cond3, 1, 0] - R[cond3, 0, 1]) / t
        q[cond3, 1] = (R[cond3, 0, 2] + R[cond3, 2, 0]) / t
        q[cond3, 2] = (R[cond3, 1, 2] + R[cond3, 2, 1]) / t
        q[cond3, 3] = 0.25 * t

    q = F.normalize(q, dim=-1)
    return q


def mat2ortho6d(matrices):
    r6d = torch.cat([matrices[..., 0], matrices[..., 1]], dim=-1)
    return r6d  


def tukey_weight(d: torch.Tensor, delta: torch.Tensor, eps_w: float = 1e-6):
    r = d / delta
    w = torch.zeros_like(d)
    mask = r < 1
    if not mask.any():
        delta = torch.quantile(d, 0.7)
        r = d / delta
        mask = r < 1
    w[mask] = (1 - r[mask]**2)**2
    w = torch.clamp(w, min=eps_w)
    return w, delta


def robust_rigid_align(
    init_pts: torch.Tensor,
    pred_pts: torch.Tensor,
    kernel: Literal["huber", "tukey"] = "huber",
    delta: float = 0.008,
    max_iter: int = 10,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device, dtype = init_pts.device, init_pts.dtype

    P = init_pts.clone()
    G = pred_pts.clone()

    if delta is None:
        # Median Absolute Deviation → 1.4826·MAD ≈ σ
        d0 = torch.norm(P - G, dim=1)
        mad_tensor = torch.median(torch.abs(d0 - torch.median(d0)))
        delta = (1.4826 * mad_tensor).clamp(min=1e-3).item()
    delta = torch.tensor(delta, dtype=dtype, device=device)

    R_tot = torch.eye(3, device=device, dtype=dtype)
    t_tot = torch.zeros(1, 3, device=device, dtype=dtype)

    last_loss = torch.tensor(float("inf"), device=device, dtype=dtype)

    for _ in range(max_iter):
        diff = P - G                      # (N,3)
        d = torch.norm(diff, dim=1)       # (N,)

        if kernel == "huber":
            w = torch.where(d <= delta,
                            torch.ones_like(d),
                            delta / d)
        elif kernel == "tukey":
            w, delta = tukey_weight(d, delta)
        else:
            raise ValueError("kernel must be 'huber' or 'tukey'")

        w_sum = w.sum().clamp_min(1e-12)
        mu_P = (w[:, None] * P).sum(0) / w_sum
        mu_G = (w[:, None] * G).sum(0) / w_sum

        P_c = P - mu_P
        G_c = G - mu_G
        H = (w[:, None] * P_c).T @ G_c
        U, _, Vt = torch.linalg.svd(H)
        R = Vt.T @ U.T
        if torch.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        t = mu_G - mu_P @ R.T             # (3,)

        t_tot = t_tot @ R.T + t
        R_tot = R @ R_tot
        P = P @ R.T + t

        loss = torch.sqrt((w * (d**2)).sum() / w_sum)
        if torch.abs(last_loss - loss) < eps:
            break
        last_loss = loss

    T44 = torch.eye(4, device=device, dtype=dtype)
    T44[:3, :3] = R_tot
    T44[:3, 3] = t_tot.squeeze(0)

    return {'transform': T44}


def warmup_weight(cur_epoch, total_epochs, min_w=0.1, max_w=5.0):
    ratio = min(cur_epoch / total_epochs, 1.0)
    return min_w + ratio * (max_w - min_w)


def differentiable_kabsch(
    src: torch.Tensor,          # (B, N, 3)
    dst: torch.Tensor,          # (B, N, 3)
    weights: torch.Tensor = None,   # (B, N), optional
    eps: float = 1e-8,
):
    assert src.shape == dst.shape
    B, N, _ = src.shape

    if weights is None:
        weights = torch.ones(B, N, device=src.device, dtype=src.dtype)

    weights = weights / (weights.sum(dim=1, keepdim=True) + eps)
    w = weights[..., None]  # (B, N, 1)

    mu_src = (w * src).sum(dim=1, keepdim=True)   # (B,1,3)
    mu_dst = (w * dst).sum(dim=1, keepdim=True)   # (B,1,3)

    X = src - mu_src
    Y = dst - mu_dst

    H = X.transpose(1, 2) @ (w * Y)               # (B,3,3)

    U, S, Vh = torch.linalg.svd(H, full_matrices=False)
    V = Vh.transpose(1, 2)

    det = torch.det(V @ U.transpose(1, 2))
    C = torch.eye(3, device=src.device, dtype=src.dtype)[None].repeat(B, 1, 1)
    C[:, 2, 2] = torch.where(det < 0, -1.0, 1.0)

    R = V @ C @ U.transpose(1, 2)                 # (B,3,3)
    t = mu_dst - mu_src @ R.transpose(1, 2)       # (B,1,3)

    return R, t


def _offset_to_batch_ids(offset: torch.Tensor) -> torch.Tensor:
    counts = torch.diff(offset, prepend=offset.new_zeros(1))
    return torch.repeat_interleave(
        torch.arange(offset.numel(), device=offset.device),
        counts,
    )


def _pack_points_by_offset(
    points: torch.Tensor,
    offset: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    points: (N_total, C)
    offset: (B,) cumulative counts

    returns:
        packed:  (B, Nmax, C)
        weights: (B, Nmax)
    """
    counts = torch.diff(offset, prepend=offset.new_zeros(1))
    B = offset.numel()
    Nmax = int(counts.max().item())

    packed = points.new_zeros(B, Nmax, points.shape[-1])
    weights = points.new_zeros(B, Nmax)

    start = 0
    for b, end in enumerate(offset.tolist()):
        end = int(end)
        n = end - start
        packed[b, :n] = points[start:end]
        weights[b, :n] = 1.0
        start = end

    return packed, weights


_SSIM_WINDOW_CACHE = {}


def _get_cached_ssim_window(
    channels: int,
    window_size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    key = (channels, window_size, float(sigma), str(device), str(dtype))
    window = _SSIM_WINDOW_CACHE.get(key, None)
    if window is None:
        coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = (g[:, None] @ g[None, :]).unsqueeze(0).unsqueeze(0)
        window = window.expand(channels, 1, window_size, window_size).contiguous()
        _SSIM_WINDOW_CACHE[key] = window
    return window


def ssim_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> torch.Tensor:
    """
    x, y: (B, C, H, W)
    return: scalar mean SSIM
    """
    assert x.shape == y.shape and x.ndim == 4

    channels = x.shape[1]
    window_2d = _get_cached_ssim_window(
        channels=channels,
        window_size=window_size,
        sigma=sigma,
        device=x.device,
        dtype=x.dtype,
    )

    mu_x = F.conv2d(x, window_2d, padding=window_size // 2, groups=channels)
    mu_y = F.conv2d(y, window_2d, padding=window_size // 2, groups=channels)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window_2d, padding=window_size // 2, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(y * y, window_2d, padding=window_size // 2, groups=channels) - mu_y2
    sigma_xy = F.conv2d(x * y, window_2d, padding=window_size // 2, groups=channels) - mu_xy

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + 1e-8
    )
    return ssim_map.mean()


class RenderGS(nn.Module):
    def __init__(
        self,
        num_views: int = 4,
        render_size: int = 256,
        dssim_weight: float = 0.2,
        alpha_weight: float = 0.0,
        radius_scale: float = 5.0,
        radius_percentile: float = 95.0,
        K=((414.5749, 0.0, 256.0),
           (0.0, 414.5749, 256.0),
           (0.0, 0.0, 1.0)),
        merge_gap_margin: float = 1.25,
        merge_gap_min_scale: float = 4.0,
        merge_gap_min_abs: float = 1.0,
    ):
        super().__init__()
        self.num_views = num_views
        self.render_size = render_size
        self.dssim_weight = dssim_weight
        self.alpha_weight = alpha_weight
        self.radius_scale = radius_scale
        self.radius_percentile = radius_percentile
        self.merge_gap_margin = merge_gap_margin
        self.merge_gap_min_scale = merge_gap_min_scale
        self.merge_gap_min_abs = merge_gap_min_abs
        
        K_tensor = torch.tensor(K, dtype=torch.float32)
        orig_w = K_tensor[0, 2] * 2.0
        orig_h = K_tensor[1, 2] * 2.0
        sx = render_size / orig_w
        sy = render_size / orig_h

        K_tensor[0, 0] *= sx   # fx
        K_tensor[0, 2] *= sx   # cx
        K_tensor[1, 1] *= sy   # fy
        K_tensor[1, 2] *= sy   # cy
        self.register_buffer("K", torch.tensor(K_tensor, dtype=torch.float32))

    @staticmethod
    def estimate_object_center_and_radius(
        means: torch.Tensor,
        radius_scale: float = 5.0,
        radius_percentile: float = 95.0,
    ):
        means = means.float()
        center = means.mean(dim=0)

        dists = torch.linalg.norm(means - center[None, :], dim=-1)
        radius_base = torch.quantile(dists, radius_percentile / 100.0)
        radius = torch.clamp_min(radius_base * radius_scale, 0.3)
        return center, radius

    @staticmethod
    def fibonacci_hemisphere_points(
        num_views: int,
        radius: torch.Tensor,
        center: torch.Tensor,
    ) -> torch.Tensor:
        device = center.device
        dtype = center.dtype

        idx = torch.arange(num_views, device=device, dtype=dtype) + 0.5
        golden_angle = torch.tensor(
            math.pi * (3.0 - math.sqrt(5.0)),
            device=device,
            dtype=dtype,
        )
        
        z = idx / float(num_views)
        r_xy = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
        theta = golden_angle * idx

        x = r_xy * torch.cos(theta)
        y = r_xy * torch.sin(theta)

        points = torch.stack([x, y, z], dim=-1) * radius
        points = points + center[None, :]
        return points

    @staticmethod
    def look_at_viewmat(
        camera_pos: torch.Tensor,
        target: torch.Tensor,
        world_up: torch.Tensor = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        if camera_pos.ndim == 1:
            camera_pos = camera_pos.unsqueeze(0)
        if target.ndim == 1:
            target = target.unsqueeze(0).expand_as(camera_pos)

        if world_up is None:
            world_up = torch.tensor([0.0, 0.0, 1.0], device=camera_pos.device, dtype=camera_pos.dtype)
        else:
            world_up = world_up.to(device=camera_pos.device, dtype=camera_pos.dtype)
        world_up = world_up.view(1, 3).expand_as(camera_pos)

        forward = target - camera_pos
        forward = F.normalize(forward, dim=-1, eps=eps)

        alt_up = torch.tensor([0.0, 1.0, 0.0], device=camera_pos.device, dtype=camera_pos.dtype)
        alt_up = alt_up.view(1, 3).expand_as(camera_pos)
        use_alt = torch.abs((forward * world_up).sum(dim=-1, keepdim=True)) > 0.98
        up = torch.where(use_alt, alt_up, world_up)

        right = torch.cross(forward, up, dim=-1)
        right = F.normalize(right, dim=-1, eps=eps)
        down = torch.cross(forward, right, dim=-1)
        down = F.normalize(down, dim=-1, eps=eps)

        c2w_rot = torch.stack([right, down, forward], dim=-1)
        w2c_rot = c2w_rot.transpose(-1, -2)
        w2c_trans = -torch.einsum('bij,bj->bi', w2c_rot, camera_pos)

        B = camera_pos.shape[0]
        w2c = torch.eye(4, device=camera_pos.device, dtype=camera_pos.dtype).unsqueeze(0).repeat(B, 1, 1)
        w2c[:, :3, :3] = w2c_rot
        w2c[:, :3, 3] = w2c_trans
        return w2c


    def render_gaussians_multi_views_raw(
        self,
        means_raw: torch.Tensor,
        quats_raw: torch.Tensor,
        scales_raw: torch.Tensor,
        opacities_raw: torch.Tensor,
        colors_raw: torch.Tensor,
        viewmats: torch.Tensor,
        packed: bool = False
    ):
        means = means_raw.contiguous().float()
        quats = F.normalize(quats_raw.contiguous().float(), dim=-1)
        scales = torch.exp(scales_raw.contiguous().float())

        opacities = opacities_raw.contiguous().float()
        if opacities.ndim == 2:
            opacities = opacities.squeeze(-1)
        opacities = torch.sigmoid(opacities)

        colors = colors_raw.contiguous().float()
        if colors.ndim != 2 or colors.shape[-1] != 3:
            raise ValueError(f"Expected colors_raw as [N,3], got {tuple(colors.shape)}")

        viewmats = viewmats.contiguous().float().to(means.device)
        K = self.K.to(means.device, means.dtype)
        if K.ndim == 2:
            K = K[None].expand(viewmats.shape[0], -1, -1).contiguous()


        renders, alphas, meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=K,
            width=self.render_size,
            height=self.render_size,
            sh_degree=None,
            render_mode="RGB",
            packed=packed,
        )

        out = {
            "meta": meta,
            "rgb": renders.permute(0, 3, 1, 2),
            "alpha": alphas.permute(0, 3, 1, 2),
        }
        return out

    def _estimate_merge_scene_gap(
        self,
        max_radius: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        K = self.K.to(device=device, dtype=dtype).clone()
        fx = torch.clamp_min(K[0, 0], 1e-6)
        fy = torch.clamp_min(K[1, 1], 1e-6)
        cx = K[0, 2]
        cy = K[1, 2]
        width = torch.tensor(float(self.render_size), device=device, dtype=dtype)
        height = torch.tensor(float(self.render_size), device=device, dtype=dtype)

        tan_half_x = torch.maximum(cx / fx, torch.clamp_min(width - cx, 0.0) / fx)
        tan_half_y = torch.maximum(cy / fy, torch.clamp_min(height - cy, 0.0) / fy)
        tan_half = torch.maximum(tan_half_x, tan_half_y)

        gap_from_fov = 2.0 * (tan_half + 1.0) * max_radius * self.merge_gap_margin
        gap_min = torch.maximum(
            max_radius * self.merge_gap_min_scale,
            torch.tensor(float(self.merge_gap_min_abs), device=device, dtype=dtype),
        )
        return torch.maximum(gap_from_fov, gap_min)

    def _build_shifted_viewmats(
        self,
        center: torch.Tensor,
        radius: torch.Tensor,
    ) -> torch.Tensor:
        cam_positions = self.fibonacci_hemisphere_points(
            num_views=self.num_views,
            radius=radius,
            center=center,
        )
        return self.look_at_viewmat(
            camera_pos=cam_positions,
            target=center,
        )

    def _prepare_merged_render_batch(
        self,
        pred_x_render: torch.Tensor,
        pred_quats_all: torch.Tensor,
        gt_x_render: torch.Tensor,
        gt_quats_all: torch.Tensor,
        scales_raw_all: torch.Tensor,
        opacities_raw_all: torch.Tensor,
        colors_raw_all: torch.Tensor,
        offset: torch.Tensor,
    ):
        device = pred_x_render.device
        dtype = pred_x_render.dtype

        offset_list = [int(x) for x in offset.tolist()]
        if len(offset_list) == 0:
            raise ValueError('offset must contain at least one object')

        objects = []
        radii = []
        start = 0
        for end in offset_list:
            pred_x_i = pred_x_render[start:end]
            gt_x_i = gt_x_render[start:end]

            pred_q_i = pred_quats_all[start:end]
            gt_q_i = gt_quats_all[start:end]
            scales_raw_i = scales_raw_all[start:end]
            opacities_raw_i = opacities_raw_all[start:end]
            colors_raw_i = colors_raw_all[start:end]

            gt_center_i, radius_i = self.estimate_object_center_and_radius(
                means=gt_x_i.detach(),
                radius_scale=self.radius_scale,
                radius_percentile=self.radius_percentile,
            )
            objects.append({
                'pred_x': pred_x_i,
                'gt_x': gt_x_i,
                'pred_q': pred_q_i,
                'gt_q': gt_q_i,
                'scales_raw': scales_raw_i,
                'opacities_raw': opacities_raw_i,
                'colors_raw': colors_raw_i,
                'gt_center': gt_center_i.to(device=device, dtype=dtype),
                'radius': radius_i.to(device=device, dtype=dtype),
            })
            radii.append(radius_i.to(device=device, dtype=dtype))
            start = end

        max_radius = torch.stack(radii).max()
        scene_gap = self._estimate_merge_scene_gap(max_radius, dtype=dtype, device=device)

        pred_means_list = []
        gt_means_list = []
        pred_quats_list = []
        gt_quats_list = []
        scales_list = []
        opacities_list = []
        colors_list = []
        viewmats_list = []

        for obj_idx, obj in enumerate(objects):
            delta = pred_x_render.new_tensor([float(obj_idx), 0.0, 0.0]) * scene_gap
            shifted_center = obj['gt_center'] + delta

            pred_means_list.append(obj['pred_x'] + delta[None, :])
            gt_means_list.append(obj['gt_x'] + delta[None, :])
            pred_quats_list.append(obj['pred_q'])
            gt_quats_list.append(obj['gt_q'])
            scales_list.append(obj['scales_raw'])
            opacities_list.append(obj['opacities_raw'])
            colors_list.append(obj['colors_raw'])
            viewmats_list.append(self._build_shifted_viewmats(shifted_center, obj['radius']).to(device=device, dtype=dtype))

        return {
            'pred_means': torch.cat(pred_means_list, dim=0),
            'gt_means': torch.cat(gt_means_list, dim=0),
            'pred_quats': torch.cat(pred_quats_list, dim=0),
            'gt_quats': torch.cat(gt_quats_list, dim=0),
            'scales_raw': torch.cat(scales_list, dim=0),
            'opacities_raw': torch.cat(opacities_list, dim=0),
            'colors_raw': torch.cat(colors_list, dim=0),
            'viewmats': torch.cat(viewmats_list, dim=0),
            'scene_gap': scene_gap,
        }

    def compute_render_loss(
        self,
        pred_x_render: torch.Tensor,
        pred_R_render: torch.Tensor,
        gt_x_render: torch.Tensor,
        gt_R_render: torch.Tensor,
        gs_raw: torch.Tensor,
        offset: torch.Tensor,
    ):
        pred_quats_all = rotmat_to_quat_wxyz(pred_R_render)
        gt_quats_all = rotmat_to_quat_wxyz(gt_R_render)

        scales_raw_all = gs_raw[:, 7:10]
        opacities_raw_all = gs_raw[:, 10]
        colors_raw_all = gs_raw[:, 11:14]

        merged = self._prepare_merged_render_batch(
            pred_x_render=pred_x_render,
            pred_quats_all=pred_quats_all,
            gt_x_render=gt_x_render,
            gt_quats_all=gt_quats_all,
            scales_raw_all=scales_raw_all,
            opacities_raw_all=opacities_raw_all,
            colors_raw_all=colors_raw_all,
            offset=offset,
        )

        pred_render = self.render_gaussians_multi_views_raw(
            means_raw=merged['pred_means'],
            quats_raw=merged['pred_quats'],
            scales_raw=merged['scales_raw'],
            opacities_raw=merged['opacities_raw'],
            colors_raw=merged['colors_raw'],
            viewmats=merged['viewmats'],
        )
        with torch.no_grad():
            gt_render = self.render_gaussians_multi_views_raw(
                means_raw=merged['gt_means'],
                quats_raw=merged['gt_quats'],
                scales_raw=merged['scales_raw'],
                opacities_raw=merged['opacities_raw'],
                colors_raw=merged['colors_raw'],
                viewmats=merged['viewmats'],
            )

        pred_rgb = pred_render['rgb'].clamp(0.0, 1.0)
        gt_rgb = gt_render['rgb'].clamp(0.0, 1.0)

        l1_rgb = F.l1_loss(pred_rgb, gt_rgb)
        dssim = 1.0 - ssim_torch(pred_rgb, gt_rgb)
        render_rgb_loss = (1.0 - self.dssim_weight) * l1_rgb + self.dssim_weight * dssim
        return render_rgb_loss


@MODELS.register_module()
class FMGSDualPT_SedOed_render(nn.Module):
    def __init__(
        self,
        num_inference_steps=None,
        network=None,
        epoch=1000,
        vT_weight=1.0,
        vR_weight=1.0,
        render_weight=[1.0,],
        error_calculator=None,
        solver="RK4", # euler/RK4
        trans_frame: Literal["spatial", "body"] = "spatial",
        rot_frame: Literal["spatial", "body"] = "spatial",
        save_traj=False,
        source_init: str = "se3_plus_gs",
        source_se3_trans_std: float = 0.02,
        source_se3_rot_std_deg: float = 5.0,
        source_gs_pos_std: float = 0.002,
        source_gs_rot_std_deg: float = 0.0,
        render_use_svd: bool = True,
        **render_params
    ):
        super().__init__()
        self.network = build_model(network)
        if error_calculator is not None:
            self.error_calculator: Error = build_error(error_calculator)
        else:
            self.error_calculator = None
        
        self.num_inference_steps = num_inference_steps
        self.epoch = epoch
        self.vT_weight = vT_weight
        self.vR_weight = vR_weight
        self.render_weight = render_weight
        self.render_use_svd = render_use_svd

        self._render_weight = 0
        
        self.solver = solver
        self.save_traj = save_traj
        
        self.source_init = source_init
        self.source_se3_trans_std = source_se3_trans_std
        self.source_se3_rot_std = math.radians(source_se3_rot_std_deg)
        self.source_gs_pos_std = source_gs_pos_std
        self.source_gs_rot_std = math.radians(source_gs_rot_std_deg)
        
        assert trans_frame in ("spatial", "body"), f"trans_frame must be 'spatial' or 'body', got {trans_frame}"
        assert rot_frame in ("spatial", "body"), f"rot_frame must be 'spatial' or 'body', got {rot_frame}"
        self.trans_frame = trans_frame
        self.rot_frame = rot_frame
        
        self.render = RenderGS(**render_params)

    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
    
    
    @staticmethod
    def _lie_bracket(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return A @ B - B @ A
    
    @staticmethod
    def _make_goal_feat(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        # x: (N,3), R: (N,3,3) -> (N, 3+6)
        return torch.cat([x, mat2ortho6d(R)], dim=-1)
    
    def _pred_to_spatial(self, v_raw: torch.Tensor, w_raw: torch.Tensor, R: torch.Tensor):
        """Convert network outputs (v_raw, w_raw) in configured frames to spatial/world twist (v_s, w_s)."""
        if self.trans_frame == "spatial":
            v_s = v_raw
        else:
            v_s = torch.einsum("nij,nj->ni", R, v_raw)

        if self.rot_frame == "spatial":
            w_s = w_raw
        else:
            w_s = torch.einsum("nij,nj->ni", R, w_raw)

        return v_s, w_s
    
    
    def _run_batched_kabsch(self, init_pts: torch.Tensor, pred_pts: torch.Tensor, offset: torch.Tensor):
        init_packed, weights = _pack_points_by_offset(init_pts, offset)
        pred_packed, _ = _pack_points_by_offset(pred_pts, offset)
        R_obj, t_obj = differentiable_kabsch(init_packed, pred_packed, weights=weights)
        return R_obj, t_obj

    def _apply_batched_rigid_to_gaussians(
        self,
        x_init: torch.Tensor,
        R_init: torch.Tensor,
        R_obj: torch.Tensor,
        t_obj: torch.Tensor,
        offset: torch.Tensor,
    ):
        batch_ids = _offset_to_batch_ids(offset)
        Rb = R_obj[batch_ids]      # (N,3,3)
        tb = t_obj[batch_ids, 0]   # (N,3)

        x_out = torch.bmm(x_init.unsqueeze(1), Rb.transpose(1, 2)).squeeze(1) + tb
        R_out = Rb @ R_init
        return x_out, R_out

    @staticmethod
    def _hat_so3(w: torch.Tensor) -> torch.Tensor:
        """
        Convert axis-angle vectors to skew-symmetric matrices.

        Args:
            w: (..., 3)

        Returns:
            W: (..., 3, 3)
        """
        wx, wy, wz = w[..., 0], w[..., 1], w[..., 2]

        O = torch.zeros_like(wx)

        W = torch.stack(
            [
                O, -wz, wy,
                wz, O, -wx,
                -wy, wx, O,
            ],
            dim=-1,
        ).reshape(*w.shape[:-1], 3, 3)

        return W


    def _sample_source_state(
        self,
        x_obs: torch.Tensor,        # (N_total, 3)
        R_obs: torch.Tensor,        # (N_total, 3, 3)
        offset: torch.Tensor,       # (B,)
        normalization_scale: torch.Tensor,  # (B, 1), in meters
    ):
        """
        Sample source state z from p0(z | G^O).

        source_init options:
            "observed":     z = G^O
            "gaussian":     old behavior, independent Gaussian source
            "se3_plus_gs":  object-level SE(3) perturbation + small per-Gaussian perturbation
        """
        if self.source_init == "observed":
            return x_obs, R_obs

        if self.source_init == "gaussian":
            x0 = torch.randn_like(x_obs)

            q0 = torch.randn(
                x_obs.shape[0],
                4,
                device=x_obs.device,
                dtype=x_obs.dtype,
            )
            q0 = F.normalize(q0, dim=-1)
            R0 = quat2mat(q0)

            return x0, R0

        if self.source_init != "se3_plus_gs":
            raise ValueError(f"Unknown source_init: {self.source_init}")

        # 1. Object-level SE(3) perturbation
        B = offset.numel()
        batch_ids = _offset_to_batch_ids(offset)
        rho = normalization_scale.to(x_obs).reshape(B, 1)

        # translation perturbation per object
        t_eps = torch.randn(
            B,
            3,
            device=x_obs.device,
            dtype=x_obs.dtype,
        ) * (self.source_se3_trans_std / rho)

        # rotation perturbation per object
        w_eps = torch.randn(
            B,
            3,
            device=x_obs.device,
            dtype=x_obs.dtype,
        ) * self.source_se3_rot_std

        R_eps = exp_so3(self._hat_so3(w_eps))   # (B, 3, 3)

        Rb = R_eps[batch_ids]                  # (N_total, 3, 3)
        tb = t_eps[batch_ids]                  # (N_total, 3)

        # row-vector convention: x' = x @ R^T + t
        x0 = torch.bmm(
            x_obs.unsqueeze(1),
            Rb.transpose(1, 2),
        ).squeeze(1) + tb

        R0 = Rb @ R_obs

        # 2. Small per-Gaussian position perturbation
        if self.source_gs_pos_std > 0:
            x0 = x0 + torch.randn_like(x0) * (self.source_gs_pos_std / rho[batch_ids])

        # 3. Small per-Gaussian orientation perturbation
        if self.source_gs_rot_std > 0:
            w_g = torch.randn(
                x_obs.shape[0],
                3,
                device=x_obs.device,
                dtype=x_obs.dtype,
            ) * self.source_gs_rot_std

            R_g = exp_so3(self._hat_so3(w_g))   # (N_total, 3, 3)

            # local perturbation after the original Gaussian orientation
            # Q_z = R_eps Q_obs DeltaQ_i
            R0 = R0 @ R_g

        return x0, R0


    @staticmethod
    def _restore_positions(features: torch.Tensor, input_dict):
        """Restore packed points or trajectories to the pre-normalization frame."""
        batch_ids = _offset_to_batch_ids(input_dict['action_offset']).to(features.device)
        center = input_dict['normalization_center'].to(features).reshape(-1, 3)[batch_ids]
        scale = input_dict['normalization_scale'].to(features).reshape(-1, 1)[batch_ids]
        restored = features.clone()
        restored[..., :3] = features[..., :3] * scale + center
        return restored


    def forward(self, input_dict, mode='loss', apply_SVD=False, cur_epoch=None):
        scene_input_dict = dict(
                coord=input_dict['anchor_coord'],
                grid_coord=input_dict['anchor_grid_coord'],
                feat=input_dict['anchor_feat'],
                gs_raw=input_dict['anchor_gs_raw'],
                offset=input_dict['anchor_offset'],
            )
        scene_point = Point(scene_input_dict)
        
        if mode != 'sample_goal':
            gt_feat = input_dict['action_goal_feat']
            x1 = gt_feat[:, :3]
            
            q1 = gt_feat[:, 3:7]
            R1 = quat2mat(q1)
            
            object_input_dict = dict(
                coord=input_dict['action_coord'],
                feat=input_dict['action_feat'],
                offset=input_dict['action_offset'],
                grid_coord=input_dict['action_grid_coord'],
                gs_raw=input_dict['action_gs_raw'],
                goal=x1
            )
            object_point = Point(object_input_dict)
            skill_emb = input_dict['skill_emb']
            
            self.B = input_dict['action_offset'].shape[0]
            timesteps = torch.rand(self.B, device=self.device)
            bincount = torch.diff(
                object_point.offset, prepend=torch.tensor([0], device=self.device, dtype=torch.long)
            )
            t_expand = timesteps.repeat_interleave(bincount).unsqueeze(-1) # (N, 1)
            
            x0, R0 = self._sample_source_state(
                x_obs=object_point.gs_raw[:, :3],
                R_obs=quat2mat(object_point.gs_raw[:, 3:7]),
                offset=object_point.offset,
                normalization_scale=input_dict['normalization_scale'],
            )

            # --- 1. Position Flow (R3) ---
            xt = (1 - t_expand) * x0 + t_expand * x1
            v_world = x1 - x0
            
            # --- 2. Rotation Flow (SO3) ---
            R_rel = torch.matmul(R0.transpose(1, 2), R1)
            log_rel = log_SO3(R_rel)
            
            exp_t = exp_so3(t_expand.unsqueeze(-1) * log_rel)
            Rt = torch.matmul(R0, exp_t)
            
            object_point.goal = self._make_goal_feat(xt, Rt)
            w_body = bracket_so3(log_rel)
            
            # Targets in configured frames
            if self.trans_frame == "spatial":
                v_target = v_world
            else:
                # body coordinates at time t: v_body(t) = R_t^T v_world
                v_target = torch.einsum("nji,nj->ni", Rt, v_world)

            if self.rot_frame == "body":
                w_target = w_body
            else:
                w_target = torch.einsum("nij,nj->ni", Rt, w_body)
                
            
            result = self.network(scene_point, object_point, skill_emb, timesteps * 1000)
            pred_full = result['pred']
            
            vT_pred = pred_full[:, :3]
            vR_pred = pred_full[:, 3:]
            
            vT_loss = F.mse_loss(vT_pred, v_target)
            vR_loss = F.mse_loss(vR_pred, w_target)
            vT_losses = self.vT_weight * vT_loss
            vR_losses = self.vR_weight * vR_loss
            loss = dict(vT_loss=vT_loss, vR_loss=vR_loss)
            loss['vT_weight'] = torch.tensor(self.vT_weight)
            loss['vR_weight'] = torch.tensor(self.vR_weight)
            total_loss = vT_losses + vR_losses
            
            if self.trans_frame == "spatial":
                pred_v_world = vT_pred
            else:
                pred_v_world = torch.einsum("nij,nj->ni", Rt, vT_pred)
            
            if self.rot_frame == "body":
                pred_w_body = vR_pred
            else:
                pred_w_body = torch.einsum("nij,nj->ni", Rt.transpose(1, 2), vR_pred)
            
            pred_x1 = xt + (1 - t_expand) * pred_v_world
            
            # render loss: cameras are generated from GT action GS only.
            if self.render_weight[-1] > 0:
                if self.render_use_svd:
                    x_init = input_dict['action_gs_raw'][:, :3]
                    R_init = quat2mat(input_dict['action_gs_raw'][:, 3:7])
                    R_obj, t_obj = self._run_batched_kabsch(
                        x_init, pred_x1, input_dict['action_offset']
                    )
                    render_x, render_R = self._apply_batched_rigid_to_gaussians(
                        x_init=x_init,
                        R_init=R_init,
                        R_obj=R_obj,
                        t_obj=t_obj,
                        offset=input_dict['action_offset'],
                    )
                else:
                    render_x = pred_x1
                    render_R = Rt @ torch.matrix_exp(
                        bracket_so3((1 - t_expand) * pred_w_body)
                    )

                render_rgb_loss = self.render.compute_render_loss(
                    pred_x_render=render_x,
                    pred_R_render=render_R,
                    gt_x_render=x1,
                    gt_R_render=R1,
                    gs_raw=input_dict['action_gs_raw'],
                    offset=input_dict['action_offset'],
                )
                loss['render_rgb_loss'] = render_rgb_loss
                if len(self.render_weight) == 2 and cur_epoch is not None:
                    self._render_weight = warmup_weight(cur_epoch, self.epoch, self.render_weight[0], self.render_weight[1])
                else:
                    self._render_weight = self.render_weight[0]
                total_loss = total_loss + self._render_weight * render_rgb_loss
                loss['render_weight'] = torch.tensor(self._render_weight, device=self.device)
            
            loss['loss'] = total_loss
            return loss

        else:
            saved_goals = []
            self.B = input_dict['action_offset'].shape[0]
            scene_point_ = None # scene_point_emb
            object_point_ = None # obj_point_emb

            object_input_dict = dict(
                coord=input_dict['action_coord'],
                feat=input_dict['action_feat'],
                offset=input_dict['action_offset'],
                grid_coord=input_dict['action_grid_coord'],
                gs_raw=input_dict['action_gs_raw'],
            )
            object_point = Point(object_input_dict)
            skill_emb = input_dict['skill_emb']
            
            N = object_point.gs_raw.shape[0]
            x_curr = input_dict['action_gs_raw'][:, :3]
            R_curr = quat2mat(object_point.gs_raw[:, 3:7])
            
            x_obs = input_dict["action_gs_raw"][:, :3]
            R_obs = quat2mat(object_point.gs_raw[:, 3:7])

            x_curr, R_curr = self._sample_source_state(
                x_obs=x_obs,
                R_obs=R_obs,
                offset=object_point.offset,
                normalization_scale=input_dict['normalization_scale'],
            )
            
            object_point.goal = self._make_goal_feat(x_curr, R_curr)

            steps = self.num_inference_steps
            dt = 1.0 / steps
            
            def eval_field(x: torch.Tensor, R: torch.Tensor, t: float, s_cache, o_cache):
                """Evaluate network field and return (v_s, w_s) in spatial/world coordinates."""
                temp_obj_point = Point(dict(
                    coord=object_point.coord,
                    feat=object_point.feat,
                    offset=object_point.offset,
                    grid_coord=object_point.grid_coord,
                    gs_raw=object_point.gs_raw,
                    goal=self._make_goal_feat(x, R),
                ))
                t_tensor = torch.full((self.B,), t, device=self.device, dtype=self.dtype)
                res = self.network(scene_point, temp_obj_point, skill_emb, t_tensor * 1000, s_cache, o_cache)
                pred = res['pred']
                v_raw = pred[:, :3]
                w_raw = pred[:, 3:]
                v_s, w_s = self._pred_to_spatial(v_raw, w_raw, R)
                return v_s, w_s, res.get('sce_point_emb', None), res.get('obj_point_emb', None)
            
            if self.solver == 'euler':
                for i in range(steps):
                    t = i / steps
                    
                    v_s, w_s, scene_point_next, object_point_next = eval_field(x_curr, R_curr, t, scene_point_, object_point_)
                    
                    x_curr = x_curr + v_s * dt
                    w_b = torch.einsum("nji,nj->ni", R_curr, w_s)
                    R_curr = torch.matmul(R_curr, exp_so3(dt * bracket_so3(w_b)))
                    
                    object_point.goal = self._make_goal_feat(x_curr, R_curr)
                    
                    scene_point_ = scene_point_next
                    object_point_ = object_point_next
                    
                result = dict(goal_wo_svd=object_point.goal)
                
            elif self.solver == 'RK4':
            # Lie-group RK4 (Munthe-Kaas) adapted from EquiGraspFlow's SE3_RK4_MK
                for i in range(steps):
                    t0 = i / steps

                    # ---- stage 1 (use caches) ----
                    v1, w_s1, scene_point_next, object_point_next = eval_field(x_curr, R_curr, t0, scene_point_, object_point_)
                    w1 = bracket_so3(torch.einsum("nji,nj->ni", R_curr, w_s1))  # body so(3) matrix
                    I1 = w1

                    # ---- stage 2 ----
                    u2 = dt * 0.5 * w1
                    u2 = u2 + (dt / 12.0) * self._lie_bracket(I1, u2)
                    R2 = torch.matmul(R_curr, exp_so3(u2))
                    x2 = x_curr + 0.5 * dt * v1
                    v2, w_s2, _, _ = eval_field(x2, R2, t0 + 0.5 * dt, scene_point_next, object_point_next)
                    w2 = bracket_so3(torch.einsum("nji,nj->ni", R2, w_s2))

                    # ---- stage 3 ----
                    u3 = dt * 0.5 * w2
                    u3 = u3 + (dt / 12.0) * self._lie_bracket(I1, u3)
                    R3 = torch.matmul(R_curr, exp_so3(u3))
                    x3 = x_curr + 0.5 * dt * v2
                    v3, w_s3, _, _ = eval_field(x3, R3, t0 + 0.5 * dt, scene_point_next, object_point_next)
                    w3 = bracket_so3(torch.einsum("nji,nj->ni", R3, w_s3))

                    # ---- stage 4 ----
                    u4 = dt * w3
                    u4 = u4 + (dt / 6.0) * self._lie_bracket(I1, u4)
                    R4 = torch.matmul(R_curr, exp_so3(u4))
                    x4 = x_curr + dt * v3
                    v4, w_s4, _, _ = eval_field(x4, R4, t0 + dt, scene_point_next, object_point_next)
                    w4 = bracket_so3(torch.einsum("nji,nj->ni", R4, w_s4))

                    # ---- combine ----
                    x_curr = x_curr + (dt / 6.0) * (v1 + 2 * v2 + 2 * v3 + v4)

                    I2 = (2 * (w2 - I1) + 2 * (w3 - I1) - (w4 - I1)) / dt

                    u = (dt / 6.0) * (w1 + 2 * w2 + 2 * w3 + w4)
                    u = u - (dt * dt / 24.0) * self._lie_bracket(I1, I2)
                    u = u - (dt * dt / 12.0) * self._lie_bracket(I1, w4 - I1)
                    u = u + (dt * dt / 12.0) * self._lie_bracket(w1, w4)
                    u = u + (dt * dt / 24.0) * self._lie_bracket(I2, w4 - I1)
                    u = u + (dt * dt / 24.0) * self._lie_bracket(w1 - w4, w2 - w3)
                    u = u + (dt * dt / 12.0) * self._lie_bracket(w2, w3)

                    R_curr = torch.matmul(R_curr, exp_so3(u))

                    object_point.goal = self._make_goal_feat(x_curr, R_curr)

                    # update caches for next step
                    scene_point_ = scene_point_next
                    object_point_ = object_point_next

                    if self.save_traj:
                        saved_goals.append(object_point.goal.detach().cpu())
                    

            result = dict(goal_wo_svd=object_point.goal)
            if self.save_traj:
                result['saved_traj'] = torch.stack(saved_goals, dim=0) # (steps, N, 9)
            
            if apply_SVD:
                assert self.B == 1
                rho = float(input_dict['normalization_scale'].reshape(-1)[0])
                res = robust_rigid_align(
                    object_point.gs_raw[:, :3], object_point.goal[:, :3],
                    delta=0.008 / rho, eps=1e-5 / rho,
                )
                tf = res['transform']
                init_coords_ = torch.ones(
                    (object_point.coord.shape[0], 4), 
                    device=self.device, dtype=object_point.coord.dtype)
                init_coords_[:, :3] = object_point.gs_raw[:, :3]
                object_point.goal = torch.matmul(init_coords_, torch.transpose(tf, 0, 1))[:, :3]
                result["goal"]=object_point.goal
                result['tf'] = tf
            else:
                result["goal"] = result['goal_wo_svd']

            for key in ('goal', 'goal_wo_svd', 'saved_traj'):
                if key in result:
                    result[key] = self._restore_positions(result[key], input_dict)
            if 'tf' in result:
                tf = result['tf'].clone()
                center = input_dict['normalization_center'].to(tf).reshape(-1, 3)[0]
                scale = input_dict['normalization_scale'].to(tf).reshape(-1)[0]
                tf[:3, 3] = scale * tf[:3, 3] + center - tf[:3, :3] @ center
                result['tf'] = tf
            
            if "action_goal_feat" in input_dict:
                goal_coords = self._restore_positions(input_dict['action_goal_feat'][:, :3], input_dict)
                if self.error_calculator is not None and 'anchor_pose' in input_dict:
                    error_input = dict(input_dict, object_goal=goal_coords, scene_pose=input_dict['anchor_pose'])
                    result['error'] = self.error_calculator.compute_error(error_input, result['goal'][:, :3])
                else: #! <--
                    result['error'] = Error._paired_l2(result['goal'][:, :3], goal_coords)

            return result
