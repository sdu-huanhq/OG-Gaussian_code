import torch

from .builder import ERRORS

import math


class Error:
    def __init__(self):
        pass

    def compute_error(self, input_dict, pred_obj_points):
        raise NotImplementedError
    

    @staticmethod
    def _paired_l2(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        return torch.norm(src - tgt, dim=-1).mean()

    @staticmethod
    def _axis_angle_batch(axis: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        axis = axis / axis.norm()
        ux, uy, uz = axis
        K = torch.tensor([[0, -uz, uy],
                        [uz, 0, -ux],
                        [-uy, ux, 0]], 
                        dtype=angles.dtype, device=angles.device)  # (3,3)
        I = torch.eye(3, dtype=angles.dtype, device=angles.device)

        cos_t = torch.cos(angles).unsqueeze(-1).unsqueeze(-1)        # (K,1,1)
        sin_t = torch.sin(angles).unsqueeze(-1).unsqueeze(-1)        # (K,1,1)

        outer = torch.outer(axis, axis)                              # (3,3)
        R = cos_t * I + (1 - cos_t) * outer + sin_t * K
        return R


@ERRORS.register_module()
class KettleError(Error):
    def __init__(self):
        super(KettleError, self).__init__()

    def compute_error(
        self, 
        input_dict, 
        pred_obj_points, 
        coarse_step_deg: int = 10, 
        refine_iters: int = 5
    ):

        device = pred_obj_points.device
        dtype  = pred_obj_points.dtype

        gt     = input_dict["object_goal"]
        scene_pose = input_dict["scene_pose"]
        axis  = scene_pose[:3, 2]
        pivot = scene_pose[:3, 3]
        best_err   = torch.tensor(float("inf"), device=device, dtype=dtype)
        best_angle = 0.0
        step       = math.radians(coarse_step_deg)

        for _ in range(refine_iters + 1):
            if best_err == float("inf"):
                num = int(2 * math.pi / step) + 1
                angles = torch.linspace(-math.pi, math.pi, steps=num,
                                        device=device, dtype=dtype)
            else:
                angles = best_angle + torch.linspace(-step, step, steps=21,
                                                    device=device, dtype=dtype)

            R = self._axis_angle_batch(axis, angles)                      # (K,3,3)
            centred_gt = gt - pivot                                  # (N2,3)

            gt_batch = torch.matmul(centred_gt.unsqueeze(0), R.transpose(1, 2)) + pivot  # (K,N2,3)

            dists = torch.stack([self._paired_l2(pred_obj_points, g) for g in gt_batch])   # (K,)
            cur_err, idx = dists.min(dim=0)

            if cur_err < best_err:
                best_err   = cur_err.clone()
                best_angle = angles[idx].item()

            step *= 0.5

        return best_err
