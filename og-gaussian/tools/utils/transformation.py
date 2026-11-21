import numpy as np


def PosRmat_to_TransMat(pos, rot):
    T = np.zeros((4, 4), dtype=float)
    T[:3, :3] = rot
    T[:, 3] = np.r_[np.array(pos), np.array([1,])]
    return T


# def PosQua_to_TransMat_batch(pos: torch.Tensor, qua: torch.Tensor):
#     """
#     input:
#     """


# def transPt_tq_batch(P_f1_pt, t_f2_f1=None, q_f2_f1=None):
#     """

#     args:

#         - pts: (B, 3)
#     """
#     # quaternion to rotation


# def TransMat_to_PosRmat(transMat):
#     """
#     """


# def removeOutLier_pcl(pcl, nb_points=10, radius=0.02):
#     """
#     """
