import os
import sys
sys.path.append(os.curdir)

import csv
import gzip
import copy
import pickle
import shutil
import tempfile
from pathlib import Path
from collections import OrderedDict
import tools.utils.transformation as trans

import numpy as np
import h5py
import torch
import torch.multiprocessing as mp
import open3d as o3d

from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R
from loguru import logger

from pointcept.engines.defaults import (
    default_config_parser,
    default_setup,
)
from pointcept.models import build_model
from pointcept.utils.misc import make_dirs
from pointcept.datasets.transform import Compose
from tools.utils.clip_encoder import CLIPTextEncoder


def safe_scalar(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return float(np.asarray(x).reshape(-1)[0])
    return float(x)


def setup_logger(log_file):
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(
        log_file,
        level="INFO",
        enqueue=True,
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | PID={process} | {message}"
    )


def build_cfg(config_file, weight_path, save_root, npy_result_dir, vis_result_dir):
    options = {
        "weight": weight_path,
        "save_path": save_root,
    }
    cfg = default_config_parser(config_file, options)
    cfg = default_setup(cfg)

    cfg.model.save_traj = True

    cfg.npy_result_dir = npy_result_dir
    cfg.vis_result_dir = vis_result_dir

    return cfg


def load_nested_dict_from_h5(h5file, path):
    result = {}

    if path == "/":
        keys = h5file.keys()
    else:
        keys = h5file[path].keys()

    for key in keys:
        if path == "/":
            full_path = key
        else:
            full_path = f"{path}/{key}"

        if isinstance(h5file[full_path], h5py.Group):
            result[key] = load_nested_dict_from_h5(h5file, full_path)
        else:
            data = h5file[full_path][()]
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            result[key] = data

    return result


def collect_anchor_items(anchor_data):
    if "gs_feat" in anchor_data:
        return [anchor_data]

    items = []
    for _, v in anchor_data.items():
        if isinstance(v, dict) and "gs_feat" in v:
            items.append(v)
    return items


def collect_anchor_gs_feat(data_raw):
    anchor_items = collect_anchor_items(data_raw["anchor"])
    anchor_feats = [item["gs_feat"] for item in anchor_items]
    return np.concatenate(anchor_feats, axis=0)


def clone_sample_for_prepare(data):
    new_data = {}
    for k, v in data.items():
        if k in ["anchor", "action"]:
            new_data[k] = copy.deepcopy(v)
        else:
            if isinstance(v, np.ndarray):
                new_data[k] = v.copy()
            else:
                new_data[k] = copy.deepcopy(v)
    return new_data


class Processer:
    def __init__(
        self,
        data_root=None,
        clip_emb_path=None,
        transform=None,
        activated=False
    ):
        self.data_root = data_root
        self.clip_emb_path = clip_emb_path
        self.transform = Compose(transform)
        self.sn_emb = None
        self.od_emb = None

        self.clip = None

        if self.data_root:
            self.load_text_emb()
            
        if activated:
            self.act_gs_feat_name = "gs_feat_activated"
            self.act_goal_gs_feat_name = "gs_goal_feat_activated"
            self.anc_gs_feat_name = "gs_feat_activated"
        else:
            self.act_gs_feat_name = "gs_feat"
            self.act_goal_gs_feat_name = "gs_goal_feat"
            self.anc_gs_feat_name = "gs_feat"

    def load_text_emb(self):
        objects_description_path = os.path.join(self.data_root, "objects_description.pkl.gz")
        if not os.path.exists(objects_description_path) and self.clip_emb_path:
            objects_description_path = os.path.join(self.clip_emb_path, "objects_description.pkl.gz")

        od_emb_path = os.path.join(self.data_root, "objects_description_emb.pkl.gz")
        if not os.path.exists(od_emb_path) and self.clip_emb_path:
            od_emb_path = os.path.join(self.clip_emb_path, "objects_description_emb.pkl.gz")

        skills_name_path = os.path.join(self.data_root, "skills_name.pkl.gz")
        if not os.path.exists(skills_name_path) and self.clip_emb_path:
            skills_name_path = os.path.join(self.clip_emb_path, "skills_name.pkl.gz")

        sn_emb_path = os.path.join(self.data_root, "skills_name_emb.pkl.gz")
        if not os.path.exists(sn_emb_path) and self.clip_emb_path:
            sn_emb_path = os.path.join(self.clip_emb_path, "skills_name_emb.pkl.gz")

        with gzip.open(objects_description_path, "rb") as f:
            objects_description = pickle.load(f)
        with gzip.open(od_emb_path, "rb") as f:
            od_emb = pickle.load(f)

        self.od_emb = {}
        for i, od in enumerate(objects_description):
            self.od_emb[od] = od_emb[i]

        with gzip.open(skills_name_path, "rb") as f:
            skills_name = pickle.load(f)
        with gzip.open(sn_emb_path, "rb") as f:
            sn_emb = pickle.load(f)

        self.sn_emb = {}
        for i, sn in enumerate(skills_name):
            self.sn_emb[sn] = sn_emb[i]

    def get_text_feat(self, text, cache_dict):
        if cache_dict is not None and text in cache_dict:
            return cache_dict[text]

        if self.clip is None:
            self.clip = CLIPTextEncoder(model_name="ViT-B/32")

        feat = self.clip.encode_text([text], reduce_dim=48)[-1]
        return feat

    def prepare_data(self, data):
        anchor_data = data["anchor"]

        anchor_feat = anchor_data[self.anc_gs_feat_name]
        if "f_distilled" in anchor_data:
            anchor_vfeat = anchor_data["f_distilled"]
            
        obj_desc = anchor_data.get("desc", "unknown object")
        if isinstance(obj_desc, bytes):
            obj_desc = obj_desc.decode("utf-8")
            
        otfeat = self.get_text_feat(obj_desc, self.od_emb)
        anchor_tfeat = np.tile(otfeat, (anchor_feat.shape[0], 1))
        
        if "init_pose" in anchor_data:
            anchor_init_pose = anchor_data["init_pose"]

        action_data = data["action"]
        action_feat = action_data[self.act_gs_feat_name]

        if "f_distilled" in action_data:
            action_vfeat = action_data["f_distilled"]
        else:
            action_vfeat = np.zeros((action_feat.shape[0], 32), dtype=np.float32)

        act_desc = action_data.get("desc", "unknown object")
        if isinstance(act_desc, bytes):
            act_desc = act_desc.decode("utf-8")

        otfeat = self.get_text_feat(act_desc, self.od_emb)
        action_tfeat = np.tile(otfeat, (action_feat.shape[0], 1))

        # 3. Skill embedding
        skill_name = data.get("skill_name", "unknown skill")
        if isinstance(skill_name, bytes):
            skill_name = skill_name.decode("utf-8")
        skill_emb = self.get_text_feat(skill_name, self.sn_emb)

        data_dict = dict(
            anchor_feat=anchor_feat,
            anchor_vfeat=anchor_vfeat,
            anchor_tfeat=anchor_tfeat,

            action_feat=action_feat,
            action_vfeat=action_vfeat,
            action_tfeat=action_tfeat,

            skill_emb=skill_emb,
        )

        if anchor_init_pose is not None:
            data_dict["anchor_init_pose"] = anchor_init_pose

        if "init_pose" in action_data:
            data_dict["action_init_pose"] = action_data["init_pose"]

        if "goal_pose" in action_data:
            data_dict["action_goal_pose"] = action_data["goal_pose"]

        if self.act_goal_gs_feat_name in action_data:
            data_dict["action_goal_feat"] = action_data[self.act_goal_gs_feat_name]

        if "move_link" in data:
            data_dict["move_link"] = data["move_link"]
        elif "move_link" in action_data:
            data_dict["move_link"] = action_data["move_link"]
        else:
            data_dict["move_link"] = False

        data_dict = self.transform(data_dict)
        return data_dict


class DemoTester:
    def __init__(self, cfg, device_id, rank=0, model=None, verbose=False):
        self.cfg = cfg
        self.device_id = device_id
        self.rank = rank
        self.verbose = verbose
        self.device = torch.device(f"cuda:{device_id}")

        log_file = os.path.join(cfg.save_path, "demo.log")
        setup_logger(log_file)

        logger.info(f"[Rank {rank}] 使用 GPU {device_id}")
        logger.info(f"[Rank {rank}] Save path: {cfg.save_path}")

        if model is None:
            logger.info(f"[Rank {rank}] 开始构建模型")
            self.model, self.epoch = self.build_model()
        else:
            self.model = model
            self.epoch = None

        demo_cfg = cfg.data.get("demo", {})
        if demo_cfg is None:
            demo_cfg = {}
        self.processer = Processer(**demo_cfg)

    def build_model(self):
        torch.cuda.set_device(self.device_id)

        model = build_model(self.cfg.model).to(self.device)
        model.eval()

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"[Rank {self.rank}] Num params: {n_parameters}")

        if os.path.isfile(self.cfg.weight):
            logger.info(f"[Rank {self.rank}] Loading weight at: {self.cfg.weight}")
            checkpoint = torch.load(self.cfg.weight, map_location=self.device)

            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if key.startswith("module."):
                    key = key[7:]
                weight[key] = value

            model.load_state_dict(weight, strict=True)
            logger.info(
                f"[Rank {self.rank}] Loaded weight '{self.cfg.weight}' "
                f"(epoch {checkpoint.get('epoch', 'unknown')})"
            )
        else:
            raise RuntimeError(f"No checkpoint found at '{self.cfg.weight}'")

        return model, checkpoint.get("epoch", None)

    def test(self, data, sample_name=None, apply_SVD=False):
        assert apply_SVD is True

        if sample_name is None:
            sample_name = "unknown_sample"

        data_input = clone_sample_for_prepare(data)
        data_dict = self.processer.prepare_data(data_input)

        for key in data_dict.keys():
            if isinstance(data_dict[key], torch.Tensor):
                data_dict[key] = data_dict[key].to(self.device, non_blocking=True)


        self.model.eval()
        with torch.inference_mode():
            pred = self.model(data_dict, mode="sample_goal", apply_SVD=apply_SVD)

        goal = pred["goal"].detach().cpu().numpy()
        goal_wo_svd = pred["goal_wo_svd"][:, :3].detach().cpu().numpy()
        tf = pred["tf"].detach().cpu().numpy()

        saved_traj = None
        if "saved_traj" in pred:
            saved_traj = pred["saved_traj"].detach().cpu().numpy()

        shift_val = None
        if "shift" in data_dict:
            shift_val = data_dict["shift"].detach().cpu().numpy()
            goal += shift_val
            goal_wo_svd += shift_val
            T_ = trans.PosRmat_to_TransMat(shift_val, np.eye(3))
            tf = T_ @ tf @ np.linalg.inv(T_)
            if saved_traj is not None:
                saved_traj[:, :, :3] += shift_val

        error_val = None
        if "error" in pred:
            error_val = pred["error"].detach().cpu().numpy()
            logger.info(f"[Rank {self.rank}] [Sample {sample_name}] error: {error_val}")


        return dict(
            tf=tf,
            goal=goal,
            goal_wo_svd=goal_wo_svd,
            error=error_val,
            saved_obj_traj=saved_traj,
            shift=shift_val,
        )
        
    def save_result(self, data, result_dict, save_name=None):
        assert save_name is not None

        goal_w_svd = result_dict["goal"]
        goal_wo_svd = result_dict["goal_wo_svd"]
        trans = result_dict["tf"]

        sample_vis_dir = os.path.join(self.cfg.vis_result_dir, save_name)
        make_dirs(sample_vis_dir)

        anchor_gs = collect_anchor_gs_feat(data)
        anchor_xyz = anchor_gs[:, :3]

        pcd_anchor = o3d.geometry.PointCloud()
        pcd_anchor.points = o3d.utility.Vector3dVector(anchor_xyz)
        pcd_anchor.paint_uniform_color([0.5, 0.5, 0.5])

        action_gs_raw = data["action"]["gs_feat"]
        action_xyz = action_gs_raw[:, :3]

        pcd_action = o3d.geometry.PointCloud()
        pcd_action.points = o3d.utility.Vector3dVector(action_xyz)
        pcd_action.paint_uniform_color([1, 0, 0])

        pcd_w_svd = o3d.geometry.PointCloud()
        pcd_w_svd.points = o3d.utility.Vector3dVector(goal_w_svd)
        pcd_w_svd.paint_uniform_color([0, 1, 0])

        o3d.io.write_point_cloud(
            os.path.join(sample_vis_dir, "goal_w_svd.ply"),
            pcd_anchor + pcd_action + pcd_w_svd
        )

        pcd_wo_svd = o3d.geometry.PointCloud()
        pcd_wo_svd.points = o3d.utility.Vector3dVector(goal_wo_svd)
        pcd_wo_svd.paint_uniform_color([0, 1, 0])

        o3d.io.write_point_cloud(
            os.path.join(sample_vis_dir, "goal_wo_svd.ply"),
            pcd_anchor + pcd_action + pcd_wo_svd
        )

        action_gs_init = data["action"]["gs_feat"].copy()
        action_gs_init[:, :3] = goal_w_svd

        action_rot_init = data["action"]["gs_feat"][:, 3:7].copy()
        action_rot_init = action_rot_init[:, [1, 2, 3, 0]]  # wxyz -> xyzw

        action_rot_new = R.from_matrix(trans[:3, :3]) * R.from_quat(action_rot_init)
        action_gs_init[:, 3:7] = action_rot_new.as_quat()[:, [3, 0, 1, 2]]  # xyzw -> wxyz

        merged_gs = np.concatenate((anchor_gs.copy(), action_gs_init), axis=0)

        xyz = merged_gs[:, 0:3]
        rot = merged_gs[:, 3:7]
        scale = merged_gs[:, 7:10]
        opacity = merged_gs[:, 10:11]
        f_dc = merged_gs[:, 11:14]
        f_rest = merged_gs[:, 14:]

        gs_save_path = os.path.join(sample_vis_dir, "gs_w_svd.ply")
        self.save_gs(gs_save_path, xyz, f_dc, f_rest, opacity, scale, rot)

    def save_gs(self, path, xyz, f_dc, f_rest, opacity, scale, rot):
            l = ["x", "y", "z", "nx", "ny", "nz"]
            for i in range(f_dc.shape[1]):
                l.append(f"f_dc_{i}")
            for i in range(f_rest.shape[1]):
                l.append(f"f_rest_{i}")
            l.append("opacity")
            for i in range(scale.shape[1]):
                l.append(f"scale_{i}")
            for i in range(rot.shape[1]):
                l.append(f"rot_{i}")

            dtype_full = [(n, "f4") for n in l]
            elements = np.empty(xyz.shape[0], dtype=dtype_full)
            normals = np.zeros_like(xyz)
            attrs = np.concatenate((xyz, normals, f_dc, f_rest, opacity, scale, rot), axis=1)
            elements[:] = list(map(tuple, attrs))
            PlyData([PlyElement.describe(elements, "vertex")]).write(path)


def draw_data(data_raw: dict, pred_goal=None):
    anchor_gs = collect_anchor_gs_feat(data_raw)
    anchor_xyz = anchor_gs[:, :3]

    action_gs = data_raw["action"]["gs_feat"]
    action_xyz = action_gs[:, :3]

    geometries = []

    pcd_anchor = o3d.geometry.PointCloud()
    pcd_anchor.points = o3d.utility.Vector3dVector(anchor_xyz)
    pcd_anchor.paint_uniform_color([0.5, 0.5, 0.5])
    geometries.append(pcd_anchor)

    pcd_action = o3d.geometry.PointCloud()
    pcd_action.points = o3d.utility.Vector3dVector(action_xyz)
    pcd_action.paint_uniform_color([1, 0, 0])
    geometries.append(pcd_action)

    if pred_goal is not None:
        pcd_goal = o3d.geometry.PointCloud()
        pcd_goal.points = o3d.utility.Vector3dVector(pred_goal)
        pcd_goal.paint_uniform_color([0, 1, 0])
        geometries.append(pcd_goal)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    geometries.append(axis)

    o3d.visualization.draw_geometries(geometries, window_name="Demo")


def test_sample(cfg, data_path, tester, visual=False, save_artifacts=True):
    save_name = Path(data_path).stem

    with h5py.File(data_path, "r") as f:
        data = load_nested_dict_from_h5(f, "/")

    result = tester.test(data, sample_name=save_name, apply_SVD=True)

    if save_artifacts:
        anchor_gs = collect_anchor_gs_feat(data)

        npy_saved = dict(
            traj=result["saved_obj_traj"] if result["saved_obj_traj"] is not None else None,
            goal_w_svd=result["goal"],
            goal_wo_svd=result["goal_wo_svd"],
            tf=result["tf"],
            error=result["error"],
            obj_gs_feat=data["action"]["gs_feat"],
            sce_gs_feat=anchor_gs,
        )

        npy_path = os.path.join(cfg.npy_result_dir, save_name + "_result.npy")
        np.save(npy_path, npy_saved)

        tester.save_result(data, result, save_name=save_name)

    if visual:
        draw_data(data, pred_goal=result["goal"])

    return result


def worker_main(
    rank,
    gpu_ids,
    config_file,
    weight_path,
    save_root,
    npy_result_dir,
    vis_result_dir,
    data_folder,
    visual,
    save_artifacts,
    verbose,
    temp_dir,
):
    device_id = gpu_ids[rank]
    torch.cuda.set_device(device_id)

    cfg = build_cfg(
        config_file=config_file,
        weight_path=weight_path,
        save_root=save_root,
        npy_result_dir=npy_result_dir,
        vis_result_dir=vis_result_dir,
    )

    tester = DemoTester(cfg, device_id=device_id, rank=rank, verbose=verbose)

    all_files = sorted([
        x for x in os.listdir(data_folder)
        if x.endswith(".h5") or x.endswith(".hdf5")
    ])

    world_size = len(gpu_ids)
    local_files = all_files[rank::world_size]

    logger.info(
        f"[Rank {rank}] GPU={device_id}, 负责 {len(local_files)} / {len(all_files)} 个样本"
    )

    rows = []

    for idx, h5_file in enumerate(local_files):
        data_path = os.path.join(data_folder, h5_file)
        result = test_sample(
            cfg=cfg,
            data_path=data_path,
            tester=tester,
            visual=visual,
            save_artifacts=save_artifacts,
        )

        err = safe_scalar(result["error"])
        rows.append({
            "file_name": h5_file,
            "error": err,
        })

        if (idx + 1) % 10 == 0 or (idx + 1) == len(local_files):
            logger.info(f"[Rank {rank}] 已完成 {idx + 1}/{len(local_files)} 个样本")

    local_csv = os.path.join(temp_dir, f"rank_{rank}.csv")
    with open(local_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "error"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"[Rank {rank}] 局部结果已写入: {local_csv}")


def merge_rank_csv(temp_dir, final_csv_path, total_samples, log_file):
    setup_logger(log_file)

    all_rows = []
    temp_csvs = sorted(Path(temp_dir).glob("rank_*.csv"))

    for csv_path in temp_csvs:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["error"] = float(row["error"])
                all_rows.append(row)

    all_rows.sort(key=lambda x: x["file_name"])

    with open(final_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "error"])
        writer.writeheader()
        writer.writerows(all_rows)

    errors = [r["error"] for r in all_rows]
    avg_error = np.mean(errors)
    median_error = np.median(errors)
    std_error = np.std(errors)
    min_error = np.min(errors)
    max_error = np.max(errors)

    summary_text = (
        "\n" + "=" * 60 + "\n"
        f"EVALUATION SUMMARY\n"
        f"Total samples:   {total_samples}\n"
        f"Average Error:   {avg_error:.6f}\n"
        f"Median Error:    {median_error:.6f}\n"
        f"Std Deviation:   {std_error:.6f}\n"
        f"Min/Max Error:   [{min_error:.6f} / {max_error:.6f}]\n"
        f"Summary CSV:     {final_csv_path}\n"
        + "=" * 60
    )

    print(summary_text)
    logger.info(summary_text)


if __name__ == "__main__":
    EXP_FOLDER = "exp_new_qf_render/kettle"

    WEIGHT_FILE_NAME = "epoch_2000.pth"

    DATA_FOLDER = (
        "dataset/generated_tasks_v2_gs_source/"
        "Kettle/Kettle-lift_kettle_and_pour_water_v5/gs_dataset_h5_new/test"
    )

    GPU_IDS = [0,1,2,3]

    VISUAL = False

    SAVE_ARTIFACTS = True

    VERBOSE = False


    if len(GPU_IDS) == 0:
        raise ValueError("GPU_IDS 不能为空。")

    if not torch.cuda.is_available():
        raise RuntimeError("当前环境没有可用 CUDA。")

    total_visible_gpus = torch.cuda.device_count()
    for gid in GPU_IDS:
        if gid < 0 or gid >= total_visible_gpus:
            raise ValueError(
                f"GPU id {gid} 越界。当前可见 GPU 数量为 {total_visible_gpus}"
            )

    CONFIG_FILE = os.path.join(EXP_FOLDER, "config.py")
    WEIGHT_PATH = os.path.join(EXP_FOLDER, "model", WEIGHT_FILE_NAME)

    # result_gsdata_v5_new/
    # └── epoch_2000/
    SAVE_ROOT = os.path.join(
        EXP_FOLDER,
        "result_gsdata_v5_new",
        Path(WEIGHT_FILE_NAME).stem
    )

    NPY_RESULT_DIR = os.path.join(SAVE_ROOT, "npy_result")
    VIS_RESULT_DIR = os.path.join(SAVE_ROOT, "vis_result")
    LOG_FILE = os.path.join(SAVE_ROOT, "demo.log")
    FINAL_CSV = os.path.join(SAVE_ROOT, "evaluation_summary.csv")

    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(f"config.py 不存在: {CONFIG_FILE}")
    if not os.path.exists(WEIGHT_PATH):
        raise FileNotFoundError(f"weight 文件不存在: {WEIGHT_PATH}")
    if not os.path.exists(DATA_FOLDER):
        raise FileNotFoundError(f"DATA_FOLDER 不存在: {DATA_FOLDER}")

    os.makedirs(SAVE_ROOT, exist_ok=True)
    os.makedirs(NPY_RESULT_DIR, exist_ok=True)
    os.makedirs(VIS_RESULT_DIR, exist_ok=True)

    setup_logger(LOG_FILE)

    all_h5_files = sorted([
        x for x in os.listdir(DATA_FOLDER)
        if x.endswith(".h5") or x.endswith(".hdf5")
    ])
    total_samples = len(all_h5_files)

    if total_samples == 0:
        raise RuntimeError(f"在目录中没有找到 h5 / hdf5 文件: {DATA_FOLDER}")

    if len(GPU_IDS) > total_samples:
        GPU_IDS = GPU_IDS[:total_samples]

    WORLD_SIZE = len(GPU_IDS)

    if WORLD_SIZE > 1 and VISUAL:
        logger.warning("多进程模式下不建议 visual=True，已自动关闭可视化。")
        VISUAL = False

    logger.info("=" * 60)
    logger.info("Multi-GPU Evaluation Start")
    logger.info(f"CONFIG_FILE   : {CONFIG_FILE}")
    logger.info(f"WEIGHT_PATH   : {WEIGHT_PATH}")
    logger.info(f"DATA_FOLDER   : {DATA_FOLDER}")
    logger.info(f"SAVE_ROOT     : {SAVE_ROOT}")
    logger.info(f"GPU_IDS       : {GPU_IDS}")
    logger.info(f"WORLD_SIZE    : {WORLD_SIZE}")
    logger.info(f"VISUAL        : {VISUAL}")
    logger.info(f"SAVE_ARTIFACTS: {SAVE_ARTIFACTS}")
    logger.info(f"TOTAL_SAMPLES : {total_samples}")
    logger.info("=" * 60)

    temp_dir = tempfile.mkdtemp(prefix=".rank_cache_", dir=SAVE_ROOT)

    mp.set_start_method("spawn", force=True)
    mp.spawn(
        worker_main,
        args=(
            GPU_IDS,
            CONFIG_FILE,
            WEIGHT_PATH,
            SAVE_ROOT,
            NPY_RESULT_DIR,
            VIS_RESULT_DIR,
            DATA_FOLDER,
            VISUAL,
            SAVE_ARTIFACTS,
            VERBOSE,
            temp_dir,
        ),
        nprocs=WORLD_SIZE,
        join=True,
    )

    merge_rank_csv(
        temp_dir=temp_dir,
        final_csv_path=FINAL_CSV,
        total_samples=total_samples,
        log_file=LOG_FILE,
    )

    shutil.rmtree(temp_dir, ignore_errors=True)

    logger.info("全部测试完成。")
