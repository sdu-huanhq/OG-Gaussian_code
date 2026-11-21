import os
import sys
import yaml
import pickle
import argparse
import traceback
import torch
import warnings
from loguru import logger

warnings.filterwarnings("ignore")

current_dir = os.getcwd()
sys.path.append(current_dir)

from pipeline_utils_3dgs.reconstruct import run_reconstruction
from pipeline_utils_3dgs.segment import run_segmentation
from pipeline_utils_3dgs.sample import process_sampling_and_h5

def setup_env_vars(weights_root):
    weights_root = os.path.abspath(weights_root)
    os.makedirs(weights_root, exist_ok=True)
    os.environ['TORCH_HOME'] = weights_root
    torch.hub.set_dir(weights_root)
    os.environ['XDG_CACHE_HOME'] = weights_root 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_path", type=str, required=True, help="Path to input .pkl file")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    
    file_name = os.path.basename(args.pkl_path)
    scene_name = os.path.splitext(file_name)[0]
    
    intermediate_scene_path = os.path.join(config['intermediate_dir'], scene_name)
    os.makedirs(intermediate_scene_path, exist_ok=True)
    
    output_h5_dir = config['final_output_dir']
    os.makedirs(output_h5_dir, exist_ok=True)
    h5_path = os.path.join(output_h5_dir, f"{scene_name}.h5")

    internal_gpu_id = 0 
    
    try:
        # Load Data
        with open(args.pkl_path, "rb") as f:
            pkl_data = pickle.load(f)

        # Step 1: Reconstruction
        logger.info(f"[{scene_name}] Running Reconstruction...")
        ply_check = os.path.join(intermediate_scene_path, "point_cloud", 
                                f"iteration_{config['optimization_params']['iterations']}", 
                                "point_cloud.ply")
        
        should_run_recon = config.get('rewrite', False) or not os.path.exists(ply_check)
        
        if should_run_recon:
            weights_root = config.get('weights_root', './weights')
            
            setup_env_vars(weights_root)
            
            run_reconstruction(config, pkl_data, scene_name, intermediate_scene_path, internal_gpu_id)
        else:
            logger.info(f"[{scene_name}] Reconstruction skipped (exists).")

        # Step 2: Segmentation
        logger.info(f"[{scene_name}] Running Segmentation...")
        seg_paths = run_segmentation(config.get('segmentation', {}), intermediate_scene_path, internal_gpu_id)
        
        if seg_paths is None:
            raise RuntimeError("Segmentation returned None")

        # Step 3: Sampling & H5
        logger.info(f"[{scene_name}] Running Sampling...")
        process_sampling_and_h5(config, seg_paths, pkl_data, h5_path, internal_gpu_id)
        
        logger.success(f"[{scene_name}] Processed successfully.")
        
    except Exception as e:
        logger.error(f"[{scene_name}] FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()