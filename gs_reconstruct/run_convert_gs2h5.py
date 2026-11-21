
import os
import sys
import yaml
import argparse
import subprocess
import time
import threading
from queue import Queue
from tqdm import tqdm
from loguru import logger

def worker_thread(gpu_queue, file_queue, config_path, pbar):
    while True:
        if file_queue.empty():
            break
            
        pkl_file = file_queue.get()
        
        gpu_id = gpu_queue.get()
        
        try:
            file_name = os.path.basename(pkl_file)
            # logger.info(f"Starting {file_name} on GPU {gpu_id}")
            
            cmd = [
                sys.executable,
                "process_single_scene_3dgs.py",
                "--pkl_path", pkl_file,
                "--config", config_path
            ]
            
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Error processing {file_name} on GPU {gpu_id}:\n{result.stderr}")
            else:
                # logger.success(f"Finished {file_name}")
                pass
                
        except Exception as e:
            logger.error(f"Thread exception: {e}")
        finally:
            gpu_queue.put(gpu_id)
            file_queue.task_done()
            pbar.update(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()
    
    if not os.path.exists("process_single_scene_3dgs.py"):
        logger.error("Please ensure process_single_scene_3dgs.py is in the current directory.")
        return

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    source_path = config['source_pkl_path']
    files = sorted([os.path.join(source_path, f) for f in os.listdir(source_path) 
                    if f.endswith('.pkl') and f != 'dataset.pkl'])
    
    logger.info(f"Found {len(files)} files to process.")
    
    file_queue = Queue()
    for f in files:
        file_queue.put(f)
        
    gpu_queue = Queue()
    gpu_ids = config['gpu_ids']
    
    total_workers = 0
    for gid in gpu_ids:
        for _ in range(config['processes_per_gpu']):
            gpu_queue.put(gid)
            total_workers += 1
            
    logger.info(f"Launching {total_workers} worker threads ({len(gpu_ids)} GPUs * {config['processes_per_gpu']} jobs).")
    
    threads = []
    pbar = tqdm(total=len(files), desc="Total Progress")
    
    for _ in range(total_workers):
        t = threading.Thread(target=worker_thread, args=(gpu_queue, file_queue, args.config, pbar))
        t.start()
        threads.append(t)
        # time.sleep(0.5)
        
    for t in threads:
        t.join()
        
    pbar.close()
    logger.success("All tasks completed.")

if __name__ == "__main__":
    main()