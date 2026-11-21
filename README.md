# OG-Gaussian

This repository provides a **partial code release** of OG-Gaussian for review. The complete code will be released upon acceptance of the paper.

## Code overview

The code is organized into three parts, covering simulation data generation, Gaussian reconstruction and segmentation, and goal-state generation.

| Directory             | Purpose                                                                                                                                                                               |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dataset_generation/` | Generates paired initial–goal manipulation scenes, multi-view RGB-D observations, and task annotations using **OmniGibson**. It also manages the training/test object-instance split. |
| `gs_reconstruct/`     | Reconstructs 3D Gaussian representations and separates action and anchor components to build Gaussian datasets, using **GraspSplat + FlashSplat** components.                         |
| `og-gaussian/`        | Contains the model code for object goal-state generation.                                                                                                                             |

## Main environment dependencies

- **Data generation:** OmniGibson, its compatible NVIDIA Isaac Sim environment, and the required object assets. Follow the upstream OmniGibson environment setup.
- **Gaussian reconstruction and segmentation:** Adapted from GraspSplat for reconstruction and FlashSplat for segmentation. Dependencies include CUDA-enabled PyTorch, NumPy, SciPy, Open3D, OpenCV, h5py, plyfile, and PyYAML; MaskCLIP/CLIP and DINOv3 for feature extraction; and the `diff-gaussian-rasterization`, `flashsplat-rasterization`, and `simple-knn` CUDA extensions under `gs_reconstruct/submodules/`.
- **Model:** Python 3.10, CUDA-enabled PyTorch, `torch-scatter`, `spconv`, `flash-attn`, and `gsplat`, together with NumPy, SciPy, and h5py for data processing.

Simulation, reconstruction, and model execution require compatible NVIDIA GPU drivers and CUDA dependencies. Native extensions should be built against the PyTorch and CUDA versions of their respective environments.

## License

Original code in this partial release may be used for non-commercial research evaluation only. Third-party code and its modifications remain subject to their original licenses and copyright notices.
