# Normal-Estination
PLY Point Cloud Normal Estimation with PyTorch Geometric

This repository provides a script for estimating normals for point clouds stored in PLY files, leveraging the power of PyTorch Geometric and GPU acceleration.

## Overview

The script efficiently computes normals for large point clouds by:

1.  **Chunking:** Processing the point cloud in manageable chunks to prevent out-of-memory (OOM) errors, especially on GPUs with limited memory.
2.  **K-Nearest Neighbors (KNN):** Using PyTorch Geometric's efficient `knn` function to find the nearest neighbors for each point.
3.  **Principal Component Analysis (PCA):** Estimating the normal vector for each point by performing PCA on its neighborhood.
4.  **Normal Orientation:** Ensuring consistent normal orientation by comparing the normal vector with a vector from the point to the centroid of the point cloud.
5.  **PLY File Handling:** Reading and writing PLY files using the `plyfile` library, preserving existing vertex data and adding normal vectors (nx, ny, nz) to the output.
6. **Directory Processing:** Ability to recursively search input directory and process all PLY Files.

## Dependencies

-   **Python 3.7+**
-   **PyTorch (>= 1.9)**: Tested with 1.13, but earlier versions should work.  Install with CUDA support if you have an NVIDIA GPU.
-   **PyTorch Geometric (>= 2.0)**: Tested with 2.2.0.
-   **NumPy**
-   **plyfile**
-   **torch-scatter, torch-sparse, torch-cluster and torch-spline-conv** (these are usually auto-installed when installing pytorch-geometric)

You can install the necessary dependencies using `pip`:

```bash
pip install torch  # Install PyTorch (with CUDA if available)
pip install torch_geometric
pip install numpy
pip install plyfile

If the pytorch geometric installation fails, try installing each dependency independently:
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f [https://data.pyg.org/whl/torch-1.13.0+$](https://www.google.com/search?q=https://data.pyg.org/whl/torch-1.13.0%2B%24){CUDA}.html

Replacing ${CUDA} by your cuda version (e.g., cu116, cu117, or cpu if you don't have a compatible cuda version) and torch-1.13.0 by your torch version.

Usage
The script is designed to be run from the command line.  Here's how to use it:
python normal_estimation.py --input <input_directory> --output <output_directory> --k <k_neighbors>

--input: The path to the input directory containing PLY files. The script will recursively search this directory for all .ply files.
--output: The path to the output directory where the processed PLY files (with normals) will be saved. The directory structure of the input directory will be mirrored in the output directory. The output directory will be created if it doesn't exist.
--k: (Optional) The number of nearest neighbors to consider for normal estimation. The default value is 50.

