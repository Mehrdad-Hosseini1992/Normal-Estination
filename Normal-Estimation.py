import os
import logging
import numpy as np
import torch
from torch_geometric.nn import knn
from pathlib import Path
from plyfile import PlyData, PlyElement

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def estimate_normals_torchgeo(points, k=50, device='cuda', chunk_size=500000):
    """Process in chunks to avoid OOM errors"""
    normals = torch.zeros_like(points)
    
    # Process in chunks
    for i in range(0, points.size(0), chunk_size):
        chunk = points[i:i+chunk_size].to(device)
        
        # KNN search
        batch = torch.zeros(chunk.size(0), dtype=torch.long, device=device)
        edge_index = knn(chunk, chunk, k, batch, batch)
        
        # PCA normal estimation
        neighbors = chunk[edge_index[1]].view(-1, k, 3)
        centered = neighbors - chunk.unsqueeze(1)
        cov = torch.matmul(centered.transpose(1, 2), centered) / k
        _, _, v = torch.linalg.svd(cov)
        chunk_normals = v[:, :, 2]
        
        # Orientation
        vectors = chunk - chunk.mean(0)
        dot_product = torch.sum(chunk_normals * vectors, dim=1)
        chunk_normals[dot_product < 0] *= -1
        
        normals[i:i+chunk_size] = chunk_normals.cpu()
        
        # Cleanup
        del chunk, edge_index, neighbors, centered, cov, v, chunk_normals
        torch.cuda.empty_cache()
        
    return normals.numpy()
def process_ply_file(input_path, output_path, k_neighbors=50):
    try:
        # Read input PLY
        ply_data = PlyData.read(input_path)
        vertex_data = ply_data['vertex'].data
        
        # Extract coordinates
        xyz = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T
        
        # Convert to tensor and estimate normals
        xyz_tensor = torch.tensor(xyz, dtype=torch.float32)
        
        # Process with chunking
        normals = estimate_normals_torchgeo(
            xyz_tensor, 
            k=k_neighbors,
            device='cuda',
            chunk_size=500000  # Adjust based on your GPU
        )
        # Create new vertex structure
        new_dtype = vertex_data.dtype.descr + [('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')]
        new_vertex = np.zeros(vertex_data.shape, dtype=new_dtype)
        
        # Copy existing fields
        for field in vertex_data.dtype.names:
            new_vertex[field] = vertex_data[field]
            
        # Add normals
        new_vertex['nx'] = normals[:, 0]
        new_vertex['ny'] = normals[:, 1]
        new_vertex['nz'] = normals[:, 2]
        
        # Save output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        PlyData([PlyElement.describe(new_vertex, 'vertex')], text=True).write(output_path)
        logging.info(f"Processed: {input_path} -> {output_path}")
        
    except Exception as e:
        logging.error(f"Error processing {input_path}: {str(e)}")

def process_directory(input_root, output_root, k_neighbors=50):
    input_path = Path(input_root)
    output_path = Path(output_root)
    
    ply_files = list(input_path.rglob('*.ply'))
    
    if not ply_files:
        logging.warning(f"No PLY files found in {input_path}")
        return
    
    logging.info(f"Processing {len(ply_files)} files with k={k_neighbors}")
    
    for file in ply_files:
        rel_path = file.relative_to(input_path)
        out_file = output_path / rel_path
        
        process_ply_file(
            input_path=str(file),
            output_path=str(out_file),
            k_neighbors=k_neighbors
        )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PLY Normal Estimation')
    parser.add_argument('--input', required=True, help='Input directory')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--k', type=int, default=50, help='Number of neighbors for normal estimation')
    
    args = parser.parse_args()
    
    # Verify CUDA
    if torch.cuda.is_available():
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logging.warning("Using CPU - processing will be slower")
    
    # Start processing
    process_directory(
        input_root=args.input,
        output_root=args.output,
        k_neighbors=args.k
    )