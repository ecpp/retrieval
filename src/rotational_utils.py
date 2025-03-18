import torch
import numpy as np
import os
import json
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

def generate_rotations(image, num_rotations=8, degrees=None):
    """
    Generate multiple rotated versions of an image.
    
    Args:
        image (PIL.Image): Input image
        num_rotations (int): Number of rotations to generate
        degrees (list): Specific rotation angles in degrees. If None, evenly spaced angles are used.
        
    Returns:
        list: List of rotated images
    """
    # If specific degrees aren't provided, create evenly spaced rotations
    if degrees is None:
        degrees = np.linspace(0, 360, num_rotations, endpoint=False)
    
    rotated_images = []
    
    for angle in degrees:
        # Rotate the image
        rotated = image.rotate(angle, resample=Image.BICUBIC, expand=False)
        rotated_images.append(rotated)
    
    return rotated_images

def visualize_rotations(image, rotated_images, save_path=None):
    """
    Visualize the original image and its rotations
    
    Args:
        image (PIL.Image): Original image
        rotated_images (list): List of rotated images
        save_path (str): Optional path to save the visualization
    """
    n = len(rotated_images) + 1  # +1 for the original image
    fig, axes = plt.subplots(1, n, figsize=(2*n, 2))
    
    # Show original image
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    # Show rotated images
    for i, rotated in enumerate(rotated_images):
        axes[i+1].imshow(rotated)
        axes[i+1].set_title(f"Rotated {i+1}")
        axes[i+1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def batch_rotate_tensor(tensor, num_rotations=8):
    """
    Generate rotated versions of a tensor batch
    
    Args:
        tensor (torch.Tensor): Input tensor of shape [B, C, H, W]
        num_rotations (int): Number of rotations to generate
        
    Returns:
        torch.Tensor: Tensor containing all rotations [B*num_rotations, C, H, W]
    """
    B, C, H, W = tensor.shape
    degrees = torch.linspace(0, 360, num_rotations, endpoint=False)
    
    rotated_tensors = []
    for i in range(B):
        for angle in degrees:
            rotated = TF.rotate(tensor[i:i+1], angle.item())
            rotated_tensors.append(rotated)
    
    return torch.cat(rotated_tensors, dim=0)

def rotation_invariant_search(image_encoder, vector_db, query_image_path, k=10, num_rotations=8,
                              use_metadata=False, metadata_encoder=None, fusion_module=None, bom_dir=None):
    """
    Perform rotation-invariant search
    
    Args:
        image_encoder: Encoder for generating embeddings
        vector_db: Vector database for similarity search
        query_image_path (str): Path to query image
        k (int): Number of results to return
        num_rotations (int): Number of rotations to try
        use_metadata (bool): Whether to use metadata
        metadata_encoder: Encoder for metadata
        fusion_module: Module for fusing visual and metadata embeddings
        bom_dir (str): Directory containing BOM data
        
    Returns:
        dict: Search results
    """
    # Load the query image
    query_image = Image.open(query_image_path).convert('RGB')
    
    # Generate rotated versions
    rotated_images = generate_rotations(query_image, num_rotations)
    
    # Encode each rotation
    embeddings = []
    for img in rotated_images:
        # Convert PIL image to tensor and process with encoder
        input_tensor = image_encoder.transform(img).unsqueeze(0).to(image_encoder.device)
        with torch.no_grad():
            if image_encoder.model_name == 'dinov2' or image_encoder.model_name == 'vit':
                outputs = image_encoder.model(input_tensor)
                embedding = outputs.last_hidden_state[:, 0].cpu()
            elif image_encoder.model_name == 'resnet50':
                embedding = image_encoder.model(input_tensor).squeeze().cpu()
        embeddings.append(embedding)
    
    # Process metadata if enabled and components are available
    if use_metadata and metadata_encoder is not None and fusion_module is not None and bom_dir is not None:
        # Extract part info from the image path
        parts = os.path.basename(query_image_path).split('_')
        if len(parts) > 1:
            parent_step = parts[0]
            part_name = '_'.join(parts[1:]).split('.')[0]  # Remove extension
        else:
            parent_step = "unknown"
            part_name = os.path.splitext(os.path.basename(query_image_path))[0]
        
        # Look for BOM data
        bom_path = os.path.join(bom_dir, f"{parent_step}_bom.json")
        
        if os.path.exists(bom_path):
            # Load BOM data
            try:
                bom_data = metadata_encoder.load_bom_data(bom_path)
                
                # Find metadata for this part
                part_metadata = metadata_encoder.find_part_metadata(bom_data, part_name)
                
                if part_metadata:
                    # Encode metadata
                    metadata_embedding = metadata_encoder.encode_metadata(part_metadata)
                    
                    # Fuse embeddings for each rotation
                    fused_embeddings = []
                    for visual_embedding in embeddings:
                        fused_embedding = fusion_module.fuse(visual_embedding, metadata_embedding)
                        fused_embeddings.append(fused_embedding)
                    
                    # Replace regular embeddings with fused ones
                    embeddings = fused_embeddings
            except Exception as e:
                print(f"Error processing metadata for {query_image_path}: {e}")
    
    # Stack embeddings if they're tensors
    if isinstance(embeddings[0], torch.Tensor):
        try:
            embeddings = torch.cat(embeddings, dim=0)
        except Exception as e:
            print(f"Error stacking embeddings: {e}")
            # Fall back to processing individually
            pass
    
    # Search with each embedding
    all_results = []
    for emb in embeddings:
        results = vector_db.search(emb, k=k)
        all_results.append(results)
    
    # Combine results
    combined_results = combine_rotation_results(all_results, k)
    
    return combined_results

def combine_rotation_results(all_results, k=10):
    """
    Combine results from multiple rotations, taking the best match for each item
    
    Args:
        all_results (list): List of search results from different rotations
        k (int): Number of results to return
        
    Returns:
        dict: Combined search results
    """
    # Create a dictionary to store the best distance for each path
    best_distances = {}
    best_indices = {}
    part_info = {}
    
    # Process each result set
    for results in all_results:
        for i, (path, distance, idx) in enumerate(zip(results["paths"], results["distances"], results["indices"])):
            if path is None:
                continue
                
            # Update if this is the best distance so far for this path
            if path not in best_distances or distance < best_distances[path]:
                best_distances[path] = distance
                best_indices[path] = idx
                
                # Store part info if available
                if "part_info" in results and i < len(results["part_info"]):
                    part_info[path] = results["part_info"][i]
    
    # Sort paths by distance
    sorted_paths = sorted(best_distances.keys(), key=lambda x: best_distances[x])
    
    # Take top k
    top_paths = sorted_paths[:k]
    top_distances = [best_distances[p] for p in top_paths]
    top_indices = [best_indices[p] for p in top_paths]
    
    # Get part info for top paths
    top_part_info = []
    for path in top_paths:
        if path in part_info:
            top_part_info.append(part_info[path])
        else:
            top_part_info.append(None)
    
    # Create final results
    combined_results = {
        "paths": top_paths,
        "distances": top_distances,
        "indices": top_indices,
        "part_info": top_part_info
    }
    
    return combined_results
