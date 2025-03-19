import torch
import numpy as np
import os
import json
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import tempfile
import math
from collections import defaultdict

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
    # If specific degrees aren't provided, create angles using the enhanced multi-view function
    if degrees is None:
        degrees = generate_multi_view_angles(num_rotations)

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

    Returns:
        save_path (str): Path where the visualization was saved
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

    # Generate a default save path if none provided
    if save_path is None:
        import tempfile
        import os
        temp_dir = tempfile.gettempdir()
        save_path = os.path.join(temp_dir, f"rotations_{id(image)}.png")

    # Always save, never show
    plt.savefig(save_path)
    plt.close()
    print(f"Rotation visualization saved to {save_path}")

    return save_path

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

def combine_rotation_results(all_results, k=10):
    """
    Combine results from multiple rotations with improved similarity scoring.

    Args:
        all_results (list): List of result dictionaries from different rotations
        k (int): Number of results to return

    Returns:
        combined_results (dict): Combined results with normalized similarities
    """
    # Initialize combined results structure
    combined_results = {
        "paths": [],
        "distances": [],
        "part_info": [],
        "similarities": []
    }

    # Create dictionaries to store the best results for each path
    path_to_best_similarity = {}
    path_to_best_distance = {}
    path_to_part_info = {}

    # Track all distances for proper normalization
    all_distances = []

    # Track the frequency of each result and its ranks across rotations
    path_frequency = defaultdict(int)
    path_ranks = defaultdict(list)
    path_distances = defaultdict(list)

    # Process each rotation's results
    for rotation_idx, result in enumerate(all_results):
        for i, path in enumerate(result["paths"]):
            if path is None:
                continue

            # Update frequency count
            path_frequency[path] += 1

            # Get distance or use default
            distance = result["distances"][i] if i < len(result["distances"]) else float('inf')
            all_distances.append(distance)
            path_distances[path].append(distance)

            # Store the rank position in this rotation (0-based)
            path_ranks[path].append(i)

            # Update best similarity and distance
            similarity = result["similarities"][i] if "similarities" in result and i < len(result["similarities"]) else 0
            if path not in path_to_best_similarity or similarity > path_to_best_similarity[path]:
                path_to_best_similarity[path] = similarity

            if path not in path_to_best_distance or distance < path_to_best_distance[path]:
                path_to_best_distance[path] = distance

            # Store part info
            if "part_info" in result and i < len(result["part_info"]) and result["part_info"][i] is not None:
                path_to_part_info[path] = result["part_info"][i]

    # Find min and max distances for normalization
    min_distance = min(all_distances) if all_distances else 0
    max_distance = max(all_distances) if all_distances else 1
    distance_range = max(0.001, max_distance - min_distance)

    # Calculate visual similarity scores that better reflect true differences
    scored_paths = []
    for path, best_distance in path_to_best_distance.items():
        # Get all distances for this path across rotations
        distances = path_distances[path]
        avg_distance = sum(distances) / len(distances)

        # Calculate average rank position (lower is better)
        avg_rank = sum(path_ranks[path]) / len(path_ranks[path])

        # Frequency indicates how many rotations this part appeared in
        freq = path_frequency[path]
        freq_ratio = freq / len(all_results)  # How many rotations contain this part

        # Normalize distance to 0-1 range (0 = best, 1 = worst)
        normalized_distance = (avg_distance - min_distance) / distance_range

        # Convert to similarity score (0-100)
        # Use an exponential decay for distances to better separate similar from dissimilar
        base_similarity = 100 * math.exp(-3 * normalized_distance)

        # Apply rank penalty - items consistently ranked lower should have lower similarity
        rank_factor = math.exp(-0.15 * avg_rank)  # Penalize high average ranks

        # Apply frequency boost for items found in multiple rotations
        # More aggressive boost for higher frequency
        freq_boost = min(30, 15 * math.log2(1 + 2 * freq_ratio))

        # Combine all factors - this creates more separation between visually similar and dissimilar items
        adjusted_similarity = base_similarity * rank_factor + freq_boost

        # Cap at 100%
        adjusted_similarity = min(100, adjusted_similarity)

        # For very high similarities (>80), push them toward 100% to highlight clear matches
        if adjusted_similarity > 80:
            adjusted_similarity = 80 + (adjusted_similarity - 80) * 2
            adjusted_similarity = min(99.5, adjusted_similarity)

        # For low similarities (<40), apply more aggressive reduction to separate poor matches
        if adjusted_similarity < 40:
            adjusted_similarity = adjusted_similarity * 0.8

        scored_paths.append((path, adjusted_similarity, best_distance))

    # Sort by adjusted similarity (descending)
    scored_paths.sort(key=lambda x: x[1], reverse=True)

    # Take top k results
    for path, similarity, distance in scored_paths[:k]:
        combined_results["paths"].append(path)
        combined_results["similarities"].append(similarity)
        combined_results["distances"].append(distance)
        combined_results["part_info"].append(path_to_part_info.get(path, None))

    # Debug output
    if scored_paths:
        min_sim = min([s for _, s, _ in scored_paths[:k]])
        max_sim = max([s for _, s, _ in scored_paths[:k]])
        print(f"Combined {len(all_results)} rotation results into {len(combined_results['paths'])} unique results")
        print(f"Similarity range: {min_sim:.1f}% - {max_sim:.1f}%")

        # Add more detailed output for top results
        print("Top visual similarity scores:")
        for i, (path, sim, _) in enumerate(scored_paths[:min(5, len(scored_paths))]):
            part_name = "unknown"
            if path in path_to_part_info and path_to_part_info[path]:
                part_name = path_to_part_info[path].get("part_name", "unknown")
            print(f"  {i+1}. {part_name}: {sim:.1f}%")

    return combined_results

def rotation_invariant_search(image_encoder, vector_db, query_image_path, k=10, num_rotations=8,
                              use_metadata=False, metadata_encoder=None, fusion_module=None, bom_dir=None):
    """
    Perform rotation-invariant search by encoding multiple rotations of the query image

    Args:
        image_encoder: Image encoder model
        vector_db: Vector database
        query_image_path (str): Path to query image
        k (int): Number of results to return
        num_rotations (int): Number of rotations to try
        use_metadata (bool): Whether to use metadata
        metadata_encoder: Metadata encoder
        fusion_module: Fusion module
        bom_dir (str): Directory containing BOM files

    Returns:
        results (dict): Search results
    """
    # Extract part info from path for metadata lookup
    basename = os.path.basename(query_image_path)
    print(f"\n--- Rotation-Invariant Search Debug ---")
    print(f"Processing file: {basename}")

    parts = basename.split('_')
    parent_step = parts[0] if len(parts) > 1 else "unknown"
    # Join the remaining parts correctly, being careful with the extension
    if len(parts) > 1:
        part_name_with_ext = '_'.join(parts[1:])
        part_name = os.path.splitext(part_name_with_ext)[0]
    else:
        part_name = os.path.splitext(basename)[0]

    print(f"Extracted parent_step: '{parent_step}', part_name: '{part_name}'")

    query_metadata = None

    # Load the image
    try:
        img = Image.open(query_image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image for rotation-invariant search: {e}")
        return {"paths": [], "distances": [], "similarities": []}

    # Calculate angle increment based on number of rotations
    angle_increment = 360 / num_rotations

    # Encode each rotation
    embeddings = []

    for i in range(num_rotations):
        # Calculate rotation angle
        angle = i * angle_increment

        # Rotate image
        rotated_img = img.rotate(angle, resample=Image.BILINEAR, expand=False)

        # Create a temporary file for the rotated image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            rotated_path = tmp.name
            rotated_img.save(rotated_path)

        # Encode the rotated image
        try:
            embedding = image_encoder.encode_image(rotated_path)
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error encoding rotation {angle} degrees: {e}")
        finally:
            # Clean up the temporary file
            os.unlink(rotated_path)

    print(f"Generated {len(embeddings)} rotation embeddings")

    # Look for metadata
    if use_metadata and metadata_encoder is not None and fusion_module is not None and bom_dir is not None:
        # Look for BOM data
        bom_path = os.path.join(bom_dir, f"{parent_step}_bom.json")
        print(f"Looking for BOM file: {bom_path}")

        if os.path.exists(bom_path):
            # Load BOM data
            try:
                print(f"Loading BOM data...")
                bom_data = metadata_encoder.load_bom_data(bom_path)

                # Debug - print part names in BOM
                bom_parts = list(bom_data.keys())
                print(f"Parts in BOM file: {bom_parts[:5]}..." if len(bom_parts) > 5 else f"Parts in BOM file: {bom_parts}")

                # First try exact match
                print(f"Looking for part: '{part_name}' in BOM")
                part_metadata = metadata_encoder.find_part_metadata(bom_data, part_name)
                query_metadata = part_metadata

                # If not found, try variations
                if not part_metadata:
                    print(f"Exact match not found, trying variations...")

                    # Try with and without spaces and underscores
                    variations = [
                        part_name.replace(" ", ""),             # Remove all spaces
                        part_name.replace("_", " "),            # Replace underscores with spaces
                        part_name.replace(" ", "_"),            # Replace spaces with underscores
                        part_name.replace(" _", "_"),           # Fix potential space-underscore combinations
                        part_name.replace("_", " _"),           # Add space before underscores
                        ' '.join(part_name.split('_')),         # Split on underscores and join with spaces
                        '_'.join(part_name.split(' ')),         # Split on spaces and join with underscores
                    ]

                    # Also try with the filename format from STEP/CAD exports
                    if parent_step != "unknown":
                        # Add possible variations without parent step prefix
                        variations.append(basename.replace(f"{parent_step}_", ""))
                        variations.append(os.path.splitext(basename)[0].replace(f"{parent_step}_", ""))

                    # Try each variation
                    for variation in variations:
                        print(f"Trying variation: '{variation}'")
                        part_metadata = metadata_encoder.find_part_metadata(bom_data, variation)
                        if part_metadata:
                            print(f"Found match with variation: '{variation}'")
                            query_metadata = part_metadata
                            break

                    # If still not found, try substring matching (more aggressive)
                    if not query_metadata:
                        print("No exact matches found. Trying substring matching...")
                        for bom_part_name in bom_data.keys():
                            # Clean strings for comparison (remove spaces, underscores, case)
                            clean_query = ''.join(c.lower() for c in part_name if c.isalnum())
                            clean_bom = ''.join(c.lower() for c in bom_part_name if c.isalnum())

                            # Check if one is a substring of the other
                            if clean_query in clean_bom or clean_bom in clean_query:
                                print(f"Found potential substring match: '{bom_part_name}'")
                                query_metadata = metadata_encoder.find_part_metadata(bom_data, bom_part_name)
                                break

                if query_metadata:
                    print(f"Successfully found metadata for part")
                    # Encode metadata
                    metadata_embedding = metadata_encoder.encode_metadata(query_metadata)

                    # Fuse embeddings for each rotation
                    fused_embeddings = []
                    for visual_embedding in embeddings:
                        fused_embedding = fusion_module.fuse(visual_embedding, metadata_embedding)
                        fused_embeddings.append(fused_embedding)

                    # Replace regular embeddings with fused ones
                    embeddings = fused_embeddings
                    print(f"Applied metadata fusion to {len(embeddings)} rotation embeddings")
                else:
                    print(f"Warning: No metadata found for part '{part_name}'")
            except Exception as e:
                print(f"Error processing metadata for {query_image_path}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"BOM file not found: {bom_path}")

    # Stack embeddings if they're tensors
    if isinstance(embeddings[0], torch.Tensor):
        try:
            embeddings = torch.cat(embeddings, dim=0)
        except Exception as e:
            print(f"Error stacking embeddings: {e}")
            return {"paths": [], "distances": [], "similarities": []}

    # Calculate embeddings for each rotation and search
    try:
        # Search with each rotated embedding
        all_results = []
        for i, embedding in enumerate(embeddings):
            # Ensure embedding is properly handled for search
            if isinstance(embedding, torch.Tensor):
                # Make a copy to avoid modifying the original tensor
                search_embedding = embedding.clone()
            else:
                search_embedding = embedding

            # Search the database
            result = vector_db.search(search_embedding, k=k)
            all_results.append(result)

        # Use the new function to combine results with proper similarity scores
        combined_results = combine_rotation_results(all_results, k=k)

        # Add query metadata to results for reranking
        if query_metadata:
            print(f"Adding query metadata to results for reranking")
            combined_results["query_metadata"] = query_metadata
        else:
            print(f"No query metadata available for reranking")

        print(f"Found {len(combined_results['paths'])} results across all rotations")
        print(f"--- End of Rotation-Invariant Search Debug ---\n")

        return combined_results

    except Exception as e:
        print(f"Error in rotation-invariant search: {e}")
        import traceback
        traceback.print_exc()
        return {"paths": [], "distances": [], "similarities": []}

# New function to generate more consistent rotation angles
def generate_multi_view_angles(num_rotations=16):
    """
    Generate a set of rotation angles that better covers the 3D view space

    Args:
        num_rotations (int): Base number of rotations (actual number may be larger)

    Returns:
        list: List of rotation angles in degrees
    """
    # Generate base rotations around z-axis (in-plane rotations)
    z_rotations = np.linspace(0, 360, num_rotations, endpoint=False)

    # For 3D parts, also sample some additional viewpoints
    # Add some standard engineering views (top, front, side, iso)
    standard_views = [0, 90, 180, 270]  # Basic orthographic views

    # Combine all rotations
    all_angles = list(z_rotations) + standard_views

    # Remove duplicates
    unique_angles = list(set(all_angles))

    return sorted(unique_angles)
