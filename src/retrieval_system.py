import os
import yaml
import json
import torch
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from .image_encoder import ImageEncoder
from .vector_database import VectorDatabase
from .data_processor import DataProcessor
from .evaluator import RetrievalEvaluator
from .rotational_utils import rotation_invariant_search
import math
import numpy as np

# Import new components for Phase 2
try:
    from .metadata_encoder import MetadataEncoder
    from .fusion_module import FusionModule
    METADATA_AVAILABLE = True
    print("Successfully imported metadata components")
except ImportError as e:
    METADATA_AVAILABLE = False
    print(f"ImportError: Could not import metadata components: {e}")

# Check if rotation utilities are available
ROTATION_AVAILABLE = True  # Since we successfully imported rotation_invariant_search

class RetrievalSystem:
    """
    Main class for the CAD part retrieval system
    """
    def __init__(self, config_path="config/config.yaml"):
        """
        Initialize the retrieval system

        Args:
            config_path (str): Path to the configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Create directories
        os.makedirs(self.config["data"]["input_dir"], exist_ok=True)
        os.makedirs(self.config["data"]["output_dir"], exist_ok=True)
        os.makedirs(self.config["data"]["database_dir"], exist_ok=True)
        os.makedirs(os.path.dirname(self.config["indexing"]["index_file"]), exist_ok=True)

        # Initialize image encoder
        self.image_encoder = ImageEncoder(
            model_name=self.config["model"]["name"],
            pretrained=self.config["model"]["pretrained"],
            embedding_dim=self.config["model"]["embedding_dim"],
            image_size=self.config["model"]["image_size"]
        )

        # Initialize vector database
        self.vector_db = VectorDatabase(
            embedding_dim=self.config["model"]["embedding_dim"],
            index_file=self.config["indexing"]["index_file"],
            metadata_file=self.config["indexing"]["metadata_file"]
        )

        # Initialize metadata components
        self.use_metadata = self.config.get("metadata", {}).get("enabled", False)
        self.metadata_encoder = None
        self.fusion_module = None

        if METADATA_AVAILABLE and self.use_metadata:
            # Create metadata directory
            bom_dir = self.config.get("metadata", {}).get("bom_dir", "data/output/bom")
            os.makedirs(bom_dir, exist_ok=True)

            # Initialize metadata encoder
            # Initialize metadata encoder with autoencoder architecture
            self.metadata_encoder = MetadataEncoder(
                output_dim=self.config.get("metadata", {}).get("embedding_dim", 256),
                hidden_dims=self.config.get("metadata", {}).get("hidden_dims", [512, 384])
            )

            # Try to load a pre-trained model if it exists
            model_path = self.config.get("metadata", {}).get("model_path", "models/metadata_autoencoder.pt")
            if os.path.exists(model_path):
                self.metadata_encoder.load_trained_model(model_path)

            # Initialize fusion module
            self.fusion_module = FusionModule(
                visual_dim=self.config["model"]["embedding_dim"],
                metadata_dim=self.config.get("metadata", {}).get("embedding_dim", 256),
                output_dim=self.config["model"]["embedding_dim"],
                fusion_method=self.config.get("metadata", {}).get("fusion_method", "concat")
            )

            print(f"Metadata integration enabled with {self.config.get('metadata', {}).get('fusion_method', 'concat')} fusion")
        else:
            if not METADATA_AVAILABLE:
                print("Metadata components not available - reverting to visual-only search")
            self.use_metadata = False

        # Initialize data processor
        self.data_processor = DataProcessor(
            output_dir=self.config["data"]["output_dir"]
        )

        # Initialize evaluator
        self.evaluator = RetrievalEvaluator(
            image_encoder=self.image_encoder,
            vector_db=self.vector_db,
            output_dir=os.path.join(self.config["data"]["output_dir"], "evaluation")
        )

        print(f"Retrieval system initialized with {self.config['model']['name']} encoder")

    def extract_part_info(self, image_path):
        """
        Extract part information from an image path

        Args:
            image_path (str): Path to the image

        Returns:
            info (dict): Dictionary with part information
        """
        try:
            # Get the filename without extension
            filename = os.path.basename(image_path)
            name_without_ext = os.path.splitext(filename)[0]

            # Try to extract parent STEP and part name
            # Assuming format is something like "step_id_part_name.png"
            parts = name_without_ext.split('_')

            if len(parts) > 1:
                # First part is the STEP ID
                parent_step = parts[0]
                # Rest is the part name
                part_name = '_'.join(parts[1:])
            else:
                # If we can't split, use the whole filename as part name
                parent_step = "unknown"
                part_name = name_without_ext

            return {
                "parent_step": parent_step,
                "part_name": part_name
            }
        except Exception as e:
            print(f"Error extracting part info from {image_path}: {e}")
            return {
                "parent_step": "unknown",
                "part_name": os.path.basename(image_path)
            }

    def ingest_data(self, dataset_dir):
        """
        Ingest data from a directory of processed STEP files

        Args:
            dataset_dir (str): Directory containing STEP output directories

        Returns:
            all_parts (list): List of processed part information
        """
        # Process the dataset
        all_parts = self.data_processor.process_dataset(dataset_dir)

        # Copy images to a flat structure
        self.data_processor.copy_to_flat_structure(all_parts)

        # Save processed data
        self.data_processor.save_processed_data(all_parts)

        # Copy BOM files to the BOM directory if metadata is enabled
        if self.use_metadata:
            bom_dir = self.config.get("metadata", {}).get("bom_dir", "data/output/bom")
            os.makedirs(bom_dir, exist_ok=True)

            # Copy BOM files from the source to the BOM directory
            bom_files_copied = 0
            for root, dirs, files in os.walk(dataset_dir):
                for file in files:
                    if file.endswith("_bom.json"):
                        src_path = os.path.join(root, file)
                        dst_path = os.path.join(bom_dir, file)

                        # Copy file
                        try:
                            with open(src_path, 'r') as src, open(dst_path, 'w') as dst:
                                dst.write(src.read())
                            bom_files_copied += 1
                        except Exception as e:
                            print(f"Error copying BOM file {file}: {e}")

            print(f"Copied {bom_files_copied} BOM files to {bom_dir}")

            # Note: Autoencoder training is now a separate process and not automatically done during ingestion
            print(f"Copied {bom_files_copied} BOM files to {bom_dir}")
            print("To train the autoencoder, use the 'train-autoencoder' command")

        # Return the processed parts
        return all_parts

    def build_index(self, image_dir=None):
        """
        Build the vector index from images

        Args:
            image_dir (str): Directory containing images to index

        Returns:
            count (int): Number of images indexed
        """
        if image_dir is None:
            image_dir = os.path.join(self.config["data"]["output_dir"], "images")

        if self.use_metadata and self.metadata_encoder and self.fusion_module:
            print("Building index with metadata integration...")

            # Check if autoencoder model is trained/loaded
            model_path = self.config.get("metadata", {}).get("model_path", "models/metadata_autoencoder.pt")
            if not os.path.exists(model_path):
                # Provide a clear error message with exact command to run
                error_msg = f"\nERROR: Autoencoder model not found at {model_path}\n\n"
                error_msg += "You've enabled metadata integration (--use-metadata), but the autoencoder model is not trained.\n"
                error_msg += "Please train the autoencoder first using the following command:\n\n"
                error_msg += "    python main.py train-autoencoder --use-metadata\n\n"
                error_msg += "Once training is complete, you can build the index with metadata integration.\n"
                raise ValueError(error_msg)

            # Build the index with metadata-enhanced embeddings
            count = self._build_index_with_metadata(image_dir)
        else:
            if self.use_metadata:
                print("Metadata integration was requested but components are not available.")
                print("Falling back to visual features only...")
            else:
                print("Building index with visual features only...")

            # Build the index with regular visual embeddings
            count = self.vector_db.build_from_directory(
                self.image_encoder,
                image_dir,
                batch_size=self.config["training"]["batch_size"],
                metadata_func=self.extract_part_info
            )

        # Save the index
        self.vector_db.save()

        return count

    def _build_index_with_metadata(self, image_dir):
        """
        Build the index with metadata integration

        Args:
            image_dir (str): Directory containing images to index

        Returns:
            count (int): Number of images indexed
        """
        # Collect all image paths
        image_paths = []
        extensions = ('.png', '.jpg', '.jpeg')
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(extensions):
                    full_path = os.path.join(root, file)
                    image_paths.append(full_path)

        if not image_paths:
            print(f"No images found in {image_dir} with extensions {extensions}")
            return 0

        print(f"Found {len(image_paths)} images, generating embeddings with metadata...")

        # Dictionary to cache BOM data to avoid repeated loading
        bom_cache = {}

        # Process images in batches
        batch_size = self.config["training"]["batch_size"]
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i:i+batch_size]

            # Get visual embeddings
            batch_visual_embeddings = self.image_encoder.encode_batch(batch_paths, batch_size)

            # Prepare for metadata integration
            batch_metadata = []
            batch_metadata_embeddings = []

            # For each image, extract part info and find metadata
            for path in batch_paths:
                part_info = self.extract_part_info(path)
                batch_metadata.append(part_info)

                # Try to find BOM data for this part
                metadata_found = False
                if part_info and part_info.get("parent_step"):
                    parent_step = part_info["parent_step"]
                    part_name = part_info["part_name"]

                    # Get BOM path
                    bom_dir = self.config.get("metadata", {}).get("bom_dir")
                    bom_path = os.path.join(bom_dir, f"{parent_step}_bom.json")

                    # Check if BOM exists
                    if os.path.exists(bom_path):
                        try:
                            # Load BOM data (or get from cache)
                            if bom_path not in bom_cache:
                                bom_cache[bom_path] = self.metadata_encoder.load_bom_data(bom_path)

                            bom_data = bom_cache[bom_path]

                            # Find metadata for this part
                            part_metadata = self.metadata_encoder.find_part_metadata(bom_data, part_name)

                            if part_metadata:
                                # Encode metadata
                                metadata_embedding = self.metadata_encoder.encode_metadata(part_metadata)
                                batch_metadata_embeddings.append(metadata_embedding)
                                metadata_found = True
                        except Exception as e:
                            print(f"Error processing metadata for {path} (part: {part_name}): {e}")
                            # Add to problematic files log for reference
                            with open(os.path.join(self.config["data"]["output_dir"], "problematic_bom_files.txt"), "a") as log:
                                log.write(f"{bom_path}: {e}\n")

                # If no metadata found, use a zero embedding of the right size
                if not metadata_found:
                    zero_embedding = torch.zeros((1, self.metadata_encoder.output_dim),
                                                device=self.metadata_encoder.device)
                    batch_metadata_embeddings.append(zero_embedding)

            # Combine all metadata embeddings
            if batch_metadata_embeddings:
                metadata_embeddings = torch.cat(batch_metadata_embeddings, dim=0)

                # Ensure dimensions match
                if metadata_embeddings.shape[0] != batch_visual_embeddings.shape[0]:
                    print(f"Warning: Metadata batch size ({metadata_embeddings.shape[0]}) doesn't match visual batch size ({batch_visual_embeddings.shape[0]})")
                    # Adjust to the smaller size
                    min_size = min(metadata_embeddings.shape[0], batch_visual_embeddings.shape[0])
                    metadata_embeddings = metadata_embeddings[:min_size]
                    batch_visual_embeddings = batch_visual_embeddings[:min_size]
                    batch_paths = batch_paths[:min_size]
                    batch_metadata = batch_metadata[:min_size]

                # Fuse embeddings
                batch_fused_embeddings = torch.zeros_like(batch_visual_embeddings)
                for j in range(batch_visual_embeddings.shape[0]):
                    visual_emb = batch_visual_embeddings[j:j+1]
                    metadata_emb = metadata_embeddings[j:j+1]
                    batch_fused_embeddings[j:j+1] = self.fusion_module.fuse(visual_emb, metadata_emb)

                # Add to index
                self.vector_db.add_embeddings(batch_fused_embeddings, batch_paths, batch_metadata)
            else:
                # If no metadata embeddings were generated, just use visual
                self.vector_db.add_embeddings(batch_visual_embeddings, batch_paths, batch_metadata)

        print(f"Added {len(image_paths)} images to the index")
        return len(image_paths)

    def retrieve_similar(self, query_image_path, k=10, rotation_invariant=True, num_rotations=8):
        """
        Retrieve similar parts to a query image

        Args:
            query_image_path (str): Path to the query image
            k (int): Number of results to retrieve
            rotation_invariant (bool): Whether to perform rotation-invariant search
            num_rotations (int): Number of rotations to try if rotation_invariant is True

        Returns:
            results (dict): Search results
        """
        # Validate input
        if not os.path.exists(query_image_path):
            print(f"Error: Query image path does not exist: {query_image_path}")
            return {"paths": [], "distances": [], "similarities": []}

        try:
            query_part_info = self.extract_part_info(query_image_path)
            query_metadata = None

            print(f"\n--- Beginning Retrieval Process ---")
            print(f"Query image: {query_image_path}")
            print(f"Part info extracted: {query_part_info}")

            # Get query part metadata if available
            if self.use_metadata and query_part_info and query_part_info.get("parent_step"):
                bom_dir = self.config.get("metadata", {}).get("bom_dir")
                bom_path = os.path.join(bom_dir, f"{query_part_info['parent_step']}_bom.json")

                print(f"Looking for BOM file: {bom_path}")
                if os.path.exists(bom_path):
                    print(f"BOM file exists, loading data...")
                    bom_data = self.metadata_encoder.load_bom_data(bom_path)

                    # Print the part names in the BOM to help debug
                    part_names = list(bom_data.keys())
                    print(f"Available parts in BOM: {part_names[:10]}...")

                    part_name = query_part_info.get("part_name")
                    print(f"Looking for part: '{part_name}' in BOM")

                    # Try exact match first
                    query_metadata = self.metadata_encoder.find_part_metadata(bom_data, part_name)

                    # If not found, try some common variations
                    if not query_metadata:
                        print(f"Part not found with exact name, trying alternate formats...")
                        # Try with additional space variations
                        alternate_names = [
                            part_name.replace(" ", ""),  # No spaces
                            part_name.replace("_", " _"), # Space before underscore
                            part_name.replace(" _", "_"),  # No space before underscore
                        ]

                        for alt_name in alternate_names:
                            print(f"Trying alternate name: '{alt_name}'")
                            query_metadata = self.metadata_encoder.find_part_metadata(bom_data, alt_name)
                            if query_metadata:
                                print(f"Found metadata using alternate name: '{alt_name}'")
                                break

                        # If still not found, try a fuzzy match approach
                        if not query_metadata:
                            print(f"Trying fuzzy matching approach...")
                            best_match = None
                            best_score = 0

                            for bom_part_name in bom_data.keys():
                                # Calculate similarity score based on character overlap
                                clean_query = ''.join(c.lower() for c in part_name if c.isalnum())
                                clean_bom = ''.join(c.lower() for c in bom_part_name if c.isalnum())

                                # If one is a substring of the other, likely a match
                                if clean_query in clean_bom or clean_bom in clean_query:
                                    match_score = len(set(clean_query) & set(clean_bom)) / max(len(clean_query), len(clean_bom))
                                    if match_score > best_score:
                                        best_score = match_score
                                        best_match = bom_part_name

                            if best_match and best_score > 0.8:  # Only use matches with high confidence
                                print(f"Found fuzzy match: '{best_match}' (score: {best_score:.2f})")
                                query_metadata = self.metadata_encoder.find_part_metadata(bom_data, best_match)
                else:
                    print(f"BOM file not found: {bom_path}")

            if query_metadata:
                print(f"Successfully retrieved metadata for query part")
            else:
                print(f"Warning: No metadata found for query part!")

            # Perform the search
            if rotation_invariant and ROTATION_AVAILABLE:
                print("Using rotation-invariant search...")
                # Use rotation-invariant search
                results = rotation_invariant_search(
                    self.image_encoder,
                    self.vector_db,
                    query_image_path,
                    k=k,
                    num_rotations=num_rotations,
                    use_metadata=self.use_metadata,
                    metadata_encoder=self.metadata_encoder,
                    fusion_module=self.fusion_module,
                    bom_dir=self.config.get("metadata", {}).get("bom_dir")
                )

                # Check if rotational search provided metadata
                if "query_metadata" in results:
                    print("Rotational search provided metadata")
                    query_metadata = results.pop("query_metadata")  # Remove and use for reranking
                else:
                    print("Rotational search did not provide metadata")
            else:
                print("Using regular search...")
                # Regular search
                # Get the query embedding
                query_embedding = self.image_encoder.encode_image(query_image_path)

                # If metadata is enabled, try to find and incorporate it
                if self.use_metadata and self.metadata_encoder and self.fusion_module:
                    # Extract part info from the image path
                    part_info = self.extract_part_info(query_image_path)

                    # Look for BOM data
                    bom_dir = self.config.get("metadata", {}).get("bom_dir")
                    if bom_dir and part_info:
                        # Construct BOM path
                        bom_path = os.path.join(bom_dir, f"{part_info['parent_step']}_bom.json")

                        if os.path.exists(bom_path):
                            # Load BOM data
                            bom_data = self.metadata_encoder.load_bom_data(bom_path)

                            # Find metadata for this part
                            part_metadata = self.metadata_encoder.find_part_metadata(bom_data, part_info.get("part_name"))

                            if part_metadata:
                                # Encode metadata
                                metadata_embedding = self.metadata_encoder.encode_metadata(part_metadata)

                                # Fuse embeddings
                                query_embedding = self.fusion_module.fuse(query_embedding, metadata_embedding)

                # Search the database
                results = self.vector_db.search(query_embedding, k=k)

            print(f"Initial search complete, got {len(results['paths'])} results")
            print(f"Use metadata for reranking: {self.use_metadata}, Query metadata available: {query_metadata is not None}")

            # Apply size-based reranking if metadata is available
            if self.use_metadata and query_metadata:
                # Get the size weight from config
                size_weight = self.config.get("metadata", {}).get("size_weight", 0.3)
                results = self._rerank_by_size(results, query_metadata, size_weight)
            else:
                print("Skipping size-based reranking due to missing metadata")

            print(f"--- Retrieval Process Complete ---\n")
            return results
        except Exception as e:
            print(f"Error retrieving similar parts: {e}")
            import traceback
            traceback.print_exc()
            return {"paths": [], "distances": [], "similarities": []}

    def _rerank_by_size(self, results, query_metadata, size_weight=0.3):
        """
        Rerank results based on size similarity

        Args:
            results (dict): Search results
            query_metadata (dict): Query part metadata
            size_weight (float): Weight to give to size similarity (0-1)

        Returns:
            results (dict): Reranked results
        """
        if not query_metadata or "properties" not in query_metadata:
            print("No query metadata available for size-based reranking")
            return results

        # Extract size features from query metadata
        query_size = self._extract_size_features(query_metadata)
        if not query_size:
            print("Could not extract size features from query metadata")
            return results

        print("\n--- Size-Based Reranking Debug Information ---")
        print(f"Query part size features: {{\n  \"length\": {query_size['length']},\n  \"width\": {query_size['width']},\n  \"height\": {query_size['height']},\n  \"volume\": {query_size['volume']},\n  \"surface_area\": {query_size['surface_area']},\n  \"max_dimension\": {query_size['max_dimension']},\n  \"min_dimension\": {query_size['min_dimension']},\n  \"mid_dimension\": {query_size['mid_dimension']}\n}}")

        # Load BOM data for each result part
        bom_dir = self.config.get("metadata", {}).get("bom_dir")
        if not bom_dir:
            print("BOM directory not specified in config")
            return results

        # Load all necessary BOM files ahead of time to avoid redundant loading
        bom_data_cache = {}
        for part_info in results["part_info"]:
            if part_info and "parent_step" in part_info:
                step = part_info["parent_step"]
                if step not in bom_data_cache:
                    bom_path = os.path.join(bom_dir, f"{step}_bom.json")
                    if os.path.exists(bom_path):
                        bom_data_cache[step] = self.metadata_encoder.load_bom_data(bom_path)
                    else:
                        bom_data_cache[step] = {}

        # Prepare to store size similarity scores
        similarities = []
        part_sizes = []

        # Find metadata for all parts
        for part_info in results["part_info"]:
            if part_info and "parent_step" in part_info:
                step = part_info["parent_step"]
                part_name = part_info.get("part_name", "unknown")

                # Get BOM data from cache
                bom_data = bom_data_cache.get(step, {})
                part_metadata = self.metadata_encoder.find_part_metadata(bom_data, part_name)

                # Extract size features if metadata is available
                if part_metadata and "properties" in part_metadata:
                    part_size = self._extract_size_features(part_metadata)
                    part_sizes.append(part_size)
                else:
                    part_sizes.append(None)
            else:
                part_sizes.append(None)

        # Calculate size similarity scores
        for idx, part_size in enumerate(part_sizes):
            part_name = results["part_info"][idx]["part_name"] if idx < len(results["part_info"]) and results["part_info"][idx] else "unknown"
            if part_size:
                # Calculate size similarity using the enhanced method
                size_sim = self._calculate_size_similarity(query_size, part_size, debug=True, part_name=part_name)
                similarities.append(size_sim)
            else:
                similarities.append(None)

        # Print original vs. adjusted similarities header
        print("\nOriginal vs. Adjusted Similarities:")
        print("---------------------------------------")
        print("Part                 | Original   | Size Sim   | Adjusted")
        print("---------------------------------------")

        # Combine size similarity with visual similarity for reranking
        reranked_scores = []
        for idx, (similarity, size_sim) in enumerate(zip(results["similarities"], similarities)):
            part_name = results["part_info"][idx]["part_name"] if idx < len(results["part_info"]) and results["part_info"][idx] else "unknown"

            # Default to original score if no size similarity available
            if size_sim is None:
                reranked_scores.append(similarity)
                continue

            # Get the original visual similarity
            original_score = similarity

            # Apply different strategies based on size similarity
            if size_sim >= 95:
                # For nearly identical size (95-100%), give a strong boost with minimal penalty
                # The higher the size_sim, the less penalty applied
                boost_factor = 1.0 + 0.5 * (size_sim - 95) / 5  # Boost factor from 1.0 to 1.5
                # New formula gives less penalty for perfect size matches
                adjusted_score = original_score + (100 - original_score) * (1 - size_weight / boost_factor)
            elif size_sim >= 50:
                # For good size matches (50-95%), apply standard weighted formula
                adjusted_score = original_score * (1 - size_weight) + size_sim * size_weight
            elif size_sim <= 5:
                # For very poor size matches (0-5%), apply a more severe penalty
                # The worse the size match, the more severe the penalty
                penalty_factor = 1.0 + 4.0 * (5 - size_sim) / 5  # Penalty factor from 1.0 to 5.0
                adjusted_score = original_score * (1 - size_weight * penalty_factor)
                # Ensure score doesn't drop below 10% of original
                adjusted_score = max(adjusted_score, original_score * 0.1)
            else:
                # For moderate size matches (5-50%), use the standard formula
                adjusted_score = original_score * (1 - size_weight) + size_sim * size_weight

            # Display debugging information
            print(f"{part_name:20} | {original_score:.2f}      | {size_sim:.2f}     | {adjusted_score:.2f}")

            reranked_scores.append(adjusted_score)

        # Update similarities in results
        results["similarities"] = reranked_scores

        # Sort results based on the reranked similarities
        indices = np.argsort(reranked_scores)[::-1]  # Sort in descending order

        # Reorder everything based on the new ranking
        results["similarities"] = [reranked_scores[i] for i in indices]
        results["distances"] = [results["distances"][i] for i in indices] if "distances" in results else []
        results["paths"] = [results["paths"][i] for i in indices] if "paths" in results else []
        results["part_info"] = [results["part_info"][i] for i in indices] if "part_info" in results else []

        print("\n--- End of Size-Based Reranking Debug ---\n")
        return results

    def _extract_size_features(self, metadata):
        """
        Extract size-related features from part metadata

        Args:
            metadata (dict): Part metadata

        Returns:
            size_features (dict): Dictionary of size features or None if not available
        """
        try:
            properties = metadata.get("properties", metadata)
            if "type" in metadata and isinstance(metadata.get("properties"), dict):
                properties = metadata.get("properties")

            if not properties or not isinstance(properties, dict):
                return None

            # Extract the dimensional features
            size_features = {
                "length": float(properties.get("length", properties.get("Length", 0.0))),
                "width": float(properties.get("width", properties.get("Width", 0.0))),
                "height": float(properties.get("height", properties.get("Height", 0.0))),
                "volume": float(properties.get("volume", properties.get("Volume", 0.0))),
                "surface_area": float(properties.get("surface_area", properties.get("SurfaceArea", 0.0))),
                "max_dimension": float(properties.get("max_dimension", properties.get("MaxDimension", 0.0))),
                "min_dimension": float(properties.get("min_dimension", properties.get("MinDimension", 0.0))),
                "mid_dimension": float(properties.get("mid_dimension", properties.get("MidDimension", 0.0)))
            }

            # Check if we have enough valid data
            valid_values = [v for v in size_features.values() if v > 0]
            if len(valid_values) < 3:  # Need at least 3 valid dimensions
                return None

            return size_features
        except Exception as e:
            print(f"Error extracting size features: {e}")
            return None

    def _calculate_size_similarity(self, size1, size2, debug=False, part_name=""):
        """
        Calculate size similarity between two parts (0-100 scale)

        Args:
            size1 (dict): Size features of first part
            size2 (dict): Size features of second part
            debug (bool): Whether to print debug information
            part_name (str): Name of the part for debugging

        Returns:
            similarity (float): Size similarity score (0-100)
        """
        if not size1 or not size2:
            return 0

        try:
            # Calculate volume ratio (larger / smaller)
            vol1 = max(0.001, size1.get("volume", 0))
            vol2 = max(0.001, size2.get("volume", 0))

            volume_ratio = max(vol1, vol2) / min(vol1, vol2)

            # Calculate difference in each dimension
            dim_diffs = []
            dim_ratios = {}
            for dim in ["length", "width", "height", "max_dimension", "min_dimension", "mid_dimension"]:
                val1 = size1.get(dim, 0)
                val2 = size2.get(dim, 0)
                if val1 > 0 and val2 > 0:
                    # Calculate ratio (larger / smaller)
                    ratio = max(val1, val2) / max(0.001, min(val1, val2))
                    dim_diffs.append(ratio)
                    dim_ratios[dim] = ratio

            # Calculate bounding box diagonal ratio
            diagonal1 = math.sqrt(size1.get("length", 0)**2 + size1.get("width", 0)**2 + size1.get("height", 0)**2)
            diagonal2 = math.sqrt(size2.get("length", 0)**2 + size2.get("width", 0)**2 + size2.get("height", 0)**2)

            if diagonal1 > 0 and diagonal2 > 0:
                diagonal_ratio = max(diagonal1, diagonal2) / min(diagonal1, diagonal2)
            else:
                diagonal_ratio = 10  # Default penalty if diagonals can't be calculated

            # Calculate overall similarity score
            # Combine volume ratio and average dimension ratio
            if dim_diffs:
                avg_dim_ratio = sum(dim_diffs) / len(dim_diffs)

                # Use a weighted combination, with more emphasis on volume
                # which is a better overall indicator of size similarity
                combined_ratio = 0.5 * volume_ratio + 0.3 * avg_dim_ratio + 0.2 * diagonal_ratio
            else:
                combined_ratio = volume_ratio

            # Convert to similarity score (1.0 = perfect match, higher ratios = lower similarity)
            # Use a gentler exponential decay function to be more tolerant of small differences
            # Fine-tune the decay factor from -0.5 to -0.3 to be more forgiving
            adjusted_ratio = combined_ratio - 1  # Normalize so 1 (perfect match) becomes 0
            similarity = 100 * math.exp(-0.3 * adjusted_ratio)

            # For nearly identical parts, ensure very high similarity
            # If the ratio is extremely close to 1 (e.g., <1.05), boost the score
            if combined_ratio < 1.05:
                similarity = max(similarity, 95)  # Ensure at least 95% similarity

            if debug:
                print(f"\nSize comparison for {part_name}:")
                print(f"  Volume: {vol1:.1f} vs {vol2:.1f} (ratio: {volume_ratio:.3f})")
                print(f"  Dimension ratios: {json.dumps(dim_ratios, indent=2)}")
                print(f"  Avg dimension ratio: {avg_dim_ratio:.3f}, Diagonal ratio: {diagonal_ratio:.3f}")
                print(f"  Combined ratio: {combined_ratio:.3f}")
                print(f"  Final size similarity: {similarity:.2f}%")

            return similarity
        except Exception as e:
            print(f"Error calculating size similarity: {e}")
            return 0

    def visualize_results(self, query_image_path, results, output_path=None):
        """
        Visualize retrieval results

        Args:
            query_image_path (str): Path to the query image
            results (dict): Search results from retrieve_similar
            output_path (str): Optional path to save the visualization

        Returns:
            output_path (str): Path to the saved visualization
        """
        try:
            import matplotlib.pyplot as plt
            from PIL import Image

            # Create default output path if none provided
            if output_path is None:
                # Create base results directory
                results_base_dir = os.path.join(self.config["data"]["output_dir"], "results")
                os.makedirs(results_base_dir, exist_ok=True)

                # Check if this is a part name search
                if "query_part_name" in results:
                    # For part name searches, create and use a dedicated directory
                    search_dir = os.path.join(results_base_dir, "name_searches")
                    os.makedirs(search_dir, exist_ok=True)

                    # For part name searches, use the specified format
                    part_name = results["query_part_name"]
                    output_path = os.path.join(
                        search_dir,
                        f"query_search_results_{part_name}.png"
                    )
                else:
                    # For image searches, create and use a dedicated directory
                    image_dir = os.path.join(results_base_dir, "image_queries")
                    os.makedirs(image_dir, exist_ok=True)

                    # For image searches, use the query image name (but avoid duplicate extensions)
                    base_name = os.path.basename(query_image_path)
                    # Remove extension to avoid .png.png issue
                    base_name_no_ext = os.path.splitext(base_name)[0]
                    output_path = os.path.join(
                        image_dir,
                        f"query_results_{base_name_no_ext}.png"
                    )

            # Get the paths to result images
            result_paths = results["paths"]

            # Get or calculate similarity scores
            if "similarities" in results and results["similarities"]:
                similarities = results["similarities"]
            else:
                # If no recalibrated similarities, calculate from distances
                similarities = [100 * (1 / (1 + d)) for d in results["distances"]]

            # Number of results to display
            n_results = len(result_paths)

            # Create a figure with n+1 subplots (query + results)
            fig_width = max(20, 2 * (n_results + 1))  # Cap the width to a reasonable size
            fig = plt.figure(figsize=(fig_width, 4))

            # Create grid spec for better layout control
            gs = fig.add_gridspec(1, n_results + 1)

            # Plot query image
            ax_query = fig.add_subplot(gs[0, 0])
            query_img = Image.open(query_image_path).convert('RGB')
            ax_query.imshow(query_img)
            ax_query.set_title("Query")
            ax_query.axis('off')

            # Plot result images with similarity score
            for i in range(n_results):
                ax = fig.add_subplot(gs[0, i+1])

                # Check if path exists
                if result_paths[i] and os.path.exists(result_paths[i]):
                    # Load and display image
                    img = Image.open(result_paths[i]).convert('RGB')
                    ax.imshow(img)

                    # Set background color of title based on similarity
                    similarity = similarities[i]

                    # Create a color gradient from red (low similarity) to green (high similarity)
                    if similarity >= 90:
                        color = 'darkgreen'  # Very high similarity
                    elif similarity >= 70:
                        color = 'green'      # High similarity
                    elif similarity >= 50:
                        color = 'darkorange'     # Medium similarity
                    elif similarity >= 20:
                        color = 'orange'     # Low similarity
                    else:
                        color = 'red'        # Very low similarity

                    # Format title with colored background
                    title = f"Similarity: {similarity:.1f}%"

                    # Add step and part information if available
                    if "part_info" in results and i < len(results["part_info"]) and results["part_info"][i]:
                        info = results["part_info"][i]
                        step = info.get("parent_step", "?")
                        part = info.get("part_name", "?")
                        title += f"\nSTEP: {step}"
                        title += f"\nPart: {part}"

                    ax.set_title(title, color=color)
                else:
                    # Handle missing images
                    ax.text(0.5, 0.5, "Image not found", ha='center', va='center')
                    ax.set_title("Missing")

                ax.axis('off')

            plt.tight_layout()

            # Always save to the output path (never show in GUI)
            plt.savefig(output_path)
            plt.close()
            print(f"Visualization saved to {output_path}")

            return output_path

        except Exception as e:
            print(f"Error visualizing results: {e}")
            return None

    def evaluate(self, query_dir=None, ground_truth=None):
        """
        Evaluate the retrieval system

        Args:
            query_dir (str): Directory containing query images
            ground_truth (dict or str): Ground truth mapping or path to JSON file

        Returns:
            results (dict): Evaluation results
        """
        if query_dir is None:
            query_dir = os.path.join(self.config["data"]["output_dir"], "evaluation", "queries")

        # Collect query images
        query_images = []
        for file in os.listdir(query_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                query_images.append(os.path.join(query_dir, file))

        # Load ground truth if it's a file
        if isinstance(ground_truth, str) and os.path.exists(ground_truth):
            with open(ground_truth, 'r') as f:
                ground_truth = json.load(f)

        # Evaluate
        results = self.evaluator.evaluate_queries(
            query_images,
            ground_truth,
            top_k=self.config["evaluation"]["top_k"]
        )

        # Visualize some results
        self.evaluator.visualize_retrieval(results["queries"])

        # Save results
        self.evaluator.save_results(results)

        # Plot metrics
        if ground_truth:
            self.evaluator.plot_metrics(results)

        return results

    def find_part_by_name(self, part_name, threshold=None):
        """
        Find a part by name and return its image path

        Args:
            part_name (str): Name of the part to search for
            threshold (float): Minimum similarity score (0-1) for matching

        Returns:
            best_match (dict): Dictionary with image path and similarity score of the best match
        """
        # Use configured threshold if none provided
        if threshold is None:
            threshold = self.config.get("text_search", {}).get("default_threshold", 0.7)
        """
        Find a part by name and return its image path

        Args:
            part_name (str): Name of the part to search for
            threshold (float): Minimum similarity score (0-1) for matching

        Returns:
            best_match (dict): Dictionary with image path and similarity score of the best match
        """
        if not part_name or not self.vector_db.metadata or "part_info" not in self.vector_db.metadata:
            print(f"No part metadata available to search for '{part_name}'")
            return None

        print(f"Searching for part matching name: '{part_name}'")
        best_match = None
        best_score = 0

        # Normalize the query part name for better matching
        query_norm = self._normalize_part_name(part_name)

        # Go through all parts in the index
        for idx, part_info in self.vector_db.metadata["part_info"].items():
            if not part_info or "part_name" not in part_info:
                continue

            # Get the part name and normalize it
            current_part_name = part_info["part_name"]
            current_norm = self._normalize_part_name(current_part_name)

            # Calculate similarity score
            similarity = self._calculate_name_similarity(query_norm, current_norm)

            if similarity > best_score:
                best_score = similarity
                image_path = self.vector_db.metadata["id_to_path"].get(idx)
                best_match = {
                    "path": image_path,
                    "similarity": similarity,
                    "part_name": current_part_name,
                    "part_info": part_info
                }

                # If we find an exact match, we can stop searching
                if similarity >= 0.99:
                    break

        # Only return matches above the threshold
        if best_match and best_score >= threshold:
            print(f"Best match: '{best_match['part_name']}' with {best_match['similarity']:.2f} similarity")
            return best_match
        else:
            print(f"No matches found above threshold {threshold}")
            return None

    def _normalize_part_name(self, part_name):
        """
        Normalize part name for better matching

        Args:
            part_name (str): Original part name

        Returns:
            normalized (str): Normalized part name
        """
        if not part_name:
            return ""

        # Convert to lowercase
        normalized = part_name.lower()

        # Remove common prefixes/suffixes that might vary
        prefixes = ["part_", "component_", "assembly_", "asm_"]
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]

        # Remove numbers and special characters (but keep spaces)
        import re
        # Keep letters and spaces, replace everything else with spaces
        cleaned = re.sub(r'[^a-z ]', ' ', normalized)
        # Replace multiple spaces with a single space
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        return cleaned

    def _calculate_name_similarity(self, name1, name2):
        """
        Calculate similarity between two part names

        Args:
            name1 (str): First normalized part name
            name2 (str): Second normalized part name

        Returns:
            similarity (float): Similarity score between 0 and 1
        """
        if not name1 or not name2:
            return 0.0

        # If one is a subset of the other, it's a good match
        if name1 in name2 or name2 in name1:
            # Calculate the ratio of the shorter to the longer string
            min_len = min(len(name1), len(name2))
            max_len = max(len(name1), len(name2))
            if max_len == 0:  # Avoid division by zero
                return 0.0
            return min_len / max_len

        # Otherwise, calculate character-level similarity
        # Using Jaccard similarity on character sets
        set1 = set(name1)
        set2 = set(name2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        if union == 0:  # Avoid division by zero
            return 0.0

        # Basic Jaccard similarity
        jaccard = intersection / union

        # If the sets are very similar but the strings are different lengths,
        # adjust the score to reflect the difference in length
        len_ratio = min(len(name1), len(name2)) / max(len(name1), len(name2)) if max(len(name1), len(name2)) > 0 else 0

        # Words in common (for multi-word part names)
        words1 = set(name1.split())
        words2 = set(name2.split())
        word_intersection = len(words1.intersection(words2))
        word_union = len(words1.union(words2))
        word_jaccard = word_intersection / word_union if word_union > 0 else 0

        # Get weights from config if available
        weights = self.config.get("text_search", {}).get("similarity_weights", {})
        jaccard_weight = weights.get("jaccard", 0.4)
        len_ratio_weight = weights.get("length_ratio", 0.3)
        word_jaccard_weight = weights.get("word_jaccard", 0.3)

        # Combine metrics with weights
        similarity = (jaccard_weight * jaccard) + \
                     (len_ratio_weight * len_ratio) + \
                     (word_jaccard_weight * word_jaccard)

        return similarity

    def retrieve_by_part_name(self, part_name, k=10, rotation_invariant=True, num_rotations=8, threshold=None):
        """
        Find a part by name and then retrieve similar parts using visual search

        Args:
            part_name (str): Name of the part to search for
            k (int): Number of results to retrieve
            rotation_invariant (bool): Whether to perform rotation-invariant search
            num_rotations (int): Number of rotations to try if rotation_invariant is True
            threshold (float): Minimum similarity score (0-1) for matching part names

        Returns:
            results (dict): Search results
        """
        # First find the part by name
        part_match = self.find_part_by_name(part_name, threshold=threshold)

        if not part_match or "path" not in part_match or not part_match["path"]:
            print(f"Could not find a part matching '{part_name}'")
            return {"paths": [], "distances": [], "similarities": []}

        # Use the found part's image as a query
        query_image_path = part_match["path"]
        print(f"Found part image at {query_image_path}, using it as query for visual search")

        # Perform visual search with the found image
        results = self.retrieve_similar(
            query_image_path,
            k=k,
            rotation_invariant=rotation_invariant,
            num_rotations=num_rotations
        )

        # Add the original query info to the results
        results["query_part_name"] = part_name
        results["query_match"] = part_match

        return results

    def retrieve_by_assembly(self, assembly_id, k=10, selected_parts=None):
        """
        Retrieve similar assemblies based on the parts in the query assembly

        Args:
            assembly_id (str): ID of the assembly to query
            k (int): Number of results to return
            selected_parts (list): Optional list of part filenames to include in the search
                                  (if None, all parts are included)

        Returns:
            results (dict): Search results with assembly similarity scores
        """
        print(f"Retrieving similar assemblies to assembly ID: {assembly_id}")

        # Create output directory for full assembly queries if it doesn't exist
        assembly_results_dir = os.path.join(self.config["data"]["output_dir"], "results", "full_assembly_queries")
        os.makedirs(assembly_results_dir, exist_ok=True)

        # Find all parts that belong to the assembly
        image_dir = os.path.join(self.config["data"]["output_dir"], "images")
        assembly_parts = []

        # Pattern for parts belonging to this assembly: "{assembly_id}_*.png"
        for filename in os.listdir(image_dir):
            if filename.startswith(f"{assembly_id}_") and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                assembly_parts.append(os.path.join(image_dir, filename))

        if not assembly_parts:
            print(f"No parts found for assembly ID {assembly_id}")
            return {"paths": [], "distances": [], "similarities": []}

        # Filter parts based on user selection if provided
        if selected_parts:
            original_count = len(assembly_parts)
            # Save list of original part filenames before filtering
            original_parts = [os.path.basename(p) for p in assembly_parts]

            # Filter to keep only the parts that were selected by the user
            assembly_parts = [part_path for part_path in assembly_parts
                             if os.path.basename(part_path) in selected_parts]

            if not assembly_parts:
                print(f"Error: None of the selected parts match any parts found for assembly ID {assembly_id}")
                print(f"Available parts: {original_parts}")
                return {"paths": [], "distances": [], "similarities": []}

            print(f"Using {len(assembly_parts)} selected parts out of {original_count} total parts for assembly ID {assembly_id}")
        else:
            print(f"Found {len(assembly_parts)} parts for assembly ID {assembly_id}")

        # Initialize scores dictionary (without manually adding the query assembly)
        assembly_scores = {}

        # For each part in the query assembly, find similar parts
        for part_path in assembly_parts:
            part_filename = os.path.basename(part_path)
            print(f"Processing part: {part_filename}")

            # Retrieve similar parts for this part
            part_results = self.retrieve_similar(
                part_path,
                k=k*3,  # Get more results to ensure good coverage of different assemblies
                rotation_invariant=True,
                num_rotations=8
            )

            # Group results by assembly ID to find best match for each assembly
            assembly_matches = {}

            # Process results for this part
            if part_results and "paths" in part_results and len(part_results["paths"]) > 0:
                for i, (result_path, info, similarity) in enumerate(zip(
                        part_results["paths"],
                        part_results.get("part_info", [None] * len(part_results["paths"])),
                        part_results.get("similarities", [None] * len(part_results["paths"]))
                    )):
                    # Get the assembly ID from the result part
                    result_assembly_id = None
                    if info and "parent_step" in info:
                        result_assembly_id = info["parent_step"]
                    else:
                        # Extract assembly ID from the filename
                        result_filename = os.path.basename(result_path)
                        parts = result_filename.split('_', 1)
                        if len(parts) > 0:
                            result_assembly_id = parts[0]

                    if result_assembly_id:
                        # For each assembly, store only the best match for this query part
                        result_part_name = os.path.basename(result_path)
                        if result_assembly_id not in assembly_matches:
                            assembly_matches[result_assembly_id] = {
                                "similarity": similarity,
                                "result_part": result_part_name,
                                "result_path": result_path
                            }
                        elif similarity > assembly_matches[result_assembly_id]["similarity"]:
                            # Keep only the best match for this assembly
                            assembly_matches[result_assembly_id] = {
                                "similarity": similarity,
                                "result_part": result_part_name,
                                "result_path": result_path
                            }

            # Now add the best match for each assembly to our overall scores
            for result_assembly_id, match_info in assembly_matches.items():
                if result_assembly_id not in assembly_scores:
                    assembly_scores[result_assembly_id] = {
                        "score": 0,
                        "part_matches": [],
                        "matched_parts": set(),  # Keep track of matched parts to avoid duplicates
                        "count": 0,
                        "total_similarity": 0,
                        "is_query": (result_assembly_id == assembly_id)  # Mark if this is the query assembly
                    }

                # Only count this match if the target part hasn't been matched yet
                result_part = match_info["result_part"]
                if result_part not in assembly_scores[result_assembly_id]["matched_parts"]:
                    assembly_scores[result_assembly_id]["matched_parts"].add(result_part)
                    assembly_scores[result_assembly_id]["total_similarity"] += match_info["similarity"]
                    assembly_scores[result_assembly_id]["count"] += 1

                    # Remember this match with its score
                    assembly_scores[result_assembly_id]["part_matches"].append({
                        "query_part": part_filename,
                        "result_part": result_part,
                        "similarity": match_info["similarity"]
                    })

        # Calculate final scores for each assembly
        for result_assembly_id, data in assembly_scores.items():
            # Calculate scores for all assemblies (including query if found naturally)
            if data["count"] > 0:
                # Calculate raw similarity score (average similarity of matched parts)
                raw_score = data["total_similarity"] / data["count"]

                # Calculate coverage ratio (how many parts of the query assembly were matched)
                coverage_ratio = data["count"] / len(assembly_parts)

                # NEW SCORING FORMULA:
                # Multiply raw similarity by coverage ratio squared to heavily penalize low coverage
                # This ensures that assemblies with few matching parts can't get high overall scores
                # An assembly with 5% coverage can at most get 5% of its raw score
                final_score = raw_score * (coverage_ratio * coverage_ratio)

                # Store both raw and coverage-adjusted scores
                data["raw_score"] = raw_score
                data["coverage_ratio"] = coverage_ratio
                data["score"] = final_score

        # Sort assemblies by score
        sorted_assemblies = sorted(
            assembly_scores.items(),
            key=lambda x: x[1]["score"],
            reverse=True
        )

        # Take top k assemblies
        top_assemblies = sorted_assemblies[:k]

        # Prepare results
        results = {
            "paths": [],
            "similarities": [],
            "part_info": [],
            "assembly_details": {}
        }

        # Look for full assembly images
        full_assembly_dir = os.path.join(self.config["data"]["output_dir"], "full_assembly_images")

        # Create the directory if it doesn't exist yet
        if not os.path.exists(full_assembly_dir):
            os.makedirs(full_assembly_dir, exist_ok=True)
            print(f"Created full assembly images directory: {full_assembly_dir}")

        # Format results
        for result_assembly_id, data in top_assemblies:
            # Try to find the full assembly image first
            full_assembly_path = os.path.join(full_assembly_dir, f"{result_assembly_id}_full_assembly.png")

            if os.path.exists(full_assembly_path):
                # Use the full assembly image for visualization
                results["paths"].append(full_assembly_path)
            else:
                # Fall back to the highest similarity part if full assembly image not found
                if data["part_matches"]:
                    # Find the highest similarity match to use for visualization
                    best_match = max(data["part_matches"], key=lambda x: x["similarity"])
                    match_filepath = os.path.join(image_dir, best_match["result_part"])
                    results["paths"].append(match_filepath)
                    print(f"Warning: Full assembly image not found for assembly {result_assembly_id}, using part image instead")

            # Add similarity score
            results["similarities"].append(data["score"])

            # Add assembly info with a special note for the query assembly
            is_query = data.get("is_query", False)
            part_name = "QUERY ASSEMBLY" if is_query else f"Assembly with {data['count']} matching parts"

            results["part_info"].append({
                "parent_step": result_assembly_id,
                "part_name": part_name,
                "is_query": is_query
            })

            # Add detailed information about this assembly match
            results["assembly_details"][result_assembly_id] = {
                "score": data["score"],
                "raw_score": data.get("raw_score", data["score"]),  # Include raw score if available
                "matching_parts_count": data["count"],
                "coverage_ratio": data["count"] / len(assembly_parts),
                "part_matches": data["part_matches"],
                "is_query": is_query
            }

        # Create a visualization of the full assembly query results
        if results["paths"]:
            # Look for the full assembly image for the query assembly
            query_full_assembly = os.path.join(full_assembly_dir, f"{assembly_id}_full_assembly.png")

            if os.path.exists(query_full_assembly):
                # Use the full assembly image as the query image
                query_image_path = query_full_assembly
            else:
                # Fall back to the first part if full assembly image not found
                query_image_path = assembly_parts[0]
                print(f"Warning: Full assembly image not found for query assembly {assembly_id}, using part image instead")

            # Create output path for visualization
            output_path = os.path.join(assembly_results_dir, f"assembly_{assembly_id}_results.png")

            # Add visualization path to results
            results["visualization_path"] = output_path

            # Call visualize_results with a custom title
            self._visualize_assembly_results(
                query_image_path,
                results,
                assembly_id,
                len(assembly_parts),
                output_path
            )

        return results

    def _visualize_assembly_results(self, query_image_path, results, assembly_id, parts_count, output_path):
        """
        Visualize assembly retrieval results with custom titles

        Args:
            query_image_path (str): Path to a representative query image
            results (dict): Search results
            assembly_id (str): Query assembly ID
            parts_count (int): Number of parts in the query assembly
            output_path (str): Path to save the visualization
        """
        try:
            import matplotlib.pyplot as plt
            from PIL import Image
            import numpy as np

            # Get the paths to result images and their assembly details
            result_paths = results["paths"]
            assembly_details = results.get("assembly_details", {})

            # Get or calculate similarity scores
            similarities = results.get("similarities", [])

            # Number of results to display
            n_results = len(result_paths)

            # Create a figure with n+1 subplots (query + results)
            fig_width = max(20, 2 * (n_results + 1))  # Cap the width to a reasonable size
            fig = plt.figure(figsize=(fig_width, 6))

            # Add a super title
            fig.suptitle(f"Assembly Query Results for Assembly {assembly_id} ({parts_count} parts)",
                         fontsize=16, fontweight='bold')

            # Create grid spec for better layout control
            gs = fig.add_gridspec(2, n_results + 1)

            # Plot query image in a larger size (spans 2 rows)
            ax_query = fig.add_subplot(gs[:, 0])
            query_img = Image.open(query_image_path).convert('RGB')
            ax_query.imshow(query_img)
            ax_query.set_title(f"Query Assembly #{assembly_id}\n{parts_count} parts", fontsize=12)
            ax_query.axis('off')

            # Plot result images with similarity score
            for i in range(n_results):
                # Get assembly details
                result_assembly_id = results["part_info"][i].get("parent_step", "unknown")
                is_query_assembly = results["part_info"][i].get("is_query", False)
                detail = assembly_details.get(result_assembly_id, {})

                # Plot the result image in the top row
                ax_img = fig.add_subplot(gs[0, i+1])
                if result_paths[i] and os.path.exists(result_paths[i]):
                    img = Image.open(result_paths[i]).convert('RGB')
                    ax_img.imshow(img)

                    # Add border to highlight query assembly when it appears in results
                    if is_query_assembly:
                        for spine in ax_img.spines.values():
                            spine.set_edgecolor('gold')
                            spine.set_linewidth(5)
                else:
                    ax_img.text(0.5, 0.5, "Image not found", ha='center', va='center')

                # Calculate a color based on similarity score
                similarity = similarities[i] if i < len(similarities) else 0
                if is_query_assembly:
                    # Use gold color for query assembly
                    color = 'darkgoldenrod'
                    title_prefix = "QUERY "
                elif similarity >= 90:
                    color = 'darkgreen'
                    title_prefix = ""
                elif similarity >= 70:
                    color = 'green'
                    title_prefix = ""
                elif similarity >= 50:
                    color = 'darkorange'
                    title_prefix = ""
                elif similarity >= 20:
                    color = 'orange'
                    title_prefix = ""
                else:
                    color = 'red'
                    title_prefix = ""

                # Set the title for the image, with special formatting for the query assembly
                ax_img.set_title(f"{title_prefix}Assembly #{result_assembly_id}",
                                fontsize=10,
                                color=color,
                                fontweight='bold' if is_query_assembly else 'normal')
                ax_img.axis('off')

                # Plot the details in the bottom row
                ax_detail = fig.add_subplot(gs[1, i+1])
                ax_detail.axis('off')

                # Format the details text
                matching_parts = detail.get("matching_parts_count", 0)
                coverage = detail.get("coverage_ratio", 0) * 100
                score = similarity

                if is_query_assembly:
                    detail_text = "EXACT MATCH\n"
                    detail_text += f"Score: {score:.1f}%\n"
                    detail_text += f"Parts: {matching_parts}"
                else:
                    detail_text = f"Score: {score:.1f}%\n"
                    detail_text += f"Matching parts: {matching_parts}\n"
                    detail_text += f"Coverage: {coverage:.1f}%"

                # Display the details with colored text based on score
                ax_detail.text(0.5, 0.5, detail_text, ha='center', va='center',
                              color=color, fontweight='bold', fontsize=9)

                # Add additional highlight for query assembly
                if is_query_assembly:
                    # Add background rectangle to make the match more obvious
                    rect = plt.Rectangle((-0.05, -0.05), 1.1, 1.1, fill=True,
                                       color='lightyellow', alpha=0.3, transform=ax_detail.transAxes)
                    ax_detail.add_patch(rect)

            plt.tight_layout()
            plt.subplots_adjust(top=0.85)  # Make room for the super title

            # Save the visualization
            plt.savefig(output_path, dpi=150)
            plt.close()

            print(f"Assembly visualization saved to {output_path}")
            return output_path

        except Exception as e:
            print(f"Error visualizing assembly results: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_system_info(self):
        """
        Get information about the retrieval system

        Returns:
            info (dict): System information
        """
        # Get index stats
        index_stats = self.vector_db.get_stats()

        # Get model info
        model_info = {
            "name": self.config["model"]["name"],
            "embedding_dim": self.config["model"]["embedding_dim"],
            "image_size": self.config["model"]["image_size"],
            "device": str(next(self.image_encoder.model.parameters()).device)
        }

        # Add metadata info if enabled
        metadata_info = {
            "enabled": self.use_metadata
        }

        if self.use_metadata:
            autoencoder_trained = self.metadata_encoder.trained if hasattr(self.metadata_encoder, 'trained') else False
            metadata_info.update({
                "embedding_dim": self.config.get("metadata", {}).get("embedding_dim"),
                "fusion_method": self.config.get("metadata", {}).get("fusion_method"),
                "bom_dir": self.config.get("metadata", {}).get("bom_dir"),
                "autoencoder_trained": autoencoder_trained,
                "autoencoder_hidden_dims": self.config.get("metadata", {}).get("hidden_dims", [512, 384]),
                "autoencoder_model_path": self.config.get("metadata", {}).get("model_path", "models/metadata_autoencoder.pt")
            })

        return {
            "model": model_info,
            "index": index_stats,
            "configuration": self.config,
            "metadata": metadata_info
        }
