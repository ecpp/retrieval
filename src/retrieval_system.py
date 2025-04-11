import os
import yaml
import json
import torch
import shutil
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
import faiss

# Import metadata components
try:
    from .metadata_encoder import MetadataEncoder
    from .fusion_module import FusionModule
    METADATA_AVAILABLE = True
    print("Successfully imported metadata components")
except ImportError as e:
    METADATA_AVAILABLE = False
    print(f"ImportError: Could not import metadata components: {e}")

# Import assembly components
try:
    from .assembly_encoder import AssemblyEncoder
    from .assembly_fusion_module import AssemblyFusionModule
    ASSEMBLY_AVAILABLE = True
    print("Successfully imported assembly components")
except ImportError as e:
    ASSEMBLY_AVAILABLE = False
    print(f"ImportError: Could not import assembly components: {e}")

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

        # Initialize assembly components
        self.use_assembly = self.config.get("assembly", {}).get("enabled", False)
        self.assembly_encoder = None
        self.assembly_db = None
        self.assembly_fusion = None

        if ASSEMBLY_AVAILABLE and self.use_assembly:
            self.init_assembly_components()
        else:
            if not ASSEMBLY_AVAILABLE:
                print("Assembly components not available")
            self.use_assembly = False

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

    def init_assembly_components(self):
        """
        Initialize assembly search components
        """
        try:
            # Create assembly directory
            graph_dir = self.config.get("assembly", {}).get("graph_dir", "data/output/hierarchical_graphs")
            os.makedirs(graph_dir, exist_ok=True)

            # Get fusion settings
            fusion_enabled = self.config.get("assembly", {}).get("fusion", {}).get("enabled", True)
            fusion_method = self.config.get("assembly", {}).get("fusion", {}).get("method", "weighted")
            graph_weight = self.config.get("assembly", {}).get("fusion", {}).get("graph_weight", 0.6)
            part_weight = self.config.get("assembly", {}).get("fusion", {}).get("part_weight", 0.4)
            part_aggregation = self.config.get("assembly", {}).get("fusion", {}).get("part_aggregation", "mean")

            # Initialize assembly encoder with part features flag if fusion is enabled
            self.assembly_encoder = AssemblyEncoder(
                feature_dim=self.config.get("assembly", {}).get("feature_dim", 512),
                hidden_dim=self.config.get("assembly", {}).get("hidden_dim", 256),
                embedding_dim=self.config.get("assembly", {}).get("embedding_dim", 768),
                device=next(self.image_encoder.model.parameters()).device.type if hasattr(self.image_encoder, 'model') else None,
                use_part_features=fusion_enabled
            )
            
            # Initialize assembly vector database (similar to part vector database)
            self.assembly_db = VectorDatabase(
                embedding_dim=self.config.get("assembly", {}).get("embedding_dim", 768),
                index_file=self.config.get("assembly", {}).get("index_file", "models/assembly_index.bin"),
                metadata_file=self.config.get("assembly", {}).get("metadata_file", "models/assembly_metadata.pkl")
            )

            # Try to load pre-trained model if it exists
            model_path = self.config.get("assembly", {}).get("model_path", "models/assembly_gnn.pt")
            if os.path.exists(model_path):
                self.assembly_encoder.load_trained_model(model_path)

            # Initialize assembly fusion module if enabled
            if fusion_enabled:
                self.assembly_fusion = AssemblyFusionModule(
                    graph_dim=self.config.get("assembly", {}).get("embedding_dim", 768),
                    part_feature_dim=self.config["model"]["embedding_dim"],
                    output_dim=self.config.get("assembly", {}).get("embedding_dim", 768),
                    fusion_method=fusion_method,
                    graph_weight=graph_weight,
                    part_weight=part_weight
                )
                print(f"Assembly similarity enabled with GNN and fusion (method: {fusion_method})")
            else:
                print(f"Assembly similarity enabled with GNN only")
        except Exception as e:
            print(f"Error initializing assembly components: {e}")
            import traceback
            traceback.print_exc()
            self.use_assembly = False

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
            print("To train the autoencoder, use the 'train-autoencoder' command")

        # Copy hierarchical graph files if assembly is enabled
        if self.use_assembly:
            graph_dir = self.config.get("assembly", {}).get("graph_dir", "data/output/hierarchical_graphs")
            os.makedirs(graph_dir, exist_ok=True)
            
            # Copy graph files from the source to the graph directory
            graph_files_copied = 0
            
            # Look for both JSON and GraphML formats
            for root, dirs, files in os.walk(dataset_dir):
                for file in files:
                    # Check for hierarchical graph files in different formats
                    if file.endswith("_hierarchical.graphml"):
                        
                        src_path = os.path.join(root, file)
                        dst_path = os.path.join(graph_dir, file)
                        
                        # Copy file
                        try:
                            # For GraphML files
                            if file.endswith(".graphml"):
                                # Direct file copy for binary formats
                                shutil.copy2(src_path, dst_path)
                            else:
                                # For JSON, we can pretty-print
                                with open(src_path, 'r') as src, open(dst_path, 'w') as dst:
                                    graph_data = json.load(src)
                                    json.dump(graph_data, dst, indent=2)
                            
                            graph_files_copied += 1
                            print(f"Copied graph file: {file}")
                        except Exception as e:
                            print(f"Error copying graph file {file}: {e}")
            
            print(f"Copied {graph_files_copied} hierarchical graph files to {graph_dir}")
            print("To build the assembly index, use the 'build-assembly' command")

        # Return the processed parts
        return all_parts

    def ingest_assembly_data(self, dataset_dir):
        """
        Ingest hierarchical graph data from a directory of processed STEP files
        
        Args:
            dataset_dir (str): Directory containing STEP output directories
            
        Returns:
            all_assemblies (list): List of processed assembly information
        """
        if not self.use_assembly:
            print("Assembly similarity is not enabled. Use --use-assembly flag or enable in config.")
            return []
        
        graph_dir = self.config.get("assembly", {}).get("graph_dir", "data/output/hierarchical_graphs")
        os.makedirs(graph_dir, exist_ok=True)
        
        print(f"Ingesting assembly graph data from {dataset_dir}")
        print(f"Target directory: {graph_dir}")
        
        # Find and copy hierarchical graph files
        all_assemblies = []
        graph_files_copied = 0
        
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                # Check for hierarchical graph files in different formats
                if file.endswith("_hierarchical.graphml"):
                    
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(graph_dir, file)
                    
                    # Extract assembly info
                    step_id = self.assembly_encoder.extract_step_id(src_path)
                    
                    # Copy file
                    try:
                        # For GraphML files
                        if file.endswith(".graphml"):
                            # Direct file copy for binary formats
                            shutil.copy2(src_path, dst_path)
                            
                            # Add to assemblies list without detailed info for GraphML
                            all_assemblies.append({
                                "step_id": step_id,
                                "graph_path": dst_path,
                                "format": "graphml"
                            })
                        else:
                            # For JSON, we can process and pretty-print
                            with open(src_path, 'r') as src:
                                graph_data = json.load(src)
                                
                            with open(dst_path, 'w') as dst:
                                json.dump(graph_data, dst, indent=2)
                            
                            # Add to assemblies list with detailed info for JSON
                            all_assemblies.append({
                                "step_id": step_id,
                                "graph_path": dst_path,
                                "num_nodes": len(graph_data.get("nodes", [])),
                                "num_edges": len(graph_data.get("edges", [])),
                                "format": "json"
                            })
                            
                            print(f"Ingested assembly graph: {file} with {len(graph_data.get('nodes', []))} nodes")
                        
                        graph_files_copied += 1
                    except Exception as e:
                        print(f"Error copying graph file {file}: {e}")
        
        print(f"Copied {graph_files_copied} hierarchical graph files to {graph_dir}")
        
        # Save assembly information
        assembly_info_path = os.path.join(self.config["data"]["output_dir"], "assembly_info.json")
        with open(assembly_info_path, 'w') as f:
            json.dump(all_assemblies, f, indent=2)
        
        print(f"Saved assembly info to {assembly_info_path}")
        
        return all_assemblies

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

    def build_assembly_index(self, graph_dir=None):
        """
        Build the vector index from hierarchical assembly graphs
        
        Args:
            graph_dir (str): Directory containing hierarchical graph files
            
        Returns:
            count (int): Number of assemblies indexed
        """
        if not self.use_assembly:
            print("Assembly similarity is not enabled. Use --use-assembly flag or enable in config.")
            return 0
        
        if graph_dir is None:
            graph_dir = self.config.get("assembly", {}).get("graph_dir", "data/output/hierarchical_graphs")
        
        # Check if we need to rebuild the index
        rebuild_index = True
        if os.path.exists(self.config["assembly"]["index_file"]) and os.path.exists(self.config["assembly"]["metadata_file"]):
            # Check if we should force rebuild
            rebuild_index = self.config.get("indexing", {}).get("force_rebuild", False)
            
            if not rebuild_index:
                print(f"Using existing index with graph_weight={self.assembly_fusion.graph_weight:.2f}, part_weight={self.assembly_fusion.part_weight:.2f}")
                return self.assembly_db.get_count()
            else:
                print(f"Force rebuilding index due to force_rebuild=true in config")
        
        # Get fusion settings from config
        graph_weight = self.config.get("assembly", {}).get("fusion", {}).get("graph_weight", 0.6)
        part_weight = self.config.get("assembly", {}).get("fusion", {}).get("part_weight", 0.4)
        
        # Store current weights for reference
        current_graph_weight = self.assembly_fusion.graph_weight if self.assembly_fusion else graph_weight
        current_part_weight = self.assembly_fusion.part_weight if self.assembly_fusion else part_weight
        
        # Save fusion settings to a JSON file for reference
        with open(os.path.join(os.path.dirname(self.config["assembly"]["index_file"]), "assembly_fusion_settings.json"), "w") as f:
            json.dump({
                "graph_weight": current_graph_weight,
                "part_weight": current_part_weight
            }, f)
        
        print("Building assembly vector index")
        print(f"Building assembly index from {graph_dir}")
        
        # Initialize a new empty index
        self.assembly_db.index = faiss.IndexFlatL2(self.assembly_db.embedding_dim)
        self.assembly_db.metadata = {"id_to_path": {}, "path_to_id": {}, "step_ids": {}}
        
        # Get all hierarchical graph files
        graph_files = []
        for root, _, files in os.walk(graph_dir):
            for file in files:
                if file.endswith("_hierarchical.graphml"):
                    graph_files.append(os.path.join(root, file))
        
        print(f"Found {len(graph_files)} hierarchical graphs, generating embeddings...")
        
        # Process graphs in batches
        batch_size = self.config["training"]["batch_size"]
        
        # Build metadata information for assemblies
        assembly_metadata = {}
        
        for i in tqdm(range(0, len(graph_files), batch_size)):
            batch_paths = graph_files[i:i+batch_size]
            batch_embeddings = []
            batch_metadata = []
            
            for path in batch_paths:
                # Load and process the graph
                graph_data = self.assembly_encoder.load_graph(path)
                
                if graph_data:
                    # Encode the graph
                    graph_embedding = self.assembly_encoder.encode_graph(graph_data)
                    
                    if graph_embedding is not None:
                        # Extract assembly information
                        step_id = self.assembly_encoder.extract_step_id(path)
                        
                        # Add to batch
                        batch_embeddings.append(graph_embedding)
                        batch_metadata.append({
                            "step_id": step_id,
                            "graph_path": path,
                            "part_info": {
                                "parent_step": step_id,
                                "assembly_name": os.path.basename(path)
                            }
                        })
            
            # Add embeddings to the assembly index
            if batch_embeddings:
                # Convert list of tensors to single tensor
                embeddings_tensor = torch.cat(batch_embeddings, dim=0)
                
                # Ensure paths are properly stored in id_to_path mapping
                self.assembly_db.add_embeddings(embeddings_tensor, batch_paths, batch_metadata)
                
                # Make sure the id_to_path mapping is properly populated
                start_id = self.assembly_db.index.ntotal - len(batch_paths)
                for j, (path, metadata) in enumerate(zip(batch_paths, batch_metadata)):
                    idx = start_id + j
                    idx_str = str(idx)
                    if 'id_to_path' not in self.assembly_db.metadata:
                        self.assembly_db.metadata['id_to_path'] = {}
                    self.assembly_db.metadata['id_to_path'][idx_str] = path
                    
                    # Store STEP ID in metadata for easy access
                    if 'step_ids' not in self.assembly_db.metadata:
                        self.assembly_db.metadata['step_ids'] = {}
                    self.assembly_db.metadata['step_ids'][idx_str] = metadata['step_id']
        
        # Save the assembly index
        self.assembly_db.save()
        
        print(f"Added {len(graph_files)} assemblies to the index")
        return len(graph_files)

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

    def retrieve_similar_assemblies(self, query, k=10):
        """
        Retrieve similar assemblies to a query STEP file
        
        Args:
            query (str): STEP ID, graph file path, or part name
            k (int): Number of results to retrieve
            
        Returns:
            results (dict): Search results
        """
        if not self.use_assembly:
            print("Assembly similarity not enabled")
            return None
            
        print(f"\n--- Beginning Assembly Retrieval Process ---")
        print(f"Query: {query}")
        
        # Determine query type and get embedding
        query_step_id = None
        query_graph_path = None
        
        # Case 1: Query is a STEP ID
        if query.isdigit() or (query.startswith("STEP") and query[4:].isdigit()):
            print(f"Assuming query is a STEP ID: {query}")
            query_step_id = query.replace("STEP", "")  # Remove "STEP" prefix if present
            
            # Find graph file for this STEP ID
            graph_dir = self.config.get("assembly", {}).get("graph_dir", "data/output/hierarchical_graphs")
            graph_path = os.path.join(graph_dir, f"{query_step_id}_hierarchical.graphml")
            
            if not os.path.exists(graph_path):
                print(f"Could not find graph file for STEP ID: {query_step_id}")
                return {"distances": [], "indices": [], "paths": [], "similarities": []}
                
            query_graph_path = graph_path
            
        # Case 2: Query is a graph file path
        elif os.path.exists(query) and query.endswith("_hierarchical.graphml"):
            print(f"Using provided graph file: {query}")
            query_graph_path = query
            query_step_id = self.assembly_encoder.extract_step_id(query_graph_path)
            
        # Case 3: Query is a part name (try to find matching STEP file)
        else:
            print(f"Assuming query is a part name, searching for matching STEP file...")
            step_match = self.find_step_by_name(query)
            
            if step_match and step_match.get("step_id"):
                query_step_id = step_match["step_id"]
                graph_dir = self.config.get("assembly", {}).get("graph_dir", "data/output/hierarchical_graphs")
                graph_path = os.path.join(graph_dir, f"{query_step_id}_hierarchical.graphml")
                
                if os.path.exists(graph_path):
                    query_graph_path = graph_path
                    print(f"Found matching STEP ID: {query_step_id}")
                else:
                    print(f"Found matching STEP ID {query_step_id} but no graph file exists")
                    return {"distances": [], "indices": [], "paths": [], "similarities": []}
            else:
                print(f"Could not find matching STEP file for part name: {query}")
                return {"distances": [], "indices": [], "paths": [], "similarities": []}
        
        # Get embedding from graph
        print(f"\n--- Processing Graph: {os.path.basename(query_graph_path)} ---")
        query_embedding = self._get_embedding_from_graph(query_graph_path)
        
        if query_embedding is None:
            print(f"Failed to get embedding for graph: {query_graph_path}")
            return {"distances": [], "indices": [], "paths": [], "similarities": []}
        
        # Search the assembly database
        print(f"Searching assembly database for {k} nearest neighbors...")
        search_results = self.assembly_db.search(query_embedding, k=k)
        
        # Handle different return formats from search method
        if isinstance(search_results, dict):
            # If search returns a dictionary with distances and indices
            distances = search_results.get("distances", [])
            indices = search_results.get("indices", [])
        elif isinstance(search_results, tuple) and len(search_results) == 2:
            # If search returns a tuple of (distances, indices)
            distances, indices = search_results
        else:
            print(f"Unexpected search result format: {type(search_results)}")
            return {"distances": [], "indices": [], "paths": [], "similarities": []}
        
        # Ensure distances and indices are in the expected format (2D arrays)
        import numpy as np
        
        # Convert to numpy arrays if they are lists
        if isinstance(distances, list):
            distances = np.array(distances)
        if isinstance(indices, list):
            indices = np.array(indices)
            
        # Reshape if needed
        if hasattr(distances, 'shape') and len(distances.shape) == 1:
            distances = distances.reshape(1, -1)
        if hasattr(indices, 'shape') and len(indices.shape) == 1:
            indices = indices.reshape(1, -1)
            
        # Ensure we have 2D arrays even if they're lists
        if not hasattr(distances, 'shape'):
            distances = [distances]
        if not hasattr(indices, 'shape'):
            indices = [indices]
            
        # Process results
        paths = []
        part_info = []
        similarities = []
        
        for i, idx in enumerate(indices[0]):
            # Get path from metadata - the metadata is a direct mapping of indices to file paths
            idx_str = str(int(idx))  # Convert to string for dictionary lookup
            if 'id_to_path' in self.assembly_db.metadata and idx_str in self.assembly_db.metadata['id_to_path']:
                path = self.assembly_db.metadata['id_to_path'][idx_str]
            else:
                path = "unknown"
            paths.append(path)
            
            # Extract STEP ID from metadata or path
            if 'step_ids' in self.assembly_db.metadata and idx_str in self.assembly_db.metadata['step_ids']:
                step_id = self.assembly_db.metadata['step_ids'][idx_str]
            elif path and path != "unknown":
                step_id = os.path.basename(path).split('_')[0]
            else:
                step_id = "Unknown"
            
            # Get part info from metadata
            info = {}
            if "part_info" in self.assembly_db.metadata and idx_str in self.assembly_db.metadata["part_info"]:
                info = self.assembly_db.metadata["part_info"][idx_str]
            part_info.append(info)
            
            # Calculate similarity score (inverse of distance)
            similarity = 100 * (1 / (1 + distances[0][i]))
            similarities.append(similarity)
        
        # Create results dictionary
        results = {
            "distances": distances[0].tolist(),
            "indices": indices[0].tolist(),
            "paths": paths,
            "part_info": part_info,
            "similarities": similarities,
            "query_step_id": query_step_id,
            "query_graph_path": query_graph_path
        }
        
        # Print raw results before deduplication
        print("\nRaw results before deduplication:")
        for i, (idx, dist, path, sim) in enumerate(zip(results["indices"], results["distances"], results["paths"], results["similarities"])):
            # Get path from metadata
            idx_str = str(int(idx))  # Convert to string for dictionary lookup
            if 'id_to_path' in self.assembly_db.metadata and idx_str in self.assembly_db.metadata['id_to_path']:
                path = self.assembly_db.metadata['id_to_path'][idx_str]
            else:
                path = "unknown"
            
            # Extract STEP ID from metadata or path
            if 'step_ids' in self.assembly_db.metadata and idx_str in self.assembly_db.metadata['step_ids']:
                step_id = self.assembly_db.metadata['step_ids'][idx_str]
            elif path and path != "unknown":
                step_id = os.path.basename(path).split('_')[0]
            else:
                step_id = "Unknown"
                
            print(f"Raw Result {i+1}: Index={idx}, STEP ID={step_id}, Path={os.path.basename(path) if path != 'unknown' else 'unknown'}, Distance={dist:.4f}, Similarity={sim:.1f}%")
        
        # Remove duplicate STEP IDs from results
        unique_results = self._deduplicate_assembly_results(results)
        
        # Print info about deduplication
        print(f"\nAfter deduplication: {len(results['indices'])} results -> {len(unique_results['indices'])} unique results")
        
        print("\nRaw distances:")
        for i, dist in enumerate(unique_results["distances"]):
            print(f"Result {i+1}: Distance = {dist:.4f}")
            
        print("\nRaw distances and indices between query and results:")
        for i, (idx, dist) in enumerate(zip(unique_results["indices"], unique_results["distances"])):
            # Get path from metadata
            idx_str = str(int(idx))  # Convert to string for dictionary lookup
            if 'id_to_path' in self.assembly_db.metadata and idx_str in self.assembly_db.metadata['id_to_path']:
                path = self.assembly_db.metadata['id_to_path'][idx_str]
            else:
                path = "unknown"
            
            # Extract STEP ID from metadata or path
            if 'step_ids' in self.assembly_db.metadata and idx_str in self.assembly_db.metadata['step_ids']:
                step_id = self.assembly_db.metadata['step_ids'][idx_str]
            elif path and path != "unknown":
                step_id = os.path.basename(path).split('_')[0]
            else:
                step_id = "Unknown"
                
            print(f"Result {i+1}: Index={idx}, STEP ID={step_id}, L2 Distance={dist:.4f}")
        
        print(f"--- Assembly Retrieval Process Complete ---\n")
        return unique_results

    def _get_embedding_from_graph(self, graph_path):
        """
        Get embedding from graph file
        
        Args:
            graph_path (str): Path to graph file
            
        Returns:
            embedding (torch.Tensor): Graph embedding
        """
        try:
            print(f"\n--- Processing Graph: {os.path.basename(graph_path)} ---")
            
            # Load and process the graph
            graph_data = self.assembly_encoder.load_graph(graph_path)
            
            if not graph_data:
                print(f"Could not load graph data from {graph_path}")
                return None
            
            print(f"Successfully loaded graph with {graph_data['num_nodes']} nodes")
            
            # Encode the graph
            graph_embedding = self.assembly_encoder.encode_graph(graph_data)
            
            if graph_embedding is None:
                print(f"Failed to encode graph {graph_path}")
                return None
            
            print(f"Successfully encoded graph: shape={graph_embedding.shape}")
            
            # Get fusion settings from config
            fusion_enabled = self.config.get("assembly", {}).get("fusion", {}).get("enabled", True)
            graph_weight = self.config.get("assembly", {}).get("fusion", {}).get("graph_weight", 0.6)
            part_weight = self.config.get("assembly", {}).get("fusion", {}).get("part_weight", 0.4)
            
            print(f"Fusion settings - Enabled: {fusion_enabled}, Graph Weight: {graph_weight}, Part Weight: {part_weight}")
            
            # If fusion is enabled, combine with part-based features
            if fusion_enabled and self.assembly_fusion is not None:
                # Get the STEP ID
                step_id = self.assembly_encoder.extract_step_id(graph_path)
                
                # Get part-based features
                print(f"Retrieving part features for assembly {step_id}...")
                part_features = self._get_part_features_for_assembly(step_id, graph_data, graph_path)
                
                # If we have valid part features, fuse them with graph features
                if part_features is not None:
                    print(f"Fusing graph features with part features using weights: graph={graph_weight}, part={part_weight}")
                    
                    # Update fusion module weights from config
                    self.assembly_fusion.graph_weight = graph_weight
                    self.assembly_fusion.part_weight = part_weight
                    
                    # Fuse embeddings
                    final_embedding = self.assembly_fusion.fuse(graph_embedding, part_features)
                    
                    # Normalize the embedding to unit length for cosine similarity-based comparison
                    norm = torch.norm(final_embedding, p=2, dim=1, keepdim=True)
                    normalized_embedding = final_embedding / (norm + 1e-10)  # Add small epsilon to prevent division by zero
                    
                    print(f"Successfully fused embeddings: shape={normalized_embedding.shape}")
                    print(f"Embedding norm before normalization: {norm.item():.4f}")
                    return normalized_embedding
                else:
                    print(f"No valid part features found. Using graph embedding only.")
                    # Normalize the graph embedding
                    norm = torch.norm(graph_embedding, p=2, dim=1, keepdim=True)
                    normalized_embedding = graph_embedding / (norm + 1e-10)
                    print(f"Embedding norm before normalization: {norm.item():.4f}")
                    return normalized_embedding
            else:
                print(f"Fusion disabled or not available. Using graph embedding only.")
                # Normalize the graph embedding
                norm = torch.norm(graph_embedding, p=2, dim=1, keepdim=True)
                normalized_embedding = graph_embedding / (norm + 1e-10)
                print(f"Embedding norm before normalization: {norm.item():.4f}")
                return normalized_embedding
        except Exception as e:
            print(f"Error getting embedding from graph {graph_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def _get_part_features_for_assembly(self, step_id, graph_data=None, graph_path=None):
        """
        Get part-based features for an assembly
        
        Args:
            step_id (str): STEP ID of the assembly
            graph_data (dict, optional): Graph data if already loaded
            graph_path (str, optional): Path to the graph file if graph_data not provided
            
        Returns:
            part_features (torch.Tensor): Aggregated part features
        """
        try:
            # Get output directory for finding part images
            output_dir = self.config["data"]["output_dir"]
            
            # Get BOM directory for metadata
            bom_dir = self.config.get("metadata", {}).get("bom_dir")
            
            # Get parts information from the assembly using images and BoM files
            parts_info = self.assembly_encoder.get_assembly_parts(step_id, graph_data, graph_path, output_dir, bom_dir)
            
            if not parts_info:
                print(f"No parts found for assembly {step_id} from images or BoM files")
                return None
            
            # Get features for each part
            part_embeddings = []
            part_metadata_embeddings = []
            
            for part_id, info in parts_info.items():
                image_path = info.get("image_path")
                part_name = info.get("name")
                
                if image_path and os.path.exists(image_path):
                    # Extract visual features
                    visual_embedding = self.image_encoder.encode_image(image_path)
                    
                    # Try to get metadata features if available
                    metadata_embedding = None
                    if self.use_metadata and self.metadata_encoder and part_name and bom_dir:
                        # Construct BOM path
                        bom_path = os.path.join(bom_dir, f"{step_id}_bom.json")
                        
                        if os.path.exists(bom_path):
                            # Load BOM data
                            bom_data = self.metadata_encoder.load_bom_data(bom_path)
                            
                            # Find metadata for this part
                            part_metadata = self.metadata_encoder.find_part_metadata(bom_data, part_name)
                            
                            if part_metadata:
                                # Encode metadata
                                metadata_embedding = self.metadata_encoder.encode_metadata(part_metadata)
                    
                    # If we have both visual and metadata features, fuse them at part level
                    if self.use_metadata and self.fusion_module and metadata_embedding is not None:
                        # Fuse part-level features
                        part_embedding = self.fusion_module.fuse(visual_embedding, metadata_embedding)
                    else:
                        # Just use visual features
                        part_embedding = visual_embedding
                    
                    # Add to the list
                    part_embeddings.append(part_embedding)
            
            if not part_embeddings:
                print(f"No valid part embeddings found for assembly {step_id} - no usable images")
                return None
            
            # Combine part embeddings into a single tensor
            part_embeddings_tensor = torch.cat(part_embeddings, dim=0)
            
            # Aggregate part features using the assembly fusion module
            aggregation_method = self.config.get("assembly", {}).get("fusion", {}).get("part_aggregation", "mean")
            aggregated_features = self.assembly_fusion.aggregate_part_features(part_embeddings_tensor, aggregation_method)
            
            print(f"Successfully aggregated features from {len(part_embeddings)} parts for assembly {step_id}")
            return aggregated_features    
        except Exception as e:
            print(f"Error getting part features for assembly {step_id}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _deduplicate_assembly_results(self, results):
        """
        Remove duplicate STEP IDs from assembly search results
        
        Args:
            results (dict): Original search results
            
        Returns:
            unique_results (dict): Deduplicated search results
        """
        # Create a dictionary to track unique entries
        unique_entries = {}
        unique_indices = []
        unique_distances = []
        unique_paths = []
        unique_part_info = []
        unique_similarities = []
        
        # Iterate through results
        for i, idx in enumerate(results["indices"]):
            step_id = self.assembly_db.metadata.get(idx, {}).get("step_id", "Unknown")
            
            # For unknown STEP IDs, use the index as a unique identifier
            if step_id == "Unknown":
                unique_key = f"Unknown_{idx}"
            else:
                unique_key = step_id
            
            # Only add if this unique key hasn't been seen before
            if unique_key not in unique_entries:
                unique_entries[unique_key] = True
                unique_indices.append(idx)
                unique_distances.append(results["distances"][i])
                
                if i < len(results["paths"]):
                    unique_paths.append(results["paths"][i])
                    
                if i < len(results["part_info"]):
                    unique_part_info.append(results["part_info"][i])
                    
                if i < len(results["similarities"]):
                    unique_similarities.append(results["similarities"][i])
        
        # Create new results dictionary with unique entries
        unique_results = {
            "distances": unique_distances,
            "indices": unique_indices,
            "paths": unique_paths,
            "part_info": unique_part_info,
            "similarities": unique_similarities
        }
        
        return unique_results

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
            import traceback
            traceback.print_exc()
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

    def find_step_by_name(self, step_name, threshold=None):
        """
        Find a STEP file by name
        
        Args:
            step_name (str): Name of the STEP file to search for
            threshold (float): Minimum similarity score (0-1) for matching
            
        Returns:
            best_match (dict): Dictionary with STEP information of the best match
        """
        if not self.use_assembly or not self.assembly_db or not hasattr(self.assembly_db, 'metadata'):
            print(f"No assembly database available to search for '{step_name}'")
            return None
        
        # Use configured threshold if none provided
        if threshold is None:
            threshold = self.config.get("text_search", {}).get("default_threshold", 0.7)
        
        print(f"Searching for STEP file matching name: '{step_name}'")
        best_match = None
        best_score = 0
        
        # Go through all assemblies in the index
        for idx, metadata in enumerate(self.assembly_db.metadata.get("item_metadata", [])):
            if not metadata or "step_id" not in metadata:
                continue
            
            # Get the STEP name
            current_step_id = metadata["step_id"]
            
            # Calculate similarity score using the same method as for parts
            similarity = self._calculate_name_similarity(step_name, current_step_id)
            
            if similarity > best_score:
                best_score = similarity
                graph_path = self.assembly_db.metadata["id_to_path"].get(str(idx))
                best_match = {
                    "step_id": current_step_id,
                    "graph_path": graph_path,
                    "similarity": similarity
                }
                
                # If we find an exact match, we can stop searching
                if similarity >= 0.99:
                    break
        
        # Only return matches above the threshold
        if best_match and best_score >= threshold:
            print(f"Best match: '{best_match['step_id']}' with {best_match['similarity']:.2f} similarity")
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
            
        # Add assembly info if enabled
        assembly_info = {
            "enabled": hasattr(self, 'use_assembly') and self.use_assembly
        }
        
        if hasattr(self, 'use_assembly') and self.use_assembly:
            # Get assembly GNN info
            gnn_trained = self.assembly_encoder.trained if hasattr(self.assembly_encoder, 'trained') else False
            
            # Add assembly index info if available
            assembly_index_stats = {}
            if hasattr(self, 'assembly_db') and self.assembly_db:
                assembly_index_stats = self.assembly_db.get_stats()
            
            # Get fusion info
            fusion_enabled = self.config.get("assembly", {}).get("fusion", {}).get("enabled", False)
            fusion_method = self.config.get("assembly", {}).get("fusion", {}).get("method", "weighted")
            part_aggregation = self.config.get("assembly", {}).get("fusion", {}).get("part_aggregation", "mean")
            
            assembly_info.update({
                "feature_dim": self.config.get("assembly", {}).get("feature_dim"),
                "hidden_dim": self.config.get("assembly", {}).get("hidden_dim"),
                "embedding_dim": self.config.get("assembly", {}).get("embedding_dim"),
                "graph_dir": self.config.get("assembly", {}).get("graph_dir"),
                "gnn_trained": gnn_trained,
                "gnn_model_path": self.config.get("assembly", {}).get("model_path"),
                "index_stats": assembly_index_stats,
                "fusion": {
                    "enabled": fusion_enabled,
                    "method": fusion_method,
                    "part_aggregation": part_aggregation,
                    "graph_weight": self.config.get("assembly", {}).get("fusion", {}).get("graph_weight", 0.6),
                    "part_weight": self.config.get("assembly", {}).get("fusion", {}).get("part_weight", 0.4)
                }
            })

        return {
            "model": model_info,
            "index": index_stats,
            "configuration": self.config,
            "metadata": metadata_info,
            "assembly": assembly_info
        }

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