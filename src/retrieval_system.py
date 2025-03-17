import os
import yaml
import json
import torch
from PIL import Image
import matplotlib.pyplot as plt
from .image_encoder import ImageEncoder
from .vector_database import VectorDatabase
from .data_processor import DataProcessor
from .evaluator import RetrievalEvaluator
from .rotational_utils import rotation_invariant_search

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
        
        # Initialize components
        self.image_encoder = ImageEncoder(
            model_name=self.config["model"]["name"],
            pretrained=self.config["model"]["pretrained"],
            embedding_dim=self.config["model"]["embedding_dim"],
            image_size=self.config["model"]["image_size"]
        )
        
        self.vector_db = VectorDatabase(
            embedding_dim=self.config["model"]["embedding_dim"],
            index_file=self.config["indexing"]["index_file"],
            metadata_file=self.config["indexing"]["metadata_file"]
        )
        
        self.data_processor = DataProcessor(
            output_dir=self.config["data"]["output_dir"]
        )
        
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
        
        # Build the index with metadata extraction
        count = self.vector_db.build_from_directory(
            self.image_encoder,
            image_dir,
            batch_size=self.config["training"]["batch_size"],
            metadata_func=self.extract_part_info  # Pass the metadata extraction function
        )
        
        # Save the index
        self.vector_db.save()
        
        return count
    
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
        if not os.path.exists(query_image_path):
            return {"error": f"Query image not found: {query_image_path}"}
        
        if rotation_invariant:
            # Use rotation-invariant search
            results = rotation_invariant_search(
                self.image_encoder, 
                self.vector_db, 
                query_image_path, 
                k=k,
                num_rotations=num_rotations
            )
        else:
            # Use standard search
            query_embedding = self.image_encoder.encode_image(query_image_path)
            
            if query_embedding is None:
                return {"error": f"Failed to encode query image: {query_image_path}"}
            
            # Search the database
            results = self.vector_db.search(query_embedding, k=k)
        
        return results
    
    def visualize_results(self, query_image_path, results, output_path=None):
        """
        Visualize retrieval results
        
        Args:
            query_image_path (str): Path to the query image
            results (dict): Search results from retrieve_similar
            output_path (str): Path to save the visualization
            
        Returns:
            output_path (str): Path to the saved visualization
        """
        if output_path is None:
            os.makedirs(os.path.join(self.config["data"]["output_dir"], "results"), exist_ok=True)
            output_path = os.path.join(
                self.config["data"]["output_dir"],
                "results",
                f"query_results_{os.path.basename(query_image_path)}.png"
            )
        
        # Create figure with more vertical space for text below images
        k = len(results["paths"])
        fig, axes = plt.subplots(1, k + 1, figsize=(3 * (k + 1), 3.5))
        plt.subplots_adjust(bottom=0.3)  # Add extra space at the bottom
        
        # Display query
        query_img = Image.open(query_image_path).convert('RGB')
        axes[0].imshow(query_img)
        axes[0].set_title("Query")
        axes[0].axis('off')
        
        # Display results
        for i, (path, distance, info) in enumerate(zip(results["paths"], results["distances"], results.get("part_info", [None] * len(results["paths"])))):
            if path and os.path.exists(path):
                result_img = Image.open(path).convert('RGB')
                # Convert distance to similarity score (0-100%), where higher is better
                similarity = 100 * (1 / (1 + distance))
                axes[i+1].imshow(result_img)
                
                # Create a title that only includes similarity
                axes[i+1].set_title(f"Similarity: {similarity:.1f}%")
                
                # Add part info at the bottom of the figure, below the image
                if info:
                    parent_step = info.get("parent_step", "unknown")
                    part_name = info.get("part_name", "unknown")
                    
                    # Add a text box at the bottom
                    axes[i+1].text(0.5, -0.15, f"STEP: {parent_step}", 
                                 horizontalalignment='center', verticalalignment='top', 
                                 transform=axes[i+1].transAxes, fontsize=8, color='black')
                    axes[i+1].text(0.5, -0.25, f"Part: {part_name}", 
                                 horizontalalignment='center', verticalalignment='top', 
                                 transform=axes[i+1].transAxes, fontsize=8, color='black')
                
                axes[i+1].axis('off')
            else:
                axes[i+1].text(0.5, 0.5, "Image not found", horizontalalignment='center')
                axes[i+1].axis('off')
        
        # Save visualization
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        print(f"Visualization saved to {output_path}")
        return output_path
    
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
        
        return {
            "model": model_info,
            "index": index_stats,
            "configuration": self.config
        }
