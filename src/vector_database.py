import os
import pickle
import numpy as np
import faiss
import torch
from tqdm import tqdm

class VectorDatabase:
    """
    Handles indexing and searching of image embeddings using FAISS
    """
    def __init__(self, embedding_dim=768, index_file=None, metadata_file=None):
        """
        Initialize the vector database
        
        Args:
            embedding_dim (int): Dimension of the embedding vectors
            index_file (str): Path to save/load the FAISS index
            metadata_file (str): Path to save/load the metadata
        """
        self.embedding_dim = embedding_dim
        self.index_file = index_file
        self.metadata_file = metadata_file
        
        # Initialize an empty index
        self.index = faiss.IndexFlatL2(embedding_dim)
        
        # Initialize an empty metadata dictionary to store mapping of indices to file paths
        self.metadata = {"id_to_path": {}, "path_to_id": {}}
        
        # Load existing index and metadata if provided
        if index_file and os.path.exists(index_file) and metadata_file and os.path.exists(metadata_file):
            self.load()
    
    def add_embeddings(self, embeddings, file_paths, metadata=None):
        """
        Add embeddings to the index with their corresponding file paths and metadata
        
        Args:
            embeddings (torch.Tensor or numpy.ndarray): Embedding vectors
            file_paths (list): List of file paths corresponding to embeddings
            metadata (list): Optional list of metadata dictionaries for each embedding
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.numpy()
        
        # Get the current size of the index
        start_id = self.index.ntotal
        
        # Add embeddings to the index
        self.index.add(embeddings.astype(np.float32))
        
        # Update metadata
        for i, path in enumerate(file_paths):
            idx = start_id + i
            self.metadata["id_to_path"][idx] = path
            self.metadata["path_to_id"][path] = idx
            
            # Store additional metadata if provided
            if metadata and i < len(metadata):
                if "part_info" not in self.metadata:
                    self.metadata["part_info"] = {}
                self.metadata["part_info"][idx] = metadata[i]
    
    def search(self, query_embedding, k=10):
        """
        Search for the k nearest neighbors of the query embedding
        
        Args:
            query_embedding (torch.Tensor or numpy.ndarray): Query embedding vector
            k (int): Number of nearest neighbors to return
            
        Returns:
            results (dict): Dictionary containing distances and paths of nearest neighbors
        """
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.numpy()
        
        # Ensure the query is 2D
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Perform the search
        distances, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        # Get the file paths and part info
        paths = []
        part_info = []
        
        for idx in indices[0]:
            idx = int(idx)
            paths.append(self.metadata["id_to_path"].get(idx, None))
            
            # Get part info if available
            if "part_info" in self.metadata and idx in self.metadata["part_info"]:
                part_info.append(self.metadata["part_info"][idx])
            else:
                # If no specific part info, extract basic info from the path
                path = self.metadata["id_to_path"].get(idx, "")
                filename = os.path.basename(path)
                # Try to extract step name and part name from filename
                parts = os.path.splitext(filename)[0].split('_')
                if len(parts) > 1:
                    step_name = parts[0]
                    part_name = '_'.join(parts[1:])  # In case part name has underscores
                else:
                    step_name = "unknown"
                    part_name = filename
                
                part_info.append({
                    "parent_step": step_name,
                    "part_name": part_name
                })
        
        return {
            "distances": distances[0].tolist(),
            "indices": indices[0].tolist(),
            "paths": paths,
            "part_info": part_info
        }
    
    def build_from_directory(self, image_encoder, directory, extensions=('.png', '.jpg', '.jpeg'), batch_size=32, metadata_func=None):
        """
        Build the index from all compatible images in a directory
        
        Args:
            image_encoder: Encoder to use for generating embeddings
            directory (str): Directory containing images
            extensions (tuple): File extensions to include
            batch_size (int): Batch size for processing
            metadata_func (callable): Optional function to extract metadata from image path
            
        Returns:
            count (int): Number of images indexed
        """
        # Collect all image paths
        image_paths = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(extensions):
                    full_path = os.path.join(root, file)
                    image_paths.append(full_path)
        
        if not image_paths:
            print(f"No images found in {directory} with extensions {extensions}")
            return 0
        
        print(f"Found {len(image_paths)} images, generating embeddings...")
        
        # Process images in batches
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i:i+batch_size]
            batch_embeddings = image_encoder.encode_batch(batch_paths, batch_size)
            
            # Generate metadata if a function is provided
            batch_metadata = None
            if metadata_func:
                batch_metadata = [metadata_func(path) for path in batch_paths]
            
            self.add_embeddings(batch_embeddings, batch_paths, batch_metadata)
        
        print(f"Added {len(image_paths)} images to the index")
        return len(image_paths)
    
    def save(self):
        """
        Save the index and metadata to disk
        """
        if self.index_file:
            faiss.write_index(self.index, self.index_file)
            print(f"Index saved to {self.index_file}")
        
        if self.metadata_file:
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(self.metadata, f)
            print(f"Metadata saved to {self.metadata_file}")
    
    def load(self):
        """
        Load the index and metadata from disk
        """
        if self.index_file and os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
            print(f"Index loaded from {self.index_file}")
        
        if self.metadata_file and os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
            print(f"Metadata loaded from {self.metadata_file}")
    
    def get_stats(self):
        """
        Get statistics about the index
        
        Returns:
            stats (dict): Dictionary containing index statistics
        """
        return {
            "num_vectors": self.index.ntotal,
            "dimension": self.embedding_dim,
            "index_type": type(self.index).__name__,
        }
