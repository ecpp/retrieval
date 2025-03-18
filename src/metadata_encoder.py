import torch
import torch.nn as nn
import numpy as np
import json
import os

class MetadataEncoder:
    """
    Encodes CAD part metadata from BOM data into embeddings
    """
    def __init__(self, output_dim=256):
        """
        Initialize the metadata encoder
        
        Args:
            output_dim (int): Dimension of the output embedding
        """
        self.output_dim = output_dim
        
        # Define a simple MLP for encoding metadata
        self.encoder = nn.Sequential(
            nn.Linear(self.get_input_dim(), 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        
        # Move to appropriate device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder.to(self.device)
        
    def get_input_dim(self):
        """Get the input dimension based on the number of features"""
        # Count all the numeric features we're extracting
        return 46
    
    def extract_features(self, metadata):
        """
        Extract relevant features from metadata
        
        Args:
            metadata (dict): Part metadata from BOM
            
        Returns:
            features (list): List of numeric features
        """
        features = []
        
        # Extract dimensional data
        features.extend([
            metadata.get("properties", {}).get("length", 0.0),
            metadata.get("properties", {}).get("width", 0.0),
            metadata.get("properties", {}).get("height", 0.0),
            metadata.get("properties", {}).get("max_dimension", 0.0),
            metadata.get("properties", {}).get("min_dimension", 0.0),
            metadata.get("properties", {}).get("mid_dimension", 0.0),
        ])
        
        # Extract derived dimensional ratios
        features.extend([
            metadata.get("properties", {}).get("thickness_ratio", 0.0),
            metadata.get("properties", {}).get("width_ratio", 0.0),
            metadata.get("properties", {}).get("elongation", 0.0),
            metadata.get("properties", {}).get("volume_ratio", 0.0),
            metadata.get("properties", {}).get("complexity", 0.0),
        ])
        
        # Extract volume and surface area
        features.extend([
            metadata.get("properties", {}).get("volume", 0.0),
            metadata.get("properties", {}).get("surface_area", 0.0),
        ])
        
        # Extract topological metrics
        features.extend([
            metadata.get("properties", {}).get("face_count", 0),
            metadata.get("properties", {}).get("edge_count", 0),
            metadata.get("properties", {}).get("vertex_count", 0),
            metadata.get("properties", {}).get("euler_characteristic", 0),
            metadata.get("properties", {}).get("edge_to_vertex_ratio", 0.0),
            metadata.get("properties", {}).get("face_to_edge_ratio", 0.0),
        ])
        
        # Extract surface composition
        surface_comp = metadata.get("properties", {}).get("surface_composition", {})
        features.extend([
            surface_comp.get("planes", 0),
            surface_comp.get("cylinders", 0),
            surface_comp.get("cones", 0),
            surface_comp.get("spheres", 0),
            surface_comp.get("tori", 0),
            surface_comp.get("bezier", 0),
            surface_comp.get("bspline", 0),
            surface_comp.get("revolution", 0),
            surface_comp.get("extrusion", 0),
            surface_comp.get("offset", 0),
            surface_comp.get("other", 0),
        ])
        
        # Extract surface ratios
        surface_ratios = metadata.get("properties", {}).get("surface_ratios", {})
        features.extend([
            surface_ratios.get("plane_ratio", 0.0),
            surface_ratios.get("cylinder_ratio", 0.0),
            surface_ratios.get("cone_ratio", 0.0),
            surface_ratios.get("sphere_ratio", 0.0),
            surface_ratios.get("torus_ratio", 0.0),
        ])
        
        # Extract primary shape data - convert to one-hot encoding
        primary_types = ["planar", "cylindrical", "spherical", "conical", "toroidal", "hybrid", "complex", "manufactured", "unknown"]
        primary_type = metadata.get("properties", {}).get("primary_type", "unknown")
        for type_name in primary_types:
            features.append(1.0 if primary_type == type_name else 0.0)
        
        # Add cylinder, cone, sphere, torus counts
        features.extend([
            metadata.get("properties", {}).get("cylinders", {}).get("count", 0),
            metadata.get("properties", {}).get("cones", {}).get("count", 0),
            metadata.get("properties", {}).get("spheres", {}).get("count", 0),
            metadata.get("properties", {}).get("tori", {}).get("count", 0),
        ])
        
        return features
    
    def encode_metadata(self, metadata, normalize=True):
        """
        Encode metadata into an embedding vector
        
        Args:
            metadata (dict): Part metadata
            normalize (bool): Whether to normalize the features
            
        Returns:
            embedding (torch.Tensor): Metadata embedding
        """
        # Extract features
        features = self.extract_features(metadata)
        
        # Convert to tensor
        features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
        
        # Normalize if requested
        if normalize:
            features_tensor = (features_tensor - features_tensor.mean()) / (features_tensor.std() + 1e-6)
        
        # Pass through the encoder
        with torch.no_grad():
            embedding = self.encoder(features_tensor)
        
        return embedding.unsqueeze(0)  # Add batch dimension
    
    def encode_batch(self, metadata_list, normalize=True):
        """
        Encode a batch of metadata
        
        Args:
            metadata_list (list): List of metadata dictionaries
            normalize (bool): Whether to normalize the features
            
        Returns:
            embeddings (torch.Tensor): Batch of metadata embeddings
        """
        embeddings = []
        for metadata in metadata_list:
            embedding = self.encode_metadata(metadata, normalize=normalize)
            embeddings.append(embedding)
        
        return torch.cat(embeddings, dim=0)
    
    def load_bom_data(self, bom_path):
        """
        Load BOM data from a file
        
        Args:
            bom_path (str): Path to BOM JSON file
            
        Returns:
            data (dict): BOM data
        """
        try:
            with open(bom_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading BOM data from {bom_path}: {e}")
            return {}
    
    def find_part_metadata(self, bom_data, part_name):
        """
        Find metadata for a specific part in BOM data
        
        Args:
            bom_data (dict): BOM data
            part_name (str): Name of the part to find
            
        Returns:
            metadata (dict): Part metadata or None if not found
        """
        if not bom_data:
            return None
            
        for part in bom_data.get("parts", []):
            if part.get("name") == part_name:
                return part
                
        return None
