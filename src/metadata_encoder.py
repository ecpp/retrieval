import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class BomDataset(Dataset):
    """
    Dataset for BOM metadata training
    """
    def __init__(self, bom_files_dir, metadata_encoder):
        """
        Initialize dataset from a directory of BOM files

        Args:
            bom_files_dir (str): Directory containing BOM JSON files
            metadata_encoder (MetadataEncoder): Encoder used to extract features
        """
        self.features_list = []
        self.metadata_encoder = metadata_encoder

        # Load all BOM files and extract features
        if os.path.exists(bom_files_dir):
            print(f"Looking for BOM files in {bom_files_dir}")
            files_found = 0
            files_loaded = 0
            parts_processed = 0

            # Get all BOM files first
            json_files = [f for f in os.listdir(bom_files_dir) 
                        if f.endswith("_bom.json") or f.endswith(".json")]
            files_found = len(json_files)
            
            # Add tqdm for file processing
            print(f"Found {files_found} BOM files to process")
            for file in tqdm(json_files, desc="Processing BOM files", unit="file"):
                bom_path = os.path.join(bom_files_dir, file)

                try:
                    # Load the JSON file
                    with open(bom_path, 'r') as f:
                        bom_data = json.load(f)

                    files_loaded += 1

                    # Print structure for the first file to help debug
                    if files_loaded == 1:
                        print(f"BOM file structure keys: {list(bom_data.keys())}")

                    # Get all part items
                    part_items = []
                    for part_name, part_data in bom_data.items():
                        if isinstance(part_data, dict):
                            part_items.append((part_name, part_data))
                    
                    # Only show inner progress bar if there are enough parts
                    if len(part_items) > 10:
                        part_iterator = tqdm(part_items, desc=f"  Parts in {file}", unit="part", leave=False)
                    else:
                        part_iterator = part_items
                        
                    # Process each part
                    for part_name, part_data in part_iterator:
                        try:
                            # Extract features directly from the part data
                            # (it already contains the properties dictionary)
                            features = self.metadata_encoder.extract_features(part_data)
                            if len(features) == self.metadata_encoder.get_input_dim():
                                self.features_list.append(features)
                                parts_processed += 1
                        except Exception as e:
                            print(f"Error extracting features from part '{part_name}': {e}")
                            if parts_processed == 0:
                                # Print the part structure to help debug
                                print(f"Part structure sample: {list(part_data.keys())[:10] if isinstance(part_data, dict) else type(part_data)}")

                except Exception as e:
                    print(f"Error processing BOM file {file}: {e}")

            print(f"Processed {files_found} BOM files, loaded {files_loaded} successfully, extracted features from {parts_processed} parts")
        else:
            print(f"Directory {bom_files_dir} does not exist!")

        print(f"Loaded {len(self.features_list)} metadata samples for training")

    def __len__(self):
        return len(self.features_list)

    def __getitem__(self, idx):
        features = self.features_list[idx]
        return torch.tensor(features, dtype=torch.float32)


class MetadataEncoder:
    """
    Encodes CAD part metadata from BOM data into embeddings using an autoencoder
    """
    def __init__(self, output_dim=256, hidden_dims=[512, 384], clip_values=True, normalization=True):
        """
        Initialize the metadata encoder with autoencoder architecture

        Args:
            output_dim (int): Dimension of the output embedding (latent space)
            hidden_dims (list): Dimensions of hidden layers in encoder/decoder
        """
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        self.clip_values = clip_values  # Whether to clip extreme values
        self.normalization = normalization  # Whether to apply feature normalization
        
        # Initialize feature scaling parameters
        self.feature_means = None
        self.feature_stds = None
        self.feature_mins = None
        self.feature_maxs = None
        
        # Set reasonable clipping thresholds for extreme values
        self.min_clip_value = -1e4
        self.max_clip_value = 1e4
        
        # Define encoder component
        encoder_layers = []
        input_dim = self.get_input_dim()

        # Add encoder layers
        prev_dim = input_dim
        for dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = dim

        # Final encoder layer to latent space
        encoder_layers.append(nn.Linear(prev_dim, output_dim))

        self.encoder = nn.Sequential(*encoder_layers)

        # Define decoder component (mirror of encoder)
        decoder_layers = []

        # First decoder layer from latent space
        decoder_layers.append(nn.Linear(output_dim, hidden_dims[-1]))
        decoder_layers.append(nn.ReLU())

        # Add remaining decoder layers
        for i in range(len(hidden_dims)-1, 0, -1):
            decoder_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i-1]))
            decoder_layers.append(nn.ReLU())

        # Final decoder layer to reconstruct input
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))

        self.decoder = nn.Sequential(*decoder_layers)

        # Move to appropriate device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        # Initialize training parameters
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.trained = False

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

        # Handle potentially different metadata structures
        properties = metadata.get("properties", metadata)
        if "type" in metadata and isinstance(metadata.get("properties"), dict):
            # This seems to be the exact format from the get_shape_type function
            properties = metadata.get("properties")

        # Check if we have the required data
        if not properties or not isinstance(properties, dict):
            # Not enough data to extract features
            raise ValueError("No valid properties dictionary found")

        
        # Safe extraction function to handle potential errors
        def safe_extract(value, default=0.0, max_val=1e9, min_val=-1e9):
            try:
                # First convert to float
                val = float(value)
                # Check if value is valid
                if not torch.isfinite(torch.tensor(val)):
                    return default
                # Apply limits
                val = max(min(val, max_val), min_val)
                return val
            except (ValueError, TypeError, OverflowError) as e:
                return default
        

        # Extract dimensional data with fallbacks
        try:
            features.extend([
                safe_extract(properties.get("length", properties.get("Length", 0.0))),
                safe_extract(properties.get("width", properties.get("Width", 0.0))),
                safe_extract(properties.get("height", properties.get("Height", 0.0))),
                safe_extract(properties.get("max_dimension", properties.get("MaxDimension", 0.0))),
                safe_extract(properties.get("min_dimension", properties.get("MinDimension", 0.0))),
                safe_extract(properties.get("mid_dimension", properties.get("MidDimension", 0.0))),
            ])
        except (ValueError, TypeError) as e:
            # If conversion fails, use zeros
            print(f"Warning: Could not extract dimensional data: {e}")
            features.extend([0.0] * 6)

        # Extract derived dimensional ratios
        try:
            features.extend([
                safe_extract(properties.get("thickness_ratio", properties.get("ThicknessRatio", 0.0)), max_val=1.0),
                safe_extract(properties.get("width_ratio", properties.get("WidthRatio", 0.0)), max_val=1.0),
                safe_extract(properties.get("elongation", properties.get("Elongation", 0.0)), max_val=100.0),
                safe_extract(properties.get("volume_ratio", properties.get("VolumeRatio", 0.0)), max_val=1.0),
                safe_extract(properties.get("complexity", properties.get("Complexity", 0.0)), max_val=1.0),
            ])
        except (ValueError, TypeError):
            features.extend([0.0] * 5)

        # Extract volume and surface area
        try:
            features.extend([
                safe_extract(properties.get("volume", properties.get("Volume", 0.0))),
                safe_extract(properties.get("surface_area", properties.get("SurfaceArea", 0.0))),
            ])
        except (ValueError, TypeError):
            features.extend([0.0] * 2)

        # Extract topological metrics
        try:
            features.extend([
                safe_extract(properties.get("face_count", properties.get("FaceCount", 0))),
                safe_extract(properties.get("edge_count", properties.get("EdgeCount", 0))),
                safe_extract(properties.get("vertex_count", properties.get("VertexCount", 0))),
                safe_extract(properties.get("euler_characteristic", properties.get("EulerCharacteristic", 0))),
                safe_extract(properties.get("edge_to_vertex_ratio", properties.get("EdgeToVertexRatio", 0.0)), max_val=10.0),
                safe_extract(properties.get("face_to_edge_ratio", properties.get("FaceToEdgeRatio", 0.0)), max_val=10.0),
            ])
        except (ValueError, TypeError):
            features.extend([0, 0, 0, 0, 0.0, 0.0])

        # Extract surface composition with fallbacks for capitalized keys
        surface_comp = properties.get("surface_composition", properties.get("SurfaceComposition", {}))
        if not isinstance(surface_comp, dict):
            surface_comp = {}

        try:
            features.extend([
                safe_extract(surface_comp.get("planes", surface_comp.get("Planes", 0)), max_val=1000),
                safe_extract(surface_comp.get("cylinders", surface_comp.get("Cylinders", 0)), max_val=1000),
                safe_extract(surface_comp.get("cones", surface_comp.get("Cones", 0)), max_val=1000),
                safe_extract(surface_comp.get("spheres", surface_comp.get("Spheres", 0)), max_val=1000),
                safe_extract(surface_comp.get("tori", surface_comp.get("Tori", 0)), max_val=1000),
                safe_extract(surface_comp.get("bezier", surface_comp.get("Bezier", 0)), max_val=1000),
                safe_extract(surface_comp.get("bspline", surface_comp.get("BSpline", 0)), max_val=1000),
                safe_extract(surface_comp.get("revolution", surface_comp.get("Revolution", 0)), max_val=1000),
                safe_extract(surface_comp.get("extrusion", surface_comp.get("Extrusion", 0)), max_val=1000),
                safe_extract(surface_comp.get("offset", surface_comp.get("Offset", 0)), max_val=1000),
                safe_extract(surface_comp.get("other", surface_comp.get("Other", 0)), max_val=1000),
            ])
        except (ValueError, TypeError):
            features.extend([0] * 11)

        # Extract surface ratios
        surface_ratios = properties.get("surface_ratios", properties.get("SurfaceRatios", {}))
        if not isinstance(surface_ratios, dict):
            surface_ratios = {}

        try:
            features.extend([
                safe_extract(surface_ratios.get("plane_ratio", surface_ratios.get("PlaneRatio", 0.0)), max_val=1.0),
                safe_extract(surface_ratios.get("cylinder_ratio", surface_ratios.get("CylinderRatio", 0.0)), max_val=1.0),
                safe_extract(surface_ratios.get("cone_ratio", surface_ratios.get("ConeRatio", 0.0)), max_val=1.0),
                safe_extract(surface_ratios.get("sphere_ratio", surface_ratios.get("SphereRatio", 0.0)), max_val=1.0),
                safe_extract(surface_ratios.get("torus_ratio", surface_ratios.get("TorusRatio", 0.0)), max_val=1.0),
            ])
        except (ValueError, TypeError):
            features.extend([0.0] * 5)

        # Extract primary shape data - convert to one-hot encoding
        primary_types = ["planar", "cylindrical", "spherical", "conical", "toroidal", "hybrid", "complex", "manufactured", "unknown"]
        primary_type = str(properties.get("primary_type", properties.get("PrimaryType", "unknown"))).lower()
        for type_name in primary_types:
            features.append(1.0 if primary_type == type_name else 0.0)

        # Add cylinder, cone, sphere, torus counts
        cylinders = properties.get("cylinders", properties.get("Cylinders", {}))
        cones = properties.get("cones", properties.get("Cones", {}))
        spheres = properties.get("spheres", properties.get("Spheres", {}))
        tori = properties.get("tori", properties.get("Tori", {}))

        # Handle both dict and scalar values
        try:
            if isinstance(cylinders, dict):
                cyl_count = safe_extract(cylinders.get("count", 0), max_val=1000)
            else:

                cyl_count = safe_extract(cylinders, max_val=1000) if str(cylinders).isdigit() else 0
                
            if isinstance(cones, dict):
                cone_count = safe_extract(cones.get("count", 0), max_val=1000)
            else:

                cone_count = safe_extract(cones, max_val=1000) if str(cones).isdigit() else 0
                

            if isinstance(spheres, dict):
                sphere_count = safe_extract(spheres.get("count", 0), max_val=1000)
            else:

                sphere_count = safe_extract(spheres, max_val=1000) if str(spheres).isdigit() else 0
                

            if isinstance(tori, dict):
                tori_count = safe_extract(tori.get("count", 0), max_val=1000)
            else:
                tori_count = safe_extract(tori, max_val=1000) if str(tori).isdigit() else 0
            

            features.extend([
                cyl_count,
                cone_count,
                sphere_count,
                tori_count,
            ])
        except (ValueError, TypeError):
            features.extend([0] * 4)

        # Ensure we have exactly the expected number of features
        if len(features) != self.get_input_dim():
            # Fill with zeros if we're missing features
            features.extend([0.0] * (self.get_input_dim() - len(features)))
            # Truncate if we have too many
            features = features[:self.get_input_dim()]

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

        
        # Apply preprocessing
        if self.clip_values:
            features_tensor = torch.clamp(features_tensor, self.min_clip_value, self.max_clip_value)
        
        # Normalize if requested and parameters are available
        if normalize:
            if self.feature_means is not None and self.feature_stds is not None:
                # Apply global normalization
                features_tensor = (features_tensor - self.feature_means) / (self.feature_stds + 1e-8)
            else:
                # Fallback to per-sample normalization
                features_tensor = (features_tensor - features_tensor.mean()) / (features_tensor.std() + 1e-6)
        

        # Pass through the encoder
        with torch.no_grad():
            embedding = self.encoder(features_tensor)

        return embedding.unsqueeze(0)  # Add batch dimension

    def reconstruct(self, features_tensor):
        """
        Reconstruct metadata from encoded representation

        Args:
            features_tensor (torch.Tensor): Input features

        Returns:
            reconstructed (torch.Tensor): Reconstructed features
        """
        # Apply preprocessing if the input hasn't been preprocessed yet
        if self.normalization and self.feature_means is not None:
            # Check if data is already normalized by comparing statistics
            tensor_mean = torch.mean(features_tensor).item()
            tensor_std = torch.std(features_tensor).item()
            
            # If statistics suggest this isn't normalized, preprocess it
            if abs(tensor_mean) > 5.0 or abs(tensor_std - 1.0) > 5.0:
                features_tensor = self._preprocess_features(features_tensor)
        
        # Encode
        embedding = self.encoder(features_tensor)
        # Decode
        reconstructed = self.decoder(embedding)
        return reconstructed

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

        
    def _preprocess_features(self, features_tensor):
        """
        Apply preprocessing to features tensor including clipping and normalization
        
        Args:
            features_tensor (torch.Tensor): Input features tensor
            
        Returns:
            processed_tensor (torch.Tensor): Processed features tensor
        """
        # Work on a copy to avoid modifying the original
        processed = features_tensor.clone()
        
        # First, handle infinity and NaN values
        processed = torch.nan_to_num(processed, nan=0.0, posinf=self.max_clip_value, neginf=self.min_clip_value)
        
        # Clip extreme values if enabled
        if self.clip_values:
            processed = torch.clamp(processed, self.min_clip_value, self.max_clip_value)
        
        # Apply feature normalization if enabled and parameters are available
        if self.normalization and self.feature_means is not None and self.feature_stds is not None:
            # Apply z-score normalization (mean 0, std 1)
            # First handle any NaN or infinite values in means/stds
            safe_means = torch.nan_to_num(self.feature_means, nan=0.0)
            safe_stds = torch.nan_to_num(self.feature_stds, nan=1.0)
            # Replace zeros in stds with 1.0 to avoid division by zero
            safe_stds[safe_stds < 1e-8] = 1.0
            # Apply normalization
            processed = (processed - safe_means) / (safe_stds + 1e-8)
        
        # One final check to clean any NaN or infinity that might have been introduced
        processed = torch.nan_to_num(processed, nan=0.0, posinf=3.0, neginf=-3.0)
        
        return processed
        
    def _compute_scaling_parameters(self, dataset):
        """
        Compute feature scaling parameters from the dataset
        
        Args:
            dataset (BomDataset): Dataset to compute parameters from
        """
        if len(dataset) == 0:
            print("Warning: Empty dataset, cannot compute scaling parameters.")
            return
        
        print("Computing scaling parameters from dataset...")
        # Create a dataloader to process the dataset in batches
        dataloader = DataLoader(dataset, batch_size=min(1000, len(dataset)), shuffle=False)
        
        # Collect all feature values
        all_features = []
        for batch in dataloader:
            # Handle any NaN or infinity in input data
            clean_batch = torch.nan_to_num(batch, nan=0.0, posinf=self.max_clip_value, neginf=self.min_clip_value)
            # Apply clipping
            clean_batch = torch.clamp(clean_batch, self.min_clip_value, self.max_clip_value)
            all_features.append(clean_batch)
        
        # Concatenate all batches
        all_features = torch.cat(all_features, dim=0)
        
        # Compute statistics with safeguards against NaN/inf
        self.feature_means = torch.mean(all_features, dim=0).to(self.device)
        self.feature_stds = torch.std(all_features, dim=0).to(self.device)
        self.feature_mins = torch.min(all_features, dim=0)[0].to(self.device)
        self.feature_maxs = torch.max(all_features, dim=0)[0].to(self.device)
        
        # Replace any remaining NaN or infinite values in the statistics
        self.feature_means = torch.nan_to_num(self.feature_means, nan=0.0)
        self.feature_stds = torch.nan_to_num(self.feature_stds, nan=1.0)
        self.feature_mins = torch.nan_to_num(self.feature_mins, nan=-self.max_clip_value, posinf=self.max_clip_value, neginf=-self.max_clip_value)
        self.feature_maxs = torch.nan_to_num(self.feature_maxs, nan=self.max_clip_value, posinf=self.max_clip_value, neginf=-self.max_clip_value)
        
        # Replace zero standard deviations with 1.0 to avoid division by zero
        self.feature_stds[self.feature_stds < 1e-8] = 1.0
        
        # Print summary of scaling parameters
        print("\nFeature scaling parameters computed:")
        print(f"Mean range: [{torch.min(self.feature_means).item():.4f}, {torch.max(self.feature_means).item():.4f}]")
        print(f"Std range: [{torch.min(self.feature_stds).item():.4f}, {torch.max(self.feature_stds).item():.4f}]")
        print(f"Min value: {torch.min(self.feature_mins).item():.4f}")
        print(f"Max value: {torch.max(self.feature_maxs).item():.4f}")
        
        # Identify extreme features
        extreme_threshold = 1e6
        extreme_indices = torch.where(torch.abs(self.feature_maxs) > extreme_threshold)[0]
        if len(extreme_indices) > 0:
            print(f"\nExtreme feature values detected in {len(extreme_indices)} features:")
            for idx in extreme_indices:
                i = idx.item()
                print(f"Feature {i}: min={self.feature_mins[i].item():.4f}, max={self.feature_maxs[i].item():.4f}, mean={self.feature_means[i].item():.4f}, std={self.feature_stds[i].item():.4f}")
    

    def train_autoencoder(self, bom_dir, batch_size=32, epochs=50, lr=1e-4, save_path=None):
        """
        Train the autoencoder using BOM data

        Args:
            bom_dir (str): Directory containing BOM JSON files
            batch_size (int): Batch size for training
            epochs (int): Number of training epochs
            lr (float): Learning rate
            save_path (str): Path to save the trained model

        Returns:
            history (dict): Training history with loss values
        """
        # Create dataset and dataloader
        dataset = BomDataset(bom_dir, self)

        if len(dataset) == 0:
            print("No BOM data found for training. Autoencoder training skipped.")
            return {"train_loss": []}

            
        # Compute feature scaling parameters
        print("Computing feature scaling parameters...")
        self._compute_scaling_parameters(dataset)
            

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Initialize optimizer
        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=lr)

        # Training history
        history = {"train_loss": []}

        # Training loop
        for epoch in range(epochs):
            self.encoder.train()
            self.decoder.train()

            running_loss = 0.0
            num_batches = 0

            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                # Move batch to device
                batch = batch.to(self.device)

                # Zero gradients
                self.optimizer.zero_grad()
                
                # Preprocess the batch
                preprocessed_batch = self._preprocess_features(batch)
                
                # Forward pass - reconstruct input
                reconstructed = self.reconstruct(preprocessed_batch)
                
                # Compute reconstruction loss - compare with preprocessed input, not raw input
                loss = self.criterion(reconstructed, preprocessed_batch)
                

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                # Update statistics
                running_loss += loss.item()
                num_batches += 1

            # Compute average loss for epoch
            avg_loss = running_loss / num_batches
            history["train_loss"].append(avg_loss)

            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

        # Save model if requested
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                "encoder": self.encoder.state_dict(),
                "decoder": self.decoder.state_dict(),
                "feature_means": self.feature_means,
                "feature_stds": self.feature_stds,
                "feature_mins": self.feature_mins,
                "feature_maxs": self.feature_maxs
            }, save_path)
            print(f"Saved trained autoencoder to {save_path}")

        self.trained = True
        return history

    def load_trained_model(self, model_path):
        """
        Load a trained autoencoder model

        Args:
            model_path (str): Path to saved model

        Returns:
            success (bool): Whether loading was successful
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.encoder.load_state_dict(checkpoint["encoder"])
            self.decoder.load_state_dict(checkpoint["decoder"])
            
            # Load scaling parameters if available
            if "feature_means" in checkpoint:
                self.feature_means = checkpoint["feature_means"]
                self.feature_stds = checkpoint["feature_stds"]
                self.feature_mins = checkpoint["feature_mins"]
                self.feature_maxs = checkpoint["feature_maxs"]
                print("Loaded feature scaling parameters from checkpoint")
            
            self.trained = True
            print(f"Successfully loaded trained model from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            return False

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
        if not bom_data or not isinstance(bom_data, dict):
            return None
        # Handle different BOM data structures
        # Case 1: Standard structure with "parts" list
        if "parts" in bom_data and isinstance(bom_data.get("parts"), list):
            for part in bom_data.get("parts", []):
                # Skip if part is not a dictionary
                if not isinstance(part, dict):
                    continue
                    
                if part.get("name") == part_name:
                    return part
        
        # Case 2: Part name is a direct key in the BOM data
        if part_name in bom_data and isinstance(bom_data[part_name], dict):
            return bom_data[part_name]
        

        # Case 3: Search through all top-level keys for matching part
        for key, value in bom_data.items():
            if isinstance(value, dict):
                # Check if this item has a name field that matches
                if value.get("name") == part_name or key == part_name:
                    return value
            elif isinstance(value, list):
                # Check if there's a list with dictionaries
                for item in value:
                    if isinstance(item, dict) and item.get("name") == part_name:
                        return item
        # If we still haven't found it, just print a message and skip this part
        print(f"Warning: Could not find metadata for part '{part_name}' in BOM data")
        return None
