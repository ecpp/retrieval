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

            for file in os.listdir(bom_files_dir):
                if file.endswith("_bom.json") or file.endswith(".json"):
                    files_found += 1
                    bom_path = os.path.join(bom_files_dir, file)
                    print(f"Processing {file}...")

                    try:
                        # Load the JSON file
                        with open(bom_path, 'r') as f:
                            bom_data = json.load(f)

                        files_loaded += 1

                        # Print structure for the first file to help debug
                        if files_loaded == 1:
                            print(f"BOM file structure keys: {list(bom_data.keys())}")

                        # In this BOM format, each top-level key is a part
                        for part_name, part_data in bom_data.items():
                            try:
                                # Skip non-dictionary values
                                if not isinstance(part_data, dict):
                                    continue

                                # Extract features directly from the part data
                                # (it already contains the properties dictionary)
                                features = self.metadata_encoder.extract_features(part_data)
                                if len(features) == self.metadata_encoder.get_input_dim():
                                    self.features_list.append(features)
                                    parts_processed += 1
                                    if parts_processed % 50 == 0:
                                        print(f"Processed {parts_processed} parts so far")
                            except Exception as e:
                                print(f"Error extracting features from part '{part_name}': {e}")
                                if parts_processed == 0:
                                    # Print the part structure to help debug
                                    print(f"Part structure sample: {list(part_data.keys())[:10] if isinstance(part_data, dict) else type(part_data)}")

                    except Exception as e:
                        print(f"Error processing BOM file {file}: {e}")

            print(f"Found {files_found} BOM files, loaded {files_loaded} successfully, processed {parts_processed} parts")
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
    def __init__(self, output_dim=256, hidden_dims=[512, 384]):
        """
        Initialize the metadata encoder with autoencoder architecture

        Args:
            output_dim (int): Dimension of the output embedding (latent space)
            hidden_dims (list): Dimensions of hidden layers in encoder/decoder
        """
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

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

        # Extract dimensional data with fallbacks
        try:
            features.extend([
                float(properties.get("length", properties.get("Length", 0.0))),
                float(properties.get("width", properties.get("Width", 0.0))),
                float(properties.get("height", properties.get("Height", 0.0))),
                float(properties.get("max_dimension", properties.get("MaxDimension", 0.0))),
                float(properties.get("min_dimension", properties.get("MinDimension", 0.0))),
                float(properties.get("mid_dimension", properties.get("MidDimension", 0.0))),
            ])
        except (ValueError, TypeError) as e:
            # If conversion fails, use zeros
            print(f"Warning: Could not extract dimensional data: {e}")
            features.extend([0.0] * 6)

        # Extract derived dimensional ratios
        try:
            features.extend([
                float(properties.get("thickness_ratio", properties.get("ThicknessRatio", 0.0))),
                float(properties.get("width_ratio", properties.get("WidthRatio", 0.0))),
                float(properties.get("elongation", properties.get("Elongation", 0.0))),
                float(properties.get("volume_ratio", properties.get("VolumeRatio", 0.0))),
                float(properties.get("complexity", properties.get("Complexity", 0.0))),
            ])
        except (ValueError, TypeError):
            features.extend([0.0] * 5)

        # Extract volume and surface area
        try:
            features.extend([
                float(properties.get("volume", properties.get("Volume", 0.0))),
                float(properties.get("surface_area", properties.get("SurfaceArea", 0.0))),
            ])
        except (ValueError, TypeError):
            features.extend([0.0] * 2)

        # Extract topological metrics
        try:
            features.extend([
                int(properties.get("face_count", properties.get("FaceCount", 0))),
                int(properties.get("edge_count", properties.get("EdgeCount", 0))),
                int(properties.get("vertex_count", properties.get("VertexCount", 0))),
                int(properties.get("euler_characteristic", properties.get("EulerCharacteristic", 0))),
                float(properties.get("edge_to_vertex_ratio", properties.get("EdgeToVertexRatio", 0.0))),
                float(properties.get("face_to_edge_ratio", properties.get("FaceToEdgeRatio", 0.0))),
            ])
        except (ValueError, TypeError):
            features.extend([0, 0, 0, 0, 0.0, 0.0])

        # Extract surface composition with fallbacks for capitalized keys
        surface_comp = properties.get("surface_composition", properties.get("SurfaceComposition", {}))
        if not isinstance(surface_comp, dict):
            surface_comp = {}

        try:
            features.extend([
                int(surface_comp.get("planes", surface_comp.get("Planes", 0))),
                int(surface_comp.get("cylinders", surface_comp.get("Cylinders", 0))),
                int(surface_comp.get("cones", surface_comp.get("Cones", 0))),
                int(surface_comp.get("spheres", surface_comp.get("Spheres", 0))),
                int(surface_comp.get("tori", surface_comp.get("Tori", 0))),
                int(surface_comp.get("bezier", surface_comp.get("Bezier", 0))),
                int(surface_comp.get("bspline", surface_comp.get("BSpline", 0))),
                int(surface_comp.get("revolution", surface_comp.get("Revolution", 0))),
                int(surface_comp.get("extrusion", surface_comp.get("Extrusion", 0))),
                int(surface_comp.get("offset", surface_comp.get("Offset", 0))),
                int(surface_comp.get("other", surface_comp.get("Other", 0))),
            ])
        except (ValueError, TypeError):
            features.extend([0] * 11)

        # Extract surface ratios
        surface_ratios = properties.get("surface_ratios", properties.get("SurfaceRatios", {}))
        if not isinstance(surface_ratios, dict):
            surface_ratios = {}

        try:
            features.extend([
                float(surface_ratios.get("plane_ratio", surface_ratios.get("PlaneRatio", 0.0))),
                float(surface_ratios.get("cylinder_ratio", surface_ratios.get("CylinderRatio", 0.0))),
                float(surface_ratios.get("cone_ratio", surface_ratios.get("ConeRatio", 0.0))),
                float(surface_ratios.get("sphere_ratio", surface_ratios.get("SphereRatio", 0.0))),
                float(surface_ratios.get("torus_ratio", surface_ratios.get("TorusRatio", 0.0))),
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
                cyl_count = cylinders.get("count", 0)
            else:
                cyl_count = int(cylinders) if str(cylinders).isdigit() else 0

            if isinstance(cones, dict):
                cone_count = cones.get("count", 0)
            else:
                cone_count = int(cones) if str(cones).isdigit() else 0

            if isinstance(spheres, dict):
                sphere_count = spheres.get("count", 0)
            else:
                sphere_count = int(spheres) if str(spheres).isdigit() else 0

            if isinstance(tori, dict):
                tori_count = tori.get("count", 0)
            else:
                tori_count = int(tori) if str(tori).isdigit() else 0

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

        # Normalize if requested
        if normalize:
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

                # Forward pass - reconstruct input
                reconstructed = self.reconstruct(batch)

                # Compute reconstruction loss
                loss = self.criterion(reconstructed, batch)

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
                "decoder": self.decoder.state_dict()
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
