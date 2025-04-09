import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Check if torch_geometric is available
try:
    from torch_geometric.nn import GCNConv, global_mean_pool
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("Warning: torch_geometric not available, using simplified GNN implementation")
    
    # Simple alternative implementation if torch_geometric is not available
    class GCNConv(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(GCNConv, self).__init__()
            self.linear = nn.Linear(in_channels, out_channels)
            
        def forward(self, x, edge_index):
            # Simple adjacency-based message passing
            # This is a simplified version without proper normalization
            src, dst = edge_index
            
            # Aggregate messages (simplified)
            for i in range(x.size(0)):
                neighbors = dst[src == i]
                if len(neighbors) > 0:
                    # Simple mean aggregation of neighbors
                    x[i] = x[i] + torch.mean(x[neighbors], dim=0)
            
            # Apply linear transformation
            x = self.linear(x)
            return x

    def global_mean_pool(x, batch=None):
        # Simple global mean pooling
        if batch is None:
            return torch.mean(x, dim=0, keepdim=True)
        else:
            output = torch.zeros((batch.max().item() + 1, x.size(1)), device=x.device)
            for i in range(batch.max().item() + 1):
                mask = (batch == i)
                output[i] = torch.mean(x[mask], dim=0)
            return output

class AssemblyGNN(nn.Module):
    """
    GNN for encoding assembly hierarchical graphs
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(AssemblyGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Input convolution
        self.conv_first = GCNConv(input_dim, hidden_dim)
        
        # Hidden convolutions
        self.convs = nn.ModuleList()
        for i in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Output convolution
        self.conv_last = GCNConv(hidden_dim, output_dim)
        
        # MLP for final readout
        self.mlp = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, x, edge_index, batch=None):
        # First convolution
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Hidden convolutions
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
        
        # Last convolution
        x = self.conv_last(x, edge_index)
        
        # Global pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = torch.mean(x, dim=0, keepdim=True)
        
        # MLP readout
        x = self.mlp(x)
        
        return x

class AssemblyEncoder:
    """
    Encoder for assembly hierarchical graphs
    """
    def __init__(self, feature_dim=512, hidden_dim=256, embedding_dim=768, device=None):
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        # Determine device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Assembly encoder using device: {self.device}")
        
        # Initialize the GNN model
        self.gnn = AssemblyGNN(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim
        ).to(self.device)
        
        self.trained = False
    
    def load_graph(self, graph_path):
        """
        Load hierarchical graph from GraphML file
        
        Args:
            graph_path (str): Path to the hierarchical graph GraphML file
            
        Returns:
            graph_data (dict): Processed graph data ready for GNN
        """
        try:
            # Check if networkx is available for GraphML parsing
            try:
                import networkx as nx
                NETWORKX_AVAILABLE = True
            except ImportError:
                NETWORKX_AVAILABLE = False
                print("Warning: networkx not available, using simplified GraphML parsing")
            
            if NETWORKX_AVAILABLE:
                # Use networkx to load GraphML
                G = nx.read_graphml(graph_path)
                
                # Convert networkx graph to our format
                nodes = []
                edges = []
                node_map = {}  # Map node IDs to indices
                
                # Process nodes
                for i, (node_id, node_data) in enumerate(G.nodes(data=True)):
                    node_map[node_id] = i
                    
                    # Create node features (can be extended with more attributes)
                    features = [0.0] * self.feature_dim
                    
                    # Set features based on node type
                    node_type = node_data.get('type', '').lower()
                    if node_type == 'part':
                        features[0] = 1.0  # One-hot encoding for part
                    elif node_type == 'assembly':
                        features[1] = 1.0  # One-hot encoding for assembly
                    elif node_type == 'shell':
                        features[2] = 1.0  # One-hot encoding for shell
                    elif node_type == 'face':
                        features[3] = 1.0  # One-hot encoding for face
                    elif node_type == 'edge':
                        features[4] = 1.0  # One-hot encoding for edge
                    
                    # Add positional information if available
                    if 'position_x' in node_data and 'position_y' in node_data and 'position_z' in node_data:
                        # Normalize position values to reasonable ranges
                        try:
                            features[5] = float(node_data['position_x']) / 1000.0 if abs(float(node_data['position_x'])) < 1000000 else 0.0
                            features[6] = float(node_data['position_y']) / 1000.0 if abs(float(node_data['position_y'])) < 1000000 else 0.0
                            features[7] = float(node_data['position_z']) / 1000.0 if abs(float(node_data['position_z'])) < 1000000 else 0.0
                        except (ValueError, TypeError):
                            pass  # Skip if conversion fails
                    
                    # Add any numerical attributes
                    attr_idx = 8
                    for key, value in node_data.items():
                        if key not in ['type', 'position_x', 'position_y', 'position_z'] and attr_idx < self.feature_dim:
                            try:
                                # Try to convert to float and normalize
                                val = float(value) / 1000.0 if abs(float(value)) < 1000000 else 0.0
                                features[attr_idx] = val
                                attr_idx += 1
                            except (ValueError, TypeError):
                                pass  # Skip if conversion fails
                    
                    nodes.append(features)
                
                # Process edges
                for u, v in G.edges():
                    if u in node_map and v in node_map:
                        source_idx = node_map[u]
                        target_idx = node_map[v]
                        edges.append([source_idx, target_idx])
                
                # If no edges, create self-loops for all nodes
                if not edges and nodes:
                    for i in range(len(nodes)):
                        edges.append([i, i])
                
                # Convert to torch tensors
                try:
                    node_features = torch.tensor(nodes, dtype=torch.float).to(self.device)
                    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(self.device)
                    
                    return {
                        'node_features': node_features,
                        'edge_index': edge_index,
                        'num_nodes': len(nodes)
                    }
                except Exception as e:
                    print(f"Error converting graph to tensors: {e}")
                    return None
                    
            else:
                # Fallback: manual XML parsing if networkx is not available
                import xml.etree.ElementTree as ET
                
                # Parse GraphML file
                tree = ET.parse(graph_path)
                root = tree.getroot()
                
                # Find namespace if any
                ns = root.tag.split('}')[0] + '}' if '}' in root.tag else ''
                
                # Extract nodes and edges
                nodes = []
                edges = []
                node_map = {}  # Map node IDs to indices
                
                # Process nodes
                for i, node in enumerate(root.findall('.//' + ns + 'node')):
                    node_id = node.get('id')
                    node_map[node_id] = i
                    
                    # Create node features
                    features = [0.0] * self.feature_dim
                    
                    # Extract node type and other data
                    for data in node.findall('.//' + ns + 'data'):
                        key = data.get('key')
                        if key == 'type':
                            node_type = data.text.lower() if data.text else ''
                            if node_type == 'part':
                                features[0] = 1.0
                            elif node_type == 'assembly':
                                features[1] = 1.0
                            elif node_type == 'shell':
                                features[2] = 1.0
                            elif node_type == 'face':
                                features[3] = 1.0
                            elif node_type == 'edge':
                                features[4] = 1.0
                        elif key == 'position_x':
                            try:
                                val = float(data.text) if data.text else 0.0
                                features[5] = val / 1000.0 if abs(val) < 1000000 else 0.0
                            except (ValueError, TypeError):
                                pass
                        elif key == 'position_y':
                            try:
                                val = float(data.text) if data.text else 0.0
                                features[6] = val / 1000.0 if abs(val) < 1000000 else 0.0
                            except (ValueError, TypeError):
                                pass
                        elif key == 'position_z':
                            try:
                                val = float(data.text) if data.text else 0.0
                                features[7] = val / 1000.0 if abs(val) < 1000000 else 0.0
                            except (ValueError, TypeError):
                                pass
                        else:
                            # Try to convert other attributes to float
                            attr_idx = 8
                            try:
                                val = float(data.text) if data.text else 0.0
                                if attr_idx < self.feature_dim:
                                    features[attr_idx] = val / 1000.0 if abs(val) < 1000000 else 0.0
                                    attr_idx += 1
                            except (ValueError, TypeError):
                                pass
                    
                    nodes.append(features)
                
                # Process edges
                for edge in root.findall('.//' + ns + 'edge'):
                    source_id = edge.get('source')
                    target_id = edge.get('target')
                    
                    if source_id in node_map and target_id in node_map:
                        source_idx = node_map[source_id]
                        target_idx = node_map[target_id]
                        edges.append([source_idx, target_idx])
                
                # If no edges, create self-loops
                if not edges and nodes:
                    for i in range(len(nodes)):
                        edges.append([i, i])
                
                # Convert to torch tensors
                try:
                    node_features = torch.tensor(nodes, dtype=torch.float).to(self.device)
                    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(self.device)
                    
                    return {
                        'node_features': node_features,
                        'edge_index': edge_index,
                        'num_nodes': len(nodes)
                    }
                except Exception as e:
                    print(f"Error converting graph to tensors: {e}")
                    return None
                
        except Exception as e:
            print(f"Error loading graph from {graph_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _process_graph(self, graph_data):
        """
        Process hierarchical graph data for GNN
        
        Args:
            graph_data (dict): Raw hierarchical graph data
            
        Returns:
            processed_data (dict): Processed data with node features and edge indices
        """
        # Extract nodes and edges from hierarchical graph
        nodes = []
        edges = []
        node_map = {}  # Map node IDs to indices
        
        # Process nodes
        for i, node in enumerate(graph_data.get('nodes', [])):
            node_map[node['id']] = i
            # Create node features (can be extended with more attributes)
            features = [0.0] * self.feature_dim
            
            # Set features based on node type
            node_type = node.get('type', '').lower()
            if node_type == 'part':
                features[0] = 1.0  # One-hot encoding for part
            elif node_type == 'assembly':
                features[1] = 1.0  # One-hot encoding for assembly
            elif node_type == 'shell':
                features[2] = 1.0  # One-hot encoding for shell
            elif node_type == 'face':
                features[3] = 1.0  # One-hot encoding for face
            elif node_type == 'edge':
                features[4] = 1.0  # One-hot encoding for edge
            
            # Add positional information if available
            if 'position' in node:
                pos = node['position']
                if isinstance(pos, list) and len(pos) >= 3:
                    # Normalize position values to reasonable ranges
                    features[5] = float(pos[0]) / 1000.0 if abs(float(pos[0])) < 1000000 else 0.0
                    features[6] = float(pos[1]) / 1000.0 if abs(float(pos[1])) < 1000000 else 0.0
                    features[7] = float(pos[2]) / 1000.0 if abs(float(pos[2])) < 1000000 else 0.0
            
            # Add any numerical attributes (size, volume, etc.)
            attrs = node.get('attributes', {})
            if isinstance(attrs, dict):
                for j, (key, value) in enumerate(attrs.items()):
                    if isinstance(value, (int, float)) and j + 8 < self.feature_dim:
                        # Normalize values to reasonable ranges
                        features[j + 8] = float(value) / 1000.0 if abs(float(value)) < 1000000 else 0.0
            
            nodes.append(features)
        
        # Process edges
        for edge in graph_data.get('edges', []):
            source_id = edge.get('source')
            target_id = edge.get('target')
            
            if source_id in node_map and target_id in node_map:
                source_idx = node_map[source_id]
                target_idx = node_map[target_id]
                edges.append([source_idx, target_idx])
                
                # Add reverse edge for undirected graph
                edges.append([target_idx, source_idx])
        
        # If no edges, create self-loops for all nodes
        if not edges and nodes:
            for i in range(len(nodes)):
                edges.append([i, i])
        
        # Convert to torch tensors
        try:
            node_features = torch.tensor(nodes, dtype=torch.float).to(self.device)
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(self.device)
            
            return {
                'node_features': node_features,
                'edge_index': edge_index,
                'num_nodes': len(nodes)
            }
        except Exception as e:
            print(f"Error converting graph to tensors: {e}")
            return None
    
    def encode_graph(self, graph_data):
        """
        Encode graph data into embedding
        
        Args:
            graph_data (dict): Processed graph data with node features and edge indices
            
        Returns:
            embedding (torch.Tensor): Graph embedding
        """
        if not graph_data:
            return None
        
        with torch.no_grad():
            self.gnn.eval()
            embedding = self.gnn(
                graph_data['node_features'],
                graph_data['edge_index']
            )
        
        return embedding
    
    def extract_step_id(self, graph_path):
        """
        Extract STEP file ID from graph path
        
        Args:
            graph_path (str): Path to the hierarchical graph JSON file
            
        Returns:
            step_id (str): ID of the STEP file
        """
        try:
            # Get the filename without extension
            filename = os.path.basename(graph_path)
            name_without_ext = os.path.splitext(filename)[0]
            
            # Assuming format is something like "step_id_graph.json" or "step_id_hierarchical_graph.json"
            if "_hierarchical" in name_without_ext:
                # Remove the "_graph" or "_hierarchical_graph" suffix
                step_id = name_without_ext.split("_hierarchical")[0]
                return step_id
            else:
                print("Name without ext", name_without_ext)
                # Just return the filename without extension
                return name_without_ext
        except Exception as e:
            print(f"Error extracting STEP ID from {graph_path}: {e}")
            return os.path.basename(graph_path)
    
    def train_gnn(self, graph_dir, epochs=50, batch_size=8, lr=1e-4, save_path=None):
        """
        Train the GNN on a dataset of hierarchical graphs using contrastive learning
        or classification if triplet learning is not possible.
        
        Args:
            graph_dir (str): Directory containing hierarchical graph files
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            lr (float): Learning rate
            save_path (str): Path to save the trained model
            
        Returns:
            history (dict): Training history
        """
        try:
            # Find all graph files
            graph_paths = []
            
            for root, _, files in os.walk(graph_dir):
                for file in files:
                    if file.endswith("_hierarchical.graphml"):
                        graph_paths.append(os.path.join(root, file))
            
            if not graph_paths:
                print(f"No graph files found in {graph_dir}. Looking for files ending with '_hierarchical.graphml'")
                return {"train_loss": []}
            
            print(f"Found {len(graph_paths)} graph files for training")
            
            # Debug: print first few files to verify
            for i, path in enumerate(graph_paths[:5]):
                print(f"  File {i+1}: {os.path.basename(path)}")
            
            # Extract IDs and check if we have multiple files per ID
            step_ids = {}
            for path in graph_paths:
                basename = os.path.basename(path)
                try:
                    step_id = basename.split('_')[0]
                    if step_id not in step_ids:
                        step_ids[step_id] = []
                    step_ids[step_id].append(path)
                except Exception as e:
                    print(f"Error parsing filename {basename}: {e}")
                    continue
            
            valid_groups = {k: v for k, v in step_ids.items() if len(v) >= 2}
            print(f"Found {len(valid_groups)} groups with at least 2 files each")
            
            # If we don't have valid groups for triplet learning, use an alternative approach
            if len(valid_groups) < 2:
                print("Not enough groups for triplet learning. Using contrastive learning with random pairs instead.")
                return self._train_with_contrastive(graph_paths, epochs, batch_size, lr, save_path)
            else:
                # Use original triplet learning approach
                return self._train_with_triplets(valid_groups, epochs, batch_size, lr, save_path)
        
        except Exception as e:
            print(f"Error in training preparation: {e}")
            import traceback
            traceback.print_exc()
            return {"train_loss": []}

    def _train_with_contrastive(self, graph_paths, epochs, batch_size, lr, save_path):
        """
        Train using contrastive learning with random pairs when triplet learning is not possible
        """
        try:
            # Import tqdm for progress reporting
            from tqdm import tqdm

            # Initialize optimizer
            optimizer = torch.optim.Adam(self.gnn.parameters(), lr=lr)
            
            # Contrastive loss (using cosine similarity)
            contrastive_loss = nn.CosineEmbeddingLoss(margin=0.5)
            
            # Training history
            history = {"train_loss": []}
            
            # Training loop
            self.gnn.train()
            # Add progress bar for epochs
            for epoch in tqdm(range(epochs), desc="Training epochs", unit="epoch"):
                epoch_loss = 0.0
                batch_count = 0
                
                # Shuffle paths for this epoch
                np.random.shuffle(graph_paths)
                
                # Create random pairs
                pairs = []
                n = len(graph_paths)
                
                # Create pairs with alternating same/different labels
                for i in range(0, n-1, 2):
                    # Similar pair (treat as similar with label 1)
                    pairs.append((graph_paths[i], graph_paths[(i+1) % n], 1))
                    
                    # Dissimilar pair (treat as dissimilar with label -1)
                    pairs.append((graph_paths[i], graph_paths[(i+n//2) % n], -1))
                
                print(f"Created {len(pairs)} training pairs for epoch {epoch+1}")
                
                # Process pairs in batches with progress bar
                batch_pbar = tqdm(range(0, len(pairs), batch_size), desc=f"Epoch {epoch+1} batches", unit="batch")
                for i in batch_pbar:
                    batch_pairs = pairs[i:i+batch_size]
                    
                    inputs1, inputs2, labels = [], [], []
                    valid_pairs = 0
                    
                    # Add inner progress bar for batch processing
                    for path1, path2, label in tqdm(batch_pairs, desc=f"Batch {i//batch_size + 1} pairs", leave=False):
                        try:
                            # Load and process graphs
                            data1 = self.load_graph(path1)
                            data2 = self.load_graph(path2)
                            
                            if not data1 or not data2:
                                continue
                            
                            # Encode graphs
                            emb1 = self.gnn(data1['node_features'], data1['edge_index'])
                            emb2 = self.gnn(data2['node_features'], data2['edge_index'])
                            
                            inputs1.append(emb1)
                            inputs2.append(emb2)
                            labels.append(torch.tensor([label], dtype=torch.float, device=self.device))
                            valid_pairs += 1
                            
                        except Exception as e:
                            print(f"Error processing pair: {e}")
                            continue
                    
                    # Skip if no valid pairs
                    if valid_pairs == 0:
                        batch_pbar.set_postfix({"status": "skipped - no valid pairs"})
                        continue
                        
                    # Concatenate embeddings and labels
                    input1_batch = torch.cat(inputs1, dim=0)
                    input2_batch = torch.cat(inputs2, dim=0)
                    label_batch = torch.cat(labels, dim=0)
                    
                    # Zero gradients
                    optimizer.zero_grad()
                    
                    # Compute loss
                    loss = contrastive_loss(input1_batch, input2_batch, label_batch)
                    
                    # Backpropagation
                    loss.backward()
                    optimizer.step()
                    
                    # Update metrics
                    current_loss = loss.item()
                    epoch_loss += current_loss
                    batch_count += 1
                    
                    # Update progress bar with loss
                    batch_pbar.set_postfix({"valid_pairs": valid_pairs, "loss": f"{current_loss:.6f}"})
                
                # Print progress
                if batch_count > 0:
                    avg_loss = epoch_loss / batch_count
                    print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.6f}")
                    history["train_loss"].append(avg_loss)
                    
                    # Early stopping (simple)
                    if avg_loss < 0.1 and epoch >= 10:  # Higher threshold and more epochs for contrastive
                        print(f"Early stopping at epoch {epoch+1} with loss {avg_loss:.6f}")
                        break
                else:
                    print(f"Warning: Epoch {epoch+1} had no valid batches")
                    history["train_loss"].append(float('nan'))
            
            # Save trained model
            if save_path and batch_count > 0:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(self.gnn.state_dict(), save_path)
                print(f"Model saved to {save_path}")
            
            self.trained = batch_count > 0
            return history
        
        except Exception as e:
            print(f"Error in contrastive training: {e}")
            import traceback
            traceback.print_exc()
            return {"train_loss": []}

    def _train_with_triplets(self, valid_groups, epochs, batch_size, lr, save_path):
        """
        Original triplet learning approach when we have enough valid groups
        """
        try:
            # Import tqdm for progress reporting
            from tqdm import tqdm
            
            # Initialize optimizer
            optimizer = torch.optim.Adam(self.gnn.parameters(), lr=lr)
            
            # Triplet loss function
            triplet_loss = nn.TripletMarginLoss(margin=1.0)
            
            # Training history
            history = {"train_loss": []}
            
            # Training loop
            self.gnn.train()
            # Add progress bar for epochs
            for epoch in tqdm(range(epochs), desc="Training epochs", unit="epoch"):
                epoch_loss = 0.0
                batch_count = 0
                
                # Create triplets for training
                triplets = []
                for step_id, paths in valid_groups.items():
                    # For each graph in the group, create triplets
                    for i, anchor_path in enumerate(paths):
                        # Positive: another graph from same STEP
                        positive_idx = (i + 1) % len(paths)  # Next graph in same group
                        positive_path = paths[positive_idx]
                        
                        # Negative: a graph from another STEP
                        other_steps = [s for s in valid_groups.keys() if s != step_id]
                        negative_step = np.random.choice(other_steps)
                        negative_path = np.random.choice(valid_groups[negative_step])
                        
                        triplets.append((anchor_path, positive_path, negative_path))
                
                print(f"Created {len(triplets)} triplets for epoch {epoch+1}")
                
                # Shuffle triplets
                np.random.shuffle(triplets)
                
                # Process in batches with progress bar
                batch_pbar = tqdm(range(0, len(triplets), batch_size), desc=f"Epoch {epoch+1} batches", unit="batch")
                for i in batch_pbar:
                    batch_triplets = triplets[i:i+batch_size]
                    
                    # Process each triplet
                    anchors, positives, negatives = [], [], []
                    valid_triplets = 0
                    
                    # Add inner progress bar for triplet processing
                    for anchor_path, positive_path, negative_path in tqdm(batch_triplets, desc=f"Batch {i//batch_size + 1} triplets", leave=False):
                        try:
                            # Load and process graphs
                            anchor_data = self.load_graph(anchor_path)
                            positive_data = self.load_graph(positive_path)
                            negative_data = self.load_graph(negative_path)
                            
                            if not anchor_data or not positive_data or not negative_data:
                                continue
                            
                            # Encode graphs
                            anchor_emb = self.gnn(anchor_data['node_features'], anchor_data['edge_index'])
                            positive_emb = self.gnn(positive_data['node_features'], positive_data['edge_index'])
                            negative_emb = self.gnn(negative_data['node_features'], negative_data['edge_index'])
                            
                            anchors.append(anchor_emb)
                            positives.append(positive_emb)
                            negatives.append(negative_emb)
                            valid_triplets += 1
                        except Exception as e:
                            print(f"Error processing triplet: {e}")
                            continue
                    
                    # Skip if no valid triplets
                    if valid_triplets == 0:
                        batch_pbar.set_postfix({"status": "skipped - no valid triplets"})
                        continue
                    
                    # Concatenate embeddings
                    anchor_batch = torch.cat(anchors, dim=0)
                    positive_batch = torch.cat(positives, dim=0)
                    negative_batch = torch.cat(negatives, dim=0)
                    
                    # Zero gradients
                    optimizer.zero_grad()
                    
                    # Compute loss
                    loss = triplet_loss(anchor_batch, positive_batch, negative_batch)
                    
                    # Backpropagation
                    loss.backward()
                    optimizer.step()
                    
                    # Update metrics
                    current_loss = loss.item()
                    epoch_loss += current_loss
                    batch_count += 1
                    
                    # Update progress bar with loss
                    batch_pbar.set_postfix({"valid_triplets": valid_triplets, "loss": f"{current_loss:.6f}"})
                
                # Print progress
                if batch_count > 0:
                    avg_loss = epoch_loss / batch_count
                    print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.6f}")
                    history["train_loss"].append(avg_loss)
                    
                    # Early stopping
                    if avg_loss < 0.01 and epoch >= 5:
                        print(f"Early stopping at epoch {epoch+1} with loss {avg_loss:.6f}")
                        break
                else:
                    print(f"Warning: Epoch {epoch+1} had no valid batches")
                    history["train_loss"].append(float('nan'))
            
            # Save trained model
            if save_path and batch_count > 0:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(self.gnn.state_dict(), save_path)
                print(f"Model saved to {save_path}")
            
            self.trained = batch_count > 0
            return history
        
        except Exception as e:
            print(f"Error in triplet training: {e}")
            import traceback
            traceback.print_exc()
            return {"train_loss": []}
    
    def load_trained_model(self, model_path):
        """
        Load a pre-trained GNN model
        
        Args:
            model_path (str): Path to the trained model
            
        Returns:
            success (bool): Whether the model was loaded successfully
        """
        try:
            # Load pre-trained weights
            self.gnn.load_state_dict(torch.load(model_path, map_location=self.device))
            self.gnn.eval()
            self.trained = True
            print(f"Loaded trained model from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading trained model: {e}")
            return False