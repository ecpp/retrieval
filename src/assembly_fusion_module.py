import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm

class AssemblyFusionModule:
    """
    Combines part-based features and graph-based features for assembly retrieval
    """
    def __init__(self, 
                 graph_dim=768, 
                 part_feature_dim=768, 
                 output_dim=768, 
                 fusion_method="weighted",
                 graph_weight=0.2,
                 part_weight=0.8):
        """
        Initialize the assembly fusion module

        Args:
            graph_dim (int): Dimension of graph embeddings
            part_feature_dim (int): Dimension of aggregated part features
            output_dim (int): Dimension of fused embeddings
            fusion_method (str): Method for fusion ("concat", "weighted", or "mlp")
            graph_weight (float): Weight for graph features in weighted fusion
            part_weight (float): Weight for part features in weighted fusion
        """
        self.graph_dim = graph_dim
        self.part_feature_dim = part_feature_dim
        self.output_dim = output_dim
        self.fusion_method = fusion_method
        
        # Store weights for weighted fusion
        self.graph_weight = graph_weight
        self.part_weight = part_weight

        # Determine device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # For concat fusion, we need a linear layer to project to output_dim
        if fusion_method == "concat":
            self.projection = nn.Sequential(
                nn.Linear(graph_dim + part_feature_dim, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim)
            ).to(self.device)
        
        # For MLP fusion, use a more sophisticated approach
        elif fusion_method == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(graph_dim + part_feature_dim, output_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(output_dim * 2, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim)
            ).to(self.device)
        
        # For weighted fusion, we'll use the provided weights
        elif fusion_method == "weighted":
            print(f"Initializing weighted fusion with graph_weight={graph_weight:.2f}, part_weight={part_weight:.2f}")
            # Ensure weights are normalized
            total_weight = graph_weight + part_weight
            self.graph_weight = graph_weight / total_weight
            self.part_weight = part_weight / total_weight
            print(f"Normalized weights: graph_weight={self.graph_weight:.2f}, part_weight={self.part_weight:.2f}")
        
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")
        
        print(f"Assembly fusion module initialized with {fusion_method} fusion method")

    def fuse(self, graph_embedding, part_features_embedding):
        """
        Fuse graph and part features embeddings

        Args:
            graph_embedding (torch.Tensor): Graph embedding from GNN
            part_features_embedding (torch.Tensor): Aggregated part features embedding

        Returns:
            fused_embedding (torch.Tensor): Fused embedding
        """
        # Move to the right device
        graph_embedding = graph_embedding.to(self.device)
        part_features_embedding = part_features_embedding.to(self.device)

        # Ensure embeddings have the same batch dimension
        if graph_embedding.shape[0] != part_features_embedding.shape[0]:
            raise ValueError(f"Batch dimensions don't match: {graph_embedding.shape[0]} vs {part_features_embedding.shape[0]}")

        # Apply fusion based on selected method
        if self.fusion_method == "concat":
            # Concatenate along the feature dimension
            combined = torch.cat([graph_embedding, part_features_embedding], dim=1)
            # Project to output dimension
            fused = self.projection(combined)
            return fused

        elif self.fusion_method == "mlp":
            # Concatenate along the feature dimension
            combined = torch.cat([graph_embedding, part_features_embedding], dim=1)
            # Pass through MLP
            fused = self.mlp(combined)
            return fused

        elif self.fusion_method == "weighted":
            # Need to ensure part_features_embedding has same dim as graph_embedding for weighted sum
            if self.graph_dim != self.part_feature_dim:
                # Simple projection via a repeat and slice approach
                if self.part_feature_dim < self.graph_dim:
                    # Repeat part features to match graph dim
                    repeat_factor = self.graph_dim // self.part_feature_dim + 1
                    expanded = part_features_embedding.repeat(1, repeat_factor)
                    part_features_embedding = expanded[:, :self.graph_dim]
                else:
                    # Truncate part features to match graph dim
                    part_features_embedding = part_features_embedding[:, :self.graph_dim]

            # Normalize weights to sum to 1 (even though they might already)
            sum_weights = self.graph_weight + self.part_weight
            norm_graph_weight = self.graph_weight / sum_weights
            norm_part_weight = self.part_weight / sum_weights
            
            # Special case: if either weight is 0, use only the other embedding
            if self.graph_weight <= 0.001:
                print(f"Graph weight is near zero ({self.graph_weight:.4f}), using only part features")
                fused = part_features_embedding
            elif self.part_weight <= 0.001:
                print(f"Part weight is near zero ({self.part_weight:.4f}), using only graph features")
                fused = graph_embedding
            else:
                # Apply weighted sum
                fused = norm_graph_weight * graph_embedding + norm_part_weight * part_features_embedding
            
            # Add debug information
            print(f"Fusion weights applied - Graph: {norm_graph_weight:.2f}, Part: {norm_part_weight:.2f}")
            print(f"Graph embedding shape: {graph_embedding.shape}, Part embedding shape: {part_features_embedding.shape}")
            print(f"Fused embedding shape: {fused.shape}")
            
            return fused

    def aggregate_part_features(self, part_embeddings, aggregation_method="mean"):
        """
        Aggregate multiple part embeddings into a single assembly-level embedding

        Args:
            part_embeddings (torch.Tensor): Tensor of part embeddings [num_parts, embedding_dim]
            aggregation_method (str): Method for aggregation ("mean", "max", "attention")

        Returns:
            aggregated (torch.Tensor): Aggregated embedding [1, embedding_dim]
        """
        if part_embeddings is None or len(part_embeddings) == 0:
            # Return a zero tensor if no parts
            return torch.zeros((1, self.part_feature_dim), device=self.device)

        # Move to device
        part_embeddings = part_embeddings.to(self.device)

        # Apply aggregation based on selected method
        if aggregation_method == "mean":
            # Mean pooling
            return torch.mean(part_embeddings, dim=0, keepdim=True)
        
        elif aggregation_method == "max":
            # Max pooling
            return torch.max(part_embeddings, dim=0, keepdim=True)[0]
        
        elif aggregation_method == "weighted":
            # Simple weighted sum based on L2 norm of each embedding
            # The idea is that more distinctive parts should have higher magnitude embeddings
            norms = torch.norm(part_embeddings, dim=1, keepdim=True)
            weights = norms / (torch.sum(norms) + 1e-8)
            weighted_sum = torch.sum(part_embeddings * weights, dim=0, keepdim=True)
            return weighted_sum
        
        else:
            # Default to mean
            return torch.mean(part_embeddings, dim=0, keepdim=True)
