import torch
import torch.nn as nn

class FusionModule:
    """
    Combines visual and metadata embeddings
    """
    def __init__(self, visual_dim=768, metadata_dim=256, output_dim=768, fusion_method="concat"):
        """
        Initialize the fusion module

        Args:
            visual_dim (int): Dimension of visual embeddings
            metadata_dim (int): Dimension of metadata embeddings
            output_dim (int): Dimension of fused embeddings
            fusion_method (str): Method for fusion ("concat" or "weighted")
        """
        self.visual_dim = visual_dim
        self.metadata_dim = metadata_dim
        self.output_dim = output_dim
        self.fusion_method = fusion_method

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # For concat fusion, we need a linear layer to project to output_dim
        if fusion_method == "concat":
            self.projection = nn.Linear(visual_dim + metadata_dim, output_dim)
            self.projection.to(self.device)
        # For weighted fusion, we use a learned weight parameter
        elif fusion_method == "weighted":
            # Initialize with slightly higher weight on visual (0.6) vs metadata (0.4)
            self.visual_weight = nn.Parameter(torch.tensor(0.8))
            self.metadata_weight = nn.Parameter(torch.tensor(0.2))
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")

    def fuse(self, visual_embedding, metadata_embedding):
        """
        Fuse visual and metadata embeddings

        Args:
            visual_embedding (torch.Tensor): Visual embedding from image encoder
            metadata_embedding (torch.Tensor): Metadata embedding from metadata encoder

        Returns:
            fused_embedding (torch.Tensor): Fused embedding
        """
        # Move to the right device
        visual_embedding = visual_embedding.to(self.device)
        metadata_embedding = metadata_embedding.to(self.device)

        # Ensure embeddings have the same batch dimension
        if visual_embedding.shape[0] != metadata_embedding.shape[0]:
            raise ValueError(f"Batch dimensions don't match: {visual_embedding.shape[0]} vs {metadata_embedding.shape[0]}")

        # Apply fusion based on selected method
        if self.fusion_method == "concat":
            # Concatenate along the feature dimension
            combined = torch.cat([visual_embedding, metadata_embedding], dim=1)
            # Project to output dimension
            with torch.no_grad():
                fused = self.projection(combined)
            return fused

        elif self.fusion_method == "weighted":
            # Need to ensure metadata_embedding has same dim as visual_embedding for weighted sum
            if self.visual_dim != self.metadata_dim:
                # Simple projection via a repeat and slice approach
                if self.metadata_dim < self.visual_dim:
                    # Repeat metadata embedding to match visual dim
                    repeat_factor = self.visual_dim // self.metadata_dim + 1
                    expanded = metadata_embedding.repeat(1, repeat_factor)
                    metadata_embedding = expanded[:, :self.visual_dim]
                else:
                    # Truncate metadata embedding to match visual dim
                    metadata_embedding = metadata_embedding[:, :self.visual_dim]

            # Normalize weights to sum to 1
            sum_weights = self.visual_weight + self.metadata_weight
            norm_visual_weight = self.visual_weight / sum_weights
            norm_metadata_weight = self.metadata_weight / sum_weights

            # Weighted sum
            fused = norm_visual_weight * visual_embedding + norm_metadata_weight * metadata_embedding

            return fused
