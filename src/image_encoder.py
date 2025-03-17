import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from transformers import ViTModel, AutoImageProcessor
from PIL import Image

# Check if DINO v2 is available in the transformers library
try:
    from transformers import Dinov2Model
    DINOV2_AVAILABLE = True
except ImportError:
    # Fall back to regular ViT if DINOv2 is not available
    DINOV2_AVAILABLE = False

class ImageEncoder:
    """
    Handles encoding of CAD part images into embeddings using various models
    """
    def __init__(self, model_name='dinov2', pretrained=True, embedding_dim=768, image_size=224):
        """
        Initialize the image encoder with specified model
        
        Args:
            model_name (str): Name of the model to use ('dinov2', 'vit', 'resnet50')
            pretrained (bool): Whether to use pretrained weights
            embedding_dim (int): Dimension of the output embedding
            image_size (int): Size to resize images to
        """
        self.model_name = model_name
        self.image_size = image_size
        self.embedding_dim = embedding_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set up model-specific transforms
        if model_name == 'dinov2':
            if DINOV2_AVAILABLE:
                self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
                self.model = Dinov2Model.from_pretrained("facebook/dinov2-base")
                self.transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.processor.image_mean, std=self.processor.image_std),
                ])
            else:
                print("DINOv2 not available in your transformers version. Falling back to ViT.")
                # Fall back to ViT
                self.processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
                self.model = ViTModel.from_pretrained("google/vit-base-patch16-224")
                self.transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.processor.image_mean, std=self.processor.image_std),
                ])
                self.model_name = 'vit'  # Update model name to reflect actual model used
        elif model_name == 'vit':
            self.processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
            self.model = ViTModel.from_pretrained("google/vit-base-patch16-224")
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.processor.image_mean, std=self.processor.image_std),
            ])
        elif model_name == 'resnet50':
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            # Remove the classification head
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        self.model.to(self.device)
        self.model.eval()
        
    def encode_image(self, image_path):
        """
        Encode a single image into an embedding vector
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            embedding (torch.Tensor): Embedding vector
        """
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                if self.model_name == 'dinov2' or self.model_name == 'vit':
                    outputs = self.model(image_tensor)
                    # Use CLS token as the embedding
                    embedding = outputs.last_hidden_state[:, 0].cpu()
                elif self.model_name == 'resnet50':
                    embedding = self.model(image_tensor).squeeze().cpu()
                
            return embedding
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None
    
    def encode_batch(self, image_paths, batch_size=32):
        """
        Encode a batch of images into embedding vectors
        
        Args:
            image_paths (list): List of paths to image files
            batch_size (int): Batch size for processing
            
        Returns:
            embeddings (torch.Tensor): Tensor of embedding vectors
        """
        all_embeddings = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            
            for path in batch_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    image_tensor = self.transform(image)
                    batch_images.append(image_tensor)
                except Exception as e:
                    print(f"Error loading image {path}: {e}")
                    # Add a tensor of zeros as a placeholder
                    batch_images.append(torch.zeros(3, self.image_size, self.image_size))
            
            # Stack the batch
            batch_tensor = torch.stack(batch_images).to(self.device)
            
            with torch.no_grad():
                if self.model_name == 'dinov2' or self.model_name == 'vit':
                    outputs = self.model(batch_tensor)
                    # Use CLS token as the embedding
                    batch_embeddings = outputs.last_hidden_state[:, 0].cpu()
                elif self.model_name == 'resnet50':
                    batch_embeddings = self.model(batch_tensor).squeeze().cpu()
            
            all_embeddings.append(batch_embeddings)
        
        return torch.cat(all_embeddings, dim=0)
