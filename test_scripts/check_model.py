#!/usr/bin/env python
"""
Script to check which model is being used by the retrieval system.
"""
from src.retrieval_system import RetrievalSystem
from src.image_encoder import ImageEncoder
import torch

def check_model():
    """Check and display model information"""
    print("\n=== Checking Retrieval System Model ===\n")
    
    # Initialize the system
    try:
        retrieval_system = RetrievalSystem()
        info = retrieval_system.get_system_info()
        
        print(f"Model name: {info['model']['name']}")
        print(f"Embedding dimension: {info['model']['embedding_dim']}")
        print(f"Image size: {info['model']['image_size']}")
        print(f"Device: {info['model']['device']}")
        
        # Check which model is actually loaded in the image encoder
        encoder = retrieval_system.image_encoder
        print(f"\nActual model type: {type(encoder.model).__name__}")
        
        # Check if CUDA is available
        print(f"\nPyTorch details:")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Current device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        
        # Check transformers version if available
        try:
            import transformers
            print(f"\nTransformers version: {transformers.__version__}")
            
            # Check if DINOv2 is available
            try:
                from transformers import Dinov2Model
                print("DINOv2 model is available in your transformers installation")
            except ImportError:
                print("DINOv2 model is NOT available in your transformers installation")
        except ImportError:
            print("\nTransformers library not found")
        
    except Exception as e:
        print(f"Error checking model: {e}")

if __name__ == "__main__":
    check_model()
