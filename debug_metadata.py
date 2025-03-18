#!/usr/bin/env python
import os
import sys

# First, print Python version and path
print(f"Python version: {sys.version}")
print(f"sys.path: {sys.path}")

# Try importing torch
print("\nTrying to import torch...")
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"Error importing torch: {e}")

# Try importing dependencies for metadata_encoder.py
print("\nTrying to import dependencies for metadata_encoder.py...")
try:
    import torch.nn as nn
    import numpy as np
    import json
    print("All dependencies for metadata_encoder.py successfully imported")
except Exception as e:
    print(f"Error importing dependencies: {e}")

# Try importing the metadata modules directly
print("\nTrying to import metadata_encoder...")
try:
    sys.path.append(os.path.abspath("src"))
    from src.metadata_encoder import MetadataEncoder
    print("Successfully imported MetadataEncoder")
    
    # Test creating a MetadataEncoder
    encoder = MetadataEncoder()
    print(f"Created MetadataEncoder with output_dim={encoder.output_dim}")
    print(f"Input features dimension: {encoder.get_input_dim()}")
    
except Exception as e:
    print(f"Error importing or initializing MetadataEncoder: {e}")

print("\nTrying to import fusion_module...")
try:
    from src.fusion_module import FusionModule
    print("Successfully imported FusionModule")
    
    # Test creating a FusionModule
    fusion = FusionModule(visual_dim=768, metadata_dim=256, output_dim=768)
    print(f"Created FusionModule with fusion_method={fusion.fusion_method}")
    
except Exception as e:
    print(f"Error importing or initializing FusionModule: {e}")

# Try the exact import syntax used in retrieval_system.py
print("\nTrying the exact import syntax from retrieval_system.py...")
os.chdir("src")  # Change to src directory
try:
    from metadata_encoder import MetadataEncoder
    from fusion_module import FusionModule
    print("Direct imports succeeded")
except Exception as e:
    print(f"Error with direct imports: {e}")

print("\nDebugging complete")
