#!/usr/bin/env python
"""
Script to check and visualize rotations of a CAD part image.
This helps debug and understand the rotation-invariant search.
"""
import os
import argparse
from PIL import Image
from src.rotational_utils import generate_rotations, visualize_rotations

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Check rotations of a CAD part image')
    parser.add_argument('--image', type=str, required=True, help='Path to the image file')
    parser.add_argument('--num-rotations', type=int, default=8, help='Number of rotations to generate')
    parser.add_argument('--output', type=str, help='Path to save visualization (optional)')
    args = parser.parse_args()
    
    # Check if the image exists
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        return
    
    # Load the image
    try:
        image = Image.open(args.image).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Generate rotations
    print(f"Generating {args.num_rotations} rotations...")
    rotated_images = generate_rotations(image, args.num_rotations)
    
    # Visualize rotations
    print("Visualizing rotations...")
    visualize_rotations(image, rotated_images, args.output)
    
    if args.output:
        print(f"Visualization saved to {args.output}")
    
    print("Done!")

if __name__ == "__main__":
    main()
