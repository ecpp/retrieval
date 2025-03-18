#!/usr/bin/env python
import argparse
import os
import yaml
import matplotlib.pyplot as plt
import numpy as np
import torch
from src.metadata_encoder import MetadataEncoder, BomDataset
from torch.utils.data import DataLoader

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Metadata Autoencoder')
    
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to configuration file')
    parser.add_argument('--bom_dir', type=str, help='Directory containing BOM files (overrides config)')
    parser.add_argument('--batch_size', type=int, help='Batch size for training (overrides config)')
    parser.add_argument('--epochs', type=int, help='Number of training epochs (overrides config)')
    parser.add_argument('--lr', type=float, help='Learning rate (overrides config)')
    parser.add_argument('--hidden_dims', type=str, help='Hidden dimensions, comma-separated (overrides config)')
    parser.add_argument('--latent_dim', type=int, help='Dimension of latent space (overrides config)')
    parser.add_argument('--output', type=str, help='Output model path (overrides config)')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the trained model')
    
    return parser.parse_args()

def plot_loss(history, output_path):
    """Plot training loss history"""
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'])
    plt.title('Autoencoder Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    print(f"Loss plot saved to {output_path}")

def evaluate_autoencoder(encoder, dataset, num_samples=5):
    """Evaluate the autoencoder by visualizing reconstructions"""
    # Create a dataloader with batch size 1 for evaluation
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Create output directory
    os.makedirs('data/output/evaluation/autoencoder', exist_ok=True)
    
    # Initialize the figure
    plt.figure(figsize=(15, 5 * num_samples))
    
    # Process a few samples
    for i, sample in enumerate(dataloader):
        if i >= num_samples:
            break
        
        # Move to appropriate device
        sample = sample.to(encoder.device)
        
        # Get reconstruction
        with torch.no_grad():
            reconstructed = encoder.reconstruct(sample)
        
        # Convert to numpy for plotting
        original = sample.cpu().numpy().flatten()
        reconstructed = reconstructed.cpu().numpy().flatten()
        
        # Plot original vs reconstructed
        plt.subplot(num_samples, 2, 2*i + 1)
        plt.bar(range(len(original)), original)
        plt.title(f'Original Sample {i+1}')
        plt.ylabel('Feature Value')
        
        plt.subplot(num_samples, 2, 2*i + 2)
        plt.bar(range(len(reconstructed)), reconstructed)
        plt.title(f'Reconstructed Sample {i+1}')
        
    plt.tight_layout()
    output_path = 'data/output/evaluation/autoencoder/reconstruction_comparison.png'
    plt.savefig(output_path)
    plt.close()
    print(f"Reconstruction comparison saved to {output_path}")

    # Also create a feature-wise error plot
    plt.figure(figsize=(12, 6))
    
    # Calculate mean reconstruction error for each feature
    all_errors = []
    
    # Process all samples
    with torch.no_grad():
        for sample in DataLoader(dataset, batch_size=len(dataset)):
            sample = sample.to(encoder.device)
            reconstructed = encoder.reconstruct(sample)
            error = torch.abs(sample - reconstructed).mean(dim=0).cpu().numpy()
            all_errors.append(error)
    
    # Average errors if we processed in batches
    mean_error = np.mean(all_errors, axis=0)
    
    # Plot feature-wise error
    plt.bar(range(len(mean_error)), mean_error)
    plt.title('Feature-wise Reconstruction Error')
    plt.xlabel('Feature Index')
    plt.ylabel('Mean Absolute Error')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    error_path = 'data/output/evaluation/autoencoder/feature_error.png'
    plt.savefig(error_path)
    plt.close()
    print(f"Feature error plot saved to {error_path}")

def main():
    """Main entry point"""
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get metadata settings
    metadata_config = config.get('metadata', {})
    
    # Override config with command line arguments
    bom_dir = args.bom_dir or metadata_config.get('bom_dir', 'data/output/bom')
    batch_size = args.batch_size or metadata_config.get('batch_size', 32)
    epochs = args.epochs or metadata_config.get('epochs', 50)
    lr = args.lr or metadata_config.get('learning_rate', 1e-4)
    output_path = args.output or metadata_config.get('model_path', 'models/metadata_autoencoder.pt')
    
    # Parse hidden dimensions if provided
    if args.hidden_dims:
        hidden_dims = [int(dim) for dim in args.hidden_dims.split(',')]
    else:
        hidden_dims = metadata_config.get('hidden_dims', [512, 384])
    
    # Get latent dimension
    latent_dim = args.latent_dim or metadata_config.get('embedding_dim', 256)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize the autoencoder
    print(f"Initializing metadata autoencoder with latent dim={latent_dim}, hidden_dims={hidden_dims}")
    encoder = MetadataEncoder(output_dim=latent_dim, hidden_dims=hidden_dims)
    
    # Create the dataset
    dataset = BomDataset(bom_dir, encoder)
    
    if len(dataset) == 0:
        print(f"No BOM data found in {bom_dir}. Please check the directory path.")
        return
    
    # Train the autoencoder
    print(f"Training autoencoder with {len(dataset)} samples, batch_size={batch_size}, epochs={epochs}, lr={lr}")
    history = encoder.train_autoencoder(bom_dir, batch_size=batch_size, epochs=epochs, lr=lr, save_path=output_path)
    
    # Plot training loss
    os.makedirs('data/output/evaluation/autoencoder', exist_ok=True)
    plot_loss(history, 'data/output/evaluation/autoencoder/training_loss.png')
    
    # Evaluate if requested
    if args.evaluate:
        print("Evaluating autoencoder performance...")
        evaluate_autoencoder(encoder, dataset)

if __name__ == "__main__":
    main()
