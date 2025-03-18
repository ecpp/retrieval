#!/usr/bin/env python
import os
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from src.metadata_encoder import MetadataEncoder, BomDataset

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test Metadata Autoencoder')
    
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to configuration file')
    parser.add_argument('--model', type=str, help='Path to trained model (overrides config)')
    parser.add_argument('--bom_dir', type=str, help='Directory containing BOM files (overrides config)')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations of the latent space')
    parser.add_argument('--save_dir', type=str, default='data/output/evaluation/autoencoder', help='Directory to save visualizations')
    
    return parser.parse_args()

def visualize_latent_space(encoder, dataset, save_dir):
    """
    Visualize the latent space embeddings
    
    Args:
        encoder (MetadataEncoder): Trained encoder
        dataset (BomDataset): Dataset of metadata samples
        save_dir (str): Directory to save visualizations
    """
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    
    # Extract all latent embeddings
    all_embeddings = []
    for batch in dataloader:
        batch = batch.to(encoder.device)
        with torch.no_grad():
            embeddings = encoder.encoder(batch)
        all_embeddings.append(embeddings.cpu().numpy())
    
    # Concatenate all embeddings
    embeddings = np.concatenate(all_embeddings, axis=0)
    
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Visualize with PCA
    if embeddings.shape[1] > 2:
        print("Applying PCA to reduce dimensions to 2...")
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)
        plt.title('PCA Visualization of Latent Space')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.grid(True)
        pca_path = os.path.join(save_dir, 'latent_space_pca.png')
        plt.savefig(pca_path)
        plt.close()
        print(f"PCA visualization saved to {pca_path}")
        
        # Display explained variance
        explained_var = pca.explained_variance_ratio_
        plt.figure(figsize=(10, 5))
        plt.bar(range(1, 3), explained_var)
        plt.title('Explained Variance by Principal Components')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.xticks(range(1, 3))
        plt.grid(True, axis='y')
        var_path = os.path.join(save_dir, 'pca_explained_variance.png')
        plt.savefig(var_path)
        plt.close()
        print(f"Explained variance plot saved to {var_path}")
    
    # Visualize with t-SNE
    if len(embeddings) > 10:  # t-SNE needs a reasonable number of points
        print("Applying t-SNE to visualize latent space...")
        tsne = TSNE(n_components=2, perplexity=min(30, len(embeddings)-1), random_state=42)
        embeddings_tsne = tsne.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], alpha=0.5)
        plt.title('t-SNE Visualization of Latent Space')
        plt.grid(True)
        tsne_path = os.path.join(save_dir, 'latent_space_tsne.png')
        plt.savefig(tsne_path)
        plt.close()
        print(f"t-SNE visualization saved to {tsne_path}")

def analyze_reconstruction(encoder, dataset, save_dir, num_samples=100):
    """
    Analyze reconstruction accuracy in detail
    
    Args:
        encoder (MetadataEncoder): Trained encoder
        dataset (BomDataset): Dataset of metadata samples
        save_dir (str): Directory to save results
        num_samples (int): Number of samples to analyze
    """
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Limit the number of samples
    num_samples = min(num_samples, len(dataset))
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    # Collect reconstruction errors for each feature
    feature_errors = []
    
    for idx in indices:
        sample = dataset[idx].unsqueeze(0).to(encoder.device)
        with torch.no_grad():
            reconstructed = encoder.reconstruct(sample)
        
        # Calculate absolute error for each feature
        error = torch.abs(sample - reconstructed).squeeze(0).cpu().numpy()
        feature_errors.append(error)
    
    # Convert to numpy array
    feature_errors = np.array(feature_errors)
    
    # Calculate statistics
    mean_error = np.mean(feature_errors, axis=0)
    median_error = np.median(feature_errors, axis=0)
    max_error = np.max(feature_errors, axis=0)
    
    # Plot error statistics by feature
    features = np.arange(len(mean_error))
    
    plt.figure(figsize=(14, 8))
    plt.bar(features, mean_error, alpha=0.7, label='Mean Error')
    plt.plot(features, median_error, 'ro-', alpha=0.7, label='Median Error')
    plt.plot(features, max_error, 'go-', alpha=0.5, label='Max Error')
    plt.xlabel('Feature Index')
    plt.ylabel('Reconstruction Error')
    plt.title('Reconstruction Error by Feature')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    error_path = os.path.join(save_dir, 'reconstruction_error_by_feature.png')
    plt.savefig(error_path)
    plt.close()
    print(f"Feature error analysis saved to {error_path}")
    
    # Overall error distribution
    plt.figure(figsize=(10, 6))
    plt.hist(feature_errors.flatten(), bins=50, alpha=0.7)
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Reconstruction Errors')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    dist_path = os.path.join(save_dir, 'error_distribution.png')
    plt.savefig(dist_path)
    plt.close()
    print(f"Error distribution plot saved to {dist_path}")
    
    # Identify worst reconstructed features
    worst_features = np.argsort(mean_error)[::-1][:10]  # Top 10 worst features
    
    print("\nTop 10 worst reconstructed features:")
    for i, feature_idx in enumerate(worst_features):
        print(f"{i+1}. Feature {feature_idx}: Mean Error {mean_error[feature_idx]:.4f}")
    
    # Save results
    with open(os.path.join(save_dir, 'reconstruction_analysis.txt'), 'w') as f:
        f.write("Autoencoder Reconstruction Analysis\n")
        f.write("=================================\n\n")
        f.write(f"Number of samples analyzed: {num_samples}\n")
        f.write(f"Average reconstruction error: {np.mean(feature_errors):.6f}\n")
        f.write(f"Median reconstruction error: {np.median(feature_errors):.6f}\n")
        f.write(f"Maximum reconstruction error: {np.max(feature_errors):.6f}\n\n")
        
        f.write("Top 10 worst reconstructed features:\n")
        for i, feature_idx in enumerate(worst_features):
            f.write(f"{i+1}. Feature {feature_idx}: Mean Error {mean_error[feature_idx]:.4f}\n")

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
    model_path = args.model or metadata_config.get('model_path', 'models/metadata_autoencoder.pt')
    save_dir = args.save_dir
    
    # Get latent dimension and hidden dimensions
    latent_dim = metadata_config.get('embedding_dim', 256)
    hidden_dims = metadata_config.get('hidden_dims', [512, 384])
    
    # Initialize the encoder
    print(f"Initializing metadata encoder with latent_dim={latent_dim}, hidden_dims={hidden_dims}")
    encoder = MetadataEncoder(output_dim=latent_dim, hidden_dims=hidden_dims)
    
    # Load the model
    if os.path.exists(model_path):
        print(f"Loading trained model from {model_path}")
        encoder.load_trained_model(model_path)
    else:
        print(f"Error: Model file not found at {model_path}")
        return
    
    # Create the dataset
    print(f"Loading BOM data from {bom_dir}")
    dataset = BomDataset(bom_dir, encoder)
    
    if len(dataset) == 0:
        print(f"No BOM data found in {bom_dir}. Please check the directory path.")
        return
    
    print(f"Loaded {len(dataset)} samples for testing")
    
    # Test reconstruction on a few random samples
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True)
    batch = next(iter(dataloader)).to(encoder.device)
    
    with torch.no_grad():
        reconstructed = encoder.reconstruct(batch)
    
    # Calculate reconstruction loss
    loss = torch.nn.functional.mse_loss(reconstructed, batch)
    print(f"Test reconstruction loss: {loss.item():.6f}")
    
    # Analyze in detail
    analyze_reconstruction(encoder, dataset, save_dir)
    
    # Visualize the latent space if requested
    if args.visualize:
        print("Visualizing latent space...")
        visualize_latent_space(encoder, dataset, save_dir)

if __name__ == "__main__":
    main()
