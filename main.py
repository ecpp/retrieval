#!/usr/bin/env python
import os
import sys
import argparse
from src.retrieval_system import RetrievalSystem

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='CAD Part Retrieval System')
    
    # Define commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest data from STEP output directories')
    ingest_parser.add_argument('--dataset_dir', type=str, required=True, help='Directory containing STEP output directories')
    
    # Build index command
    build_parser = subparsers.add_parser('build', help='Build vector index from images')
    build_parser.add_argument('--image_dir', type=str, help='Directory containing images to index')
    
    # Retrieve command
    retrieve_parser = subparsers.add_parser('retrieve', help='Retrieve similar parts')
    retrieve_parser.add_argument('--query', type=str, required=True, help='Path to query image')
    retrieve_parser.add_argument('--k', type=int, default=10, help='Number of results to retrieve')
    retrieve_parser.add_argument('--visualize', action='store_true', help='Visualize the results')
    retrieve_parser.add_argument('--rotation-invariant', action='store_true', help='Enable rotation-invariant search')
    retrieve_parser.add_argument('--num-rotations', type=int, default=8, help='Number of rotations to try')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate retrieval system')
    eval_parser.add_argument('--query_dir', type=str, help='Directory containing query images')
    eval_parser.add_argument('--ground_truth', type=str, help='Path to ground truth JSON file')
    
    # Info command
    subparsers.add_parser('info', help='Display system information')
    
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_args()
    
    # Initialize the retrieval system
    retrieval_system = RetrievalSystem()
    
    if args.command == 'ingest':
        print(f"Ingesting data from {args.dataset_dir}")
        retrieval_system.ingest_data(args.dataset_dir)
    
    elif args.command == 'build':
        print("Building vector index")
        retrieval_system.build_index(args.image_dir)
    
    elif args.command == 'retrieve':
        print(f"Retrieving similar parts to {args.query}")
        if args.rotation_invariant:
            print(f"Using rotation-invariant search with {args.num_rotations} rotations")
        
        if not os.path.exists(args.query):
            print(f"Error: Query image not found: {args.query}")
            return
        
        results = retrieval_system.retrieve_similar(
            args.query, 
            k=args.k, 
            rotation_invariant=args.rotation_invariant, 
            num_rotations=args.num_rotations
        )
        
        # Print results
        print(f"Top {len(results['paths'])} results:")
        for i, (path, distance, info) in enumerate(zip(results["paths"], results["distances"], results.get("part_info", [None] * len(results["paths"])))):
            # Convert distance to similarity score (0-100%), where higher is better
            similarity = 100 * (1 / (1 + distance))
            print(f"{i+1}. {os.path.basename(path)} (similarity: {similarity:.2f}%)")
            # Print part information if available
            if info:
                parent_step = info.get("parent_step", "unknown")
                part_name = info.get("part_name", "unknown")
                print(f"   STEP File: {parent_step}")
                print(f"   Part Name: {part_name}")
            # Also print distance for reference
            print(f"   (distance: {distance:.4f})")
        
        # Visualize if requested
        if args.visualize:
            retrieval_system.visualize_results(args.query, results)
    
    elif args.command == 'evaluate':
        print("Evaluating retrieval system")
        retrieval_system.evaluate(args.query_dir, args.ground_truth)
    
    elif args.command == 'info':
        info = retrieval_system.get_system_info()
        print("System Information:")
        print(f"Model: {info['model']['name']}")
        print(f"Embedding dimension: {info['model']['embedding_dim']}")
        print(f"Device: {info['model']['device']}")
        print(f"Index: {info['index']['index_type']}")
        print(f"Vectors in index: {info['index']['num_vectors']}")
    
    else:
        print("Please specify a command. Run with --help for options.")

if __name__ == "__main__":
    main()
