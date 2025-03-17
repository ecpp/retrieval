#!/usr/bin/env python
"""
Test script for the CAD part retrieval system.
This demonstrates how to use the retrieval system in code.
"""
import os
import sys
from src.retrieval_system import RetrievalSystem

def test_system():
    """Test the retrieval system with a small example"""
    # Initialize the system
    retrieval_system = RetrievalSystem()
    
    # Print system info
    print("=== System Information ===")
    info = retrieval_system.get_system_info()
    print(f"Model: {info['model']['name']}")
    print(f"Embedding dimension: {info['model']['embedding_dim']}")
    print(f"Device: {info['model']['device']}")
    
    # Check if we have an index
    if info['index']['num_vectors'] == 0:
        print("\nNo vectors in the index. You need to ingest data and build the index first.")
        print("Run the following commands:")
        print("  python main.py ingest --dataset_dir /path/to/step/outputs")
        print("  python main.py build")
    else:
        print(f"\nIndex contains {info['index']['num_vectors']} vectors.")
        
        # If we have a query image, test retrieval
        sample_query = os.path.join("data", "output", "evaluation", "queries", "sample_query.png")
        if os.path.exists(sample_query):
            print(f"\n=== Testing retrieval with sample query {sample_query} ===")
            results = retrieval_system.retrieve_similar(sample_query, k=5)
            
            print("Top 5 results:")
            for i, (path, distance) in enumerate(zip(results["paths"], results["distances"])):
                print(f"{i+1}. {os.path.basename(path)} (distance: {distance:.4f})")
            
            # Visualize results
            retrieval_system.visualize_results(sample_query, results)
        else:
            print("\nNo sample query image found. Add a query image to test retrieval.")

if __name__ == "__main__":
    test_system()
