#!/usr/bin/env python
"""
Automated evaluation script for part retrieval.
Selects random part images from the dataset and runs queries,
saving the results for manual inspection.
"""
import os
import sys
import json
import random
import argparse
from datetime import datetime
from src.retrieval_system import RetrievalSystem

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Automated evaluation for part retrieval'
    )
    parser.add_argument('--num-queries', type=int, default=5,
                        help='Number of random queries to run')
    parser.add_argument('--k', type=int, default=10,
                        help='Number of results to retrieve per query')
    parser.add_argument('--rotation-invariant', action='store_true',
                        help='Enable rotation-invariant search')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--output-dir', type=str, default='data/evaluation/part_retrieval',
                        help='Directory to store evaluation results')

    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_args()

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)

    # Create timestamp for this evaluation run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"Starting part retrieval evaluation with {args.num_queries} random queries")
    print(f"Results will be saved to {run_dir}")

    # Initialize retrieval system
    retrieval_system = RetrievalSystem()

    # Check for the image directory
    image_dir = os.path.join(retrieval_system.config["data"]["output_dir"], "images")
    if not os.path.exists(image_dir):
        print(f"Error: Image directory {image_dir} does not exist!")
        return 1

    # Get list of available images
    available_images = [
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    if not available_images:
        print("Error: No images found in the dataset!")
        return 1

    print(f"Found {len(available_images)} candidate images for queries")

    # Check if we have enough images
    if len(available_images) < args.num_queries:
        print(f"Warning: Requested {args.num_queries} queries but only {len(available_images)} images available")
        print(f"Proceeding with {len(available_images)} queries instead")
        args.num_queries = len(available_images)

    # Randomly select query images
    query_images = random.sample(available_images, args.num_queries)

    # Create a summary file to track evaluation
    summary = {
        "timestamp": timestamp,
        "num_queries": args.num_queries,
        "k": args.k,
        "rotation_invariant": args.rotation_invariant,
        "results": []
    }

    # Run queries and save results
    for i, query_path in enumerate(query_images):
        print(f"Processing query {i+1}/{args.num_queries}: {os.path.basename(query_path)}")

        # Get part info if available
        part_info = retrieval_system.extract_part_info(query_path)

        try:
            # Perform retrieval
            results = retrieval_system.retrieve_similar(
                query_path,
                k=args.k,
                rotation_invariant=args.rotation_invariant
            )

            # Save the result visualization
            output_path = os.path.join(run_dir, f"query_{i+1}_results.png")
            try:
                retrieval_system.visualize_results(query_path, results, output_path)
                print(f"  Saved visualization to {output_path}")
            except Exception as e:
                print(f"  Error visualizing results: {e}")

            # Calculate basic metrics
            # For non-repeated results
            unique_results = len(set(results["paths"]))
            retrieval_time = results.get("retrieval_time", 0)

            # Calculate similarity stats
            similarities = []
            for distance in results["distances"]:
                # Convert distance to similarity score (0-100%), where higher is better
                similarity = 100 * (1 / (1 + distance))
                similarities.append(similarity)

            avg_similarity = sum(similarities) / len(similarities) if similarities else 0

            # Add to summary
            summary["results"].append({
                "query_idx": i+1,
                "query_path": query_path,
                "query_filename": os.path.basename(query_path),
                "part_info": part_info,
                "unique_results": unique_results,
                "retrieval_time": retrieval_time,
                "avg_similarity": avg_similarity,
                "visualization": output_path,
                "top_results": [
                    {
                        "path": path,
                        "distance": distance,
                        "similarity": 100 * (1 / (1 + distance)),
                        "part_info": info
                    }
                    for path, distance, info in zip(
                        results["paths"],
                        results["distances"],
                        results.get("part_info", [None] * len(results["paths"]))
                    )
                ][:args.k]
            })

        except Exception as e:
            print(f"Error processing query {os.path.basename(query_path)}: {e}")

    # Save the summary
    summary_path = os.path.join(run_dir, "evaluation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Evaluation complete! Summary saved to {summary_path}")
    print(f"Average retrieval time: {sum(r['retrieval_time'] for r in summary['results']) / len(summary['results']):.4f} seconds")
    print(f"Average similarity score: {sum(r['avg_similarity'] for r in summary['results']) / len(summary['results']):.2f}%")

    return 0

if __name__ == "__main__":
    sys.exit(main())