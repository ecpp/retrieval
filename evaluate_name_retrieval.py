#!/usr/bin/env python
"""
Automated evaluation script for part name retrieval.
Tests retrieval using sample part names and saves the results for manual inspection.
"""
import os
import sys
import json
import random
import argparse
from datetime import datetime
from src.retrieval_system import RetrievalSystem
from PIL import Image, ImageDraw

# Sample part name categories for testing
# These are example categories - you might need to adjust these based on your actual data
SAMPLE_NAME_CATEGORIES = {
    "common_parts": [
        "screw", "bolt", "nut", "washer", "pin", "gear", "shaft", "bearing"
    ],
    "materials": [
        "steel", "aluminum", "plastic", "rubber", "metal", "brass", "copper",
        "nylon", "titanium", "cast", "forged"
    ],
    "custom": [
        # Add your own specific part names here that you want to test
    ]
}

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Automated evaluation for part name retrieval'
    )
    parser.add_argument('--num-queries', type=int, default=10,
                        help='Number of random part name queries to run')
    parser.add_argument('--k', type=int, default=10,
                        help='Number of results to retrieve per query')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Threshold for part name matching (0-1)')
    parser.add_argument('--rotation-invariant', action='store_true',
                        help='Enable rotation-invariant search')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--output-dir', type=str, default='data/evaluation/name_retrieval',
                        help='Directory to store evaluation results')
    parser.add_argument('--custom-queries', nargs='+', type=str, default=[],
                        help='Custom part name queries to use (instead of random selection)')

    return parser.parse_args()

def extract_part_names_from_dataset(retrieval_system, max_names=100):
    """Extract actual part names from the dataset to use as queries"""
    # Check if vector database is initialized and has metadata
    if not retrieval_system.vector_db:
        print("Vector database not initialized")
        return []

    all_part_names = set()

    try:
        # Try to access metadata
        metadata = retrieval_system.vector_db.metadata

        if not metadata:
            print("No metadata available")
            return []

        # Extract part names from metadata
        for item in metadata:
            if isinstance(item, dict) and "part_name" in item:
                name = item["part_name"]
                if name and name.lower() not in ("unknown", "none"):
                    all_part_names.add(name)

                    # If we have enough names, stop
                    if len(all_part_names) >= max_names:
                        break
    except Exception as e:
        print(f"Error retrieving part names: {e}")
        return []

    return list(all_part_names)

def generate_queries(num_queries, dataset_part_names, seed=None):
    """Generate a list of part name queries for testing"""
    if seed is not None:
        random.seed(seed)

    queries = []

    # Try to get some from the actual dataset
    if dataset_part_names:
        num_dataset_queries = min(num_queries // 2, len(dataset_part_names))
        queries.extend(random.sample(dataset_part_names, num_dataset_queries))

    # Add some from our sample categories
    remaining = num_queries - len(queries)
    if remaining > 0:
        # Flatten the categories into a single list, excluding empty categories
        all_samples = []
        for category, names in SAMPLE_NAME_CATEGORIES.items():
            all_samples.extend(names)

        if all_samples:
            # Select random samples, possibly with replacement if we don't have enough
            if len(all_samples) >= remaining:
                queries.extend(random.sample(all_samples, remaining))
            else:
                queries.extend(all_samples)
                # Add random combinations if needed
                for _ in range(remaining - len(all_samples)):
                    # Combine two random terms
                    term1 = random.choice(all_samples)
                    term2 = random.choice(all_samples)
                    if term1 != term2:
                        queries.append(f"{term1} {term2}")
                    else:
                        queries.append(term1)

    # Ensure we don't have duplicates
    unique_queries = list(dict.fromkeys(queries))

    # If we still need more, just repeat some
    while len(unique_queries) < num_queries:
        unique_queries.append(random.choice(unique_queries))

    return unique_queries[:num_queries]

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

    print(f"Starting part name retrieval evaluation")
    print(f"Results will be saved to {run_dir}")

    # Initialize retrieval system
    retrieval_system = RetrievalSystem()

    # Determine queries to run
    queries = args.custom_queries

    if not queries:
        # Get some part names from the dataset
        dataset_part_names = extract_part_names_from_dataset(retrieval_system)
        print(f"Found {len(dataset_part_names)} part names in the dataset")

        # Generate random queries
        queries = generate_queries(args.num_queries, dataset_part_names, args.seed)

    print(f"Running {len(queries)} part name queries")

    # Create a summary file to track evaluation
    summary = {
        "timestamp": timestamp,
        "num_queries": len(queries),
        "k": args.k,
        "threshold": args.threshold,
        "rotation_invariant": args.rotation_invariant,
        "queries": queries,
        "results": []
    }

    # Run queries and save results
    for i, query in enumerate(queries):
        print(f"Processing query {i+1}/{len(queries)}: '{query}'")

        try:
            # Perform retrieval by part name
            results = retrieval_system.retrieve_by_part_name(
                query,
                k=args.k,
                rotation_invariant=args.rotation_invariant,
                threshold=args.threshold
            )

            # If there are no results, try lowering the threshold
            if not results["paths"] and args.threshold > 0.3:
                print(f"  No results found with threshold {args.threshold}, trying with lower threshold 0.3")
                results = retrieval_system.retrieve_by_part_name(
                    query,
                    k=args.k,
                    rotation_invariant=args.rotation_invariant,
                    threshold=0.3
                )

            # Save the result visualization
            output_path = os.path.join(run_dir, f"query_{i+1}_results.png")

            # Create a simple text-based image for the query instead of a text file
            # Create blank image with text
            img = Image.new('RGB', (300, 100), color=(255, 255, 255))
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), f"Query: {query}", fill=(0, 0, 0))

            # If successful, set query_img_path to the matched image path for visualization
            if results and "paths" in results and len(results["paths"]) > 0:
                if "query_match" in results and "path" in results["query_match"]:
                    query_img_path = results["query_match"]["path"]
                    print(f"Using matched part image for visualization: {query_img_path}")

            # Use visualize_results with the image
            try:
                retrieval_system.visualize_results(query_img_path, results, output_path)
                print(f"  Found {len(results['paths'])} results, saved visualization to {output_path}")
            except Exception as e:
                print(f"  Error visualizing results: {e}")

            # Calculate basic metrics
            unique_results = len(set(results["paths"]))
            retrieval_time = results.get("retrieval_time", 0)

            # Calculate name match scores
            name_scores = results.get("name_scores", [0] * len(results["paths"]))
            avg_name_score = sum(name_scores) / len(name_scores) if name_scores else 0

            # Add to summary
            summary["results"].append({
                "query_idx": i+1,
                "query": query,
                "unique_results": unique_results,
                "retrieval_time": retrieval_time,
                "avg_name_score": avg_name_score,
                "visualization": output_path,
                "top_results": [
                    {
                        "path": path,
                        "distance": distance,
                        "name_score": score,
                        "part_info": info
                    }
                    for path, distance, score, info in zip(
                        results["paths"],
                        results.get("distances", [0] * len(results["paths"])),
                        results.get("name_scores", [0] * len(results["paths"])),
                        results.get("part_info", [None] * len(results["paths"]))
                    )
                ][:args.k]
            })

        except Exception as e:
            print(f"Error processing query '{query}': {e}")

    # Save the summary
    summary_path = os.path.join(run_dir, "evaluation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Evaluation complete! Summary saved to {summary_path}")
    if summary["results"]:
        print(f"Average retrieval time: {sum(r['retrieval_time'] for r in summary['results']) / len(summary['results']):.4f} seconds")
        print(f"Average name match score: {sum(r['avg_name_score'] for r in summary['results']) / len(summary['results']):.2f}")

    return 0

if __name__ == "__main__":
    sys.exit(main())