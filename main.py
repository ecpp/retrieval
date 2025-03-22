#!/usr/bin/env python
import os
import sys
import argparse
from src.retrieval_system import RetrievalSystem


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='CAD Part Retrieval System - Search by image or part name', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""Examples:
        - Search by image: python main.py retrieve --query path/to/image.png --k 5 --visualize
        - Search by part name: python main.py retrieve --part-name "screw" --k 5 --visualize
        - Advanced search: python main.py retrieve --part-name "bolt" --match-threshold 0.6 --rotation-invariant
        """
    )

    # Define commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest data from STEP output directories')

    ingest_parser.add_argument('--dataset_dir', type=str, required=True,
                               help='Directory containing STEP output directories')



    # Build index command
    build_parser = subparsers.add_parser('build', help='Build vector index from images')
    build_parser.add_argument('--image_dir', type=str, help='Directory containing images to index')
    build_parser.add_argument('--use-metadata', action='store_true', help='Use metadata for indexing')

    # Retrieve command
    retrieve_parser = subparsers.add_parser('retrieve', help='Retrieve similar parts')
    retrieve_parser.add_argument('--query', type=str, help='Path to query image')
    retrieve_parser.add_argument('--part-name', type=str, help='Part name to search for')
    retrieve_parser.add_argument('--k', type=int, default=10, help='Number of results to retrieve')
    retrieve_parser.add_argument('--visualize', action='store_true', help='Visualize the results')
    retrieve_parser.add_argument('--rotation-invariant', action='store_true', help='Enable rotation-invariant search')
    retrieve_parser.add_argument('--num-rotations', type=int, default=8, help='Number of rotations to try')
    retrieve_parser.add_argument('--use-metadata', action='store_true', help='Use metadata for retrieval')
    retrieve_parser.add_argument('--match-threshold', type=float, default=0.7, 
                                 help='Threshold for part name matching (0-1)')

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate retrieval system')
    eval_parser.add_argument('--query_dir', type=str, help='Directory containing query images')
    eval_parser.add_argument('--ground_truth', type=str, help='Path to ground truth JSON file')
    eval_parser.add_argument('--use-metadata', action='store_true', help='Use metadata for evaluation')

    # Info command
    subparsers.add_parser('info', help='Display system information')

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    # Initialize the retrieval system
    retrieval_system = RetrievalSystem()

    # Override metadata usage based on command line arguments
    if hasattr(args, 'use_metadata') and args.use_metadata:
        retrieval_system.use_metadata = True
        print("Metadata integration enabled via command line")

    if args.command == 'ingest':
        print(f"Ingesting data from {args.dataset_dir}")
        retrieval_system.ingest_data(args.dataset_dir)

    elif args.command == 'build':
        print("Building vector index")
        retrieval_system.build_index(args.image_dir)

    elif args.command == 'retrieve':
        # Validate input - either query image or part name must be specified
        if not args.query and not args.part_name:
            print("Error: Either --query or --part-name must be specified")
            return
        
        if args.query and args.part_name:
            print("Warning: Both query image and part name specified, using query image")
            args.part_name = None
            
        # Process part name search if specified
        if args.part_name:
            print(f"Searching for part name: {args.part_name}")
            if args.rotation_invariant:
                print(f"Using rotation-invariant search with {args.num_rotations} rotations")
                
            # First find the part by name using the threshold
            part_match = retrieval_system.find_part_by_name(
                args.part_name,
                threshold=args.match_threshold
            )
            
            # If no match found, exit early
            if not part_match:
                print(f"Could not find a part matching '{args.part_name}' with threshold {args.match_threshold}")
                return
                
            print(f"Found matching part: '{part_match['part_name']}' with {part_match['similarity']:.2f} similarity")
                
            # Now retrieve similar parts
            results = retrieval_system.retrieve_by_part_name(
                args.part_name,
                k=args.k,
                rotation_invariant=args.rotation_invariant,
                num_rotations=args.num_rotations,
                threshold=args.match_threshold
            )
            
            # If successful, set args.query to the matched image path for visualization
            if results and "paths" in results and len(results["paths"]) > 0:
                if "query_match" in results and "path" in results["query_match"]:
                    args.query = results["query_match"]["path"]
                    print(f"Using matched part image for visualization: {args.query}")
        else:
            # Process standard image query
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
        if results and "paths" in results:
            print(f"Top {len(results['paths'])} results:")

            # Create a format string for consistent output
            format_str = "{:3} | {:30} | {:20} | {:20} | {:8}"

            # Print header
            print(format_str.format("Rank", "Part", "STEP File", "Part Name", "Similarity"))
            print("-" * 90)

            for i, (path, info, similarity) in enumerate(zip(
                    results["paths"],
                    results.get("part_info", [None] * len(results["paths"])),
                    results.get("similarities", [None] * len(results["paths"]))
                )):

                # Get filename without extension
                filename = os.path.splitext(os.path.basename(path))[0] if path else "N/A"

                # Get part information
                parent_step = "unknown"
                part_name = "unknown"
                if info:
                    parent_step = info.get("parent_step", "unknown")
                    part_name = info.get("part_name", "unknown")

                # Use recalibrated similarity score if available, otherwise calculate from distance
                if similarity is not None:
                    similarity_str = f"{similarity:.1f}%"
                else:
                    # Legacy fallback - calculate from distance
                    distance = results["distances"][i]
                    fallback_similarity = 100 * (1 / (1 + distance))
                    similarity_str = f"{fallback_similarity:.1f}%"

                # Print the result in a nicely formatted table
                print(format_str.format(
                    f"{i+1}.",
                    filename,
                    parent_step,
                    part_name,
                    similarity_str
                ))

            # Visualize if requested
            if args.visualize and args.query:
                retrieval_system.visualize_results(args.query, results)
        else:
            print("No results found.")

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
        print(f"Metadata enabled: {retrieval_system.use_metadata}")
        if retrieval_system.use_metadata:
            print(f"Metadata embedding dimension: {info.get('metadata', {}).get('embedding_dim')}")
            print(f"Fusion method: {info.get('metadata', {}).get('fusion_method')}")

    else:
        print("Please specify a command. Run with --help for options.")


if __name__ == "__main__":
    main()
