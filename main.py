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
        - Assembly search: python main.py retrieve --full-assembly "A123" --k 5
        - List assembly parts: python main.py list-assembly-parts --assembly-id "A123"
        - Search with selected parts: python main.py retrieve --full-assembly "A123" --select-parts "A123_part1.png" "A123_part2.png"
        """
    )

    # Define commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest data from STEP output directories')

    ingest_parser.add_argument('--dataset_dir', type=str, required=True,
                               help='Directory containing STEP output directories')

    # Train autoencoder command
    train_parser = subparsers.add_parser('train-autoencoder', help='Train metadata autoencoder for improved retrieval')
    train_parser.add_argument('--bom_dir', type=str, help='Directory containing BOM files (default: data/output/bom)')
    train_parser.add_argument('--batch_size', type=int, help='Batch size for training')
    train_parser.add_argument('--epochs', type=int, help='Number of epochs to train')
    train_parser.add_argument('--lr', type=float, help='Learning rate')
    train_parser.add_argument('--evaluate', action='store_true', help='Evaluate autoencoder after training')
    train_parser.add_argument('--use-metadata', action='store_true', help='Enable metadata integration')



    # Build index command
    build_parser = subparsers.add_parser('build', help='Build vector index from images')
    build_parser.add_argument('--image_dir', type=str, help='Directory containing images to index')
    build_parser.add_argument('--use-metadata', action='store_true', help='Use metadata for indexing')

    # Retrieve command
    retrieve_parser = subparsers.add_parser('retrieve', help='Retrieve similar parts')
    retrieve_parser.add_argument('--query', type=str, help='Path to query image')
    retrieve_parser.add_argument('--part-name', type=str, help='Part name to search for')
    retrieve_parser.add_argument('--full-assembly', type=str, help='Assembly ID to search for similar assemblies')
    retrieve_parser.add_argument('--select-parts', type=str, nargs='+',
                                help='List of part filenames to include in assembly search (only used with --full-assembly)')
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

    # List assembly parts command
    list_parts_parser = subparsers.add_parser('list-assembly-parts', help='List all parts in an assembly')
    list_parts_parser.add_argument('--assembly-id', type=str, required=True, help='Assembly ID to list parts for')

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

    # Process commands
    if args.command == 'ingest':
        print(f"Ingesting data from {args.dataset_dir}")
        retrieval_system.ingest_data(args.dataset_dir)

    elif args.command == 'list-assembly-parts':
        # List all parts for the specified assembly
        assembly_id = args.assembly_id
        print(f"Listing parts for assembly ID: {assembly_id}")

        # Find all parts that belong to the assembly
        image_dir = os.path.join(retrieval_system.config["data"]["output_dir"], "images")
        assembly_parts = []

        # Pattern for parts belonging to this assembly: "{assembly_id}_*.png"
        for filename in os.listdir(image_dir):
            if filename.startswith(f"{assembly_id}_") and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                assembly_parts.append(filename)

        if not assembly_parts:
            print(f"No parts found for assembly ID {assembly_id}")
            return

        print(f"Found {len(assembly_parts)} parts for assembly ID {assembly_id}:")

        # Sort parts by name for easier viewing
        assembly_parts.sort()

        # Print parts in a formatted list
        for i, part_filename in enumerate(assembly_parts):
            # Try to extract more information about the part if available
            part_info = None
            try:
                part_info = retrieval_system.extract_part_info(os.path.join(image_dir, part_filename))
            except:
                pass

            part_name = part_info.get("part_name", "Unknown") if part_info else "Unknown"
            print(f"{i+1}. {part_filename} - {part_name}")

        # Print a command example for selecting specific parts
        parts_example = " ".join(assembly_parts[:min(3, len(assembly_parts))])
        print(f"\nTo search using specific parts, use: python main.py retrieve --full-assembly {assembly_id} --select-parts {parts_example}")

    elif args.command == 'train-autoencoder':
        # Make sure metadata is enabled first
        if not retrieval_system.use_metadata:
            print("Error: Metadata integration is not enabled. Please enable it in config.yaml or use --use-metadata.")
            return

        print("Training metadata autoencoder...")

        # Get BOM directory
        bom_dir = args.bom_dir or retrieval_system.config.get("metadata", {}).get("bom_dir", "data/output/bom")

        # Create models directory if it doesn't exist
        model_path = retrieval_system.config.get("metadata", {}).get("model_path", "models/metadata_autoencoder.pt")
        model_dir = os.path.dirname(model_path)
        os.makedirs(model_dir, exist_ok=True)

        # Get training parameters
        batch_size = args.batch_size or retrieval_system.config.get("metadata", {}).get("batch_size", 32)
        epochs = args.epochs or retrieval_system.config.get("metadata", {}).get("epochs", 50)
        lr = args.lr or retrieval_system.config.get("metadata", {}).get("learning_rate", 1e-4)

        # Train the autoencoder
        history = retrieval_system.metadata_encoder.train_autoencoder(
            bom_dir=bom_dir,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            save_path=model_path
        )

        # Evaluate if requested
        if args.evaluate and 'train_loss' in history and len(history['train_loss']) > 0:
            print("Evaluating trained autoencoder...")
            # Create dataset for evaluation
            from src.metadata_encoder import BomDataset
            dataset = BomDataset(bom_dir, retrieval_system.metadata_encoder)

            # Import evaluate function from train_autoencoder.py
            from train_autoencoder import evaluate_autoencoder
            evaluate_autoencoder(retrieval_system.metadata_encoder, dataset)

        print("Autoencoder training complete!")

    elif args.command == 'build':
        print("Building vector index")
        try:
            retrieval_system.build_index(args.image_dir)
        except ValueError as e:
            print(f"Error: {e}")
            return

    elif args.command == 'retrieve':
        # Validate input - either query image or part name must be specified
        if not args.query and not args.part_name and not args.full_assembly:
            print("Error: Either --query, --part-name, or --full-assembly must be specified")
            return

        if args.query and (args.part_name or args.full_assembly):
            print("Warning: Multiple query types specified, using query image")
            args.part_name = None
            args.full_assembly = None

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
        elif args.full_assembly:
            # Process full assembly search
            print(f"Searching for similar assemblies to assembly ID: {args.full_assembly}")

            # If parts are selected, pass them to the retrieve_by_assembly function
            if args.select_parts:
                print(f"User selected {len(args.select_parts)} parts for similarity search: {args.select_parts}")
                results = retrieval_system.retrieve_by_assembly(
                    args.full_assembly,
                    k=args.k,
                    selected_parts=args.select_parts
                )
            else:
                # No parts selected, use all parts
                results = retrieval_system.retrieve_by_assembly(
                    args.full_assembly,
                    k=args.k
                )
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
                # For part name searches, pass the results directly for proper naming
                if args.part_name:
                    retrieval_system.visualize_results(args.query, results)
                else:
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
