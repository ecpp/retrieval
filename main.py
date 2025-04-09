#!/usr/bin/env python
import os
import sys
import argparse
from src.retrieval_system import RetrievalSystem


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='CAD Part and Assembly Retrieval System - Search by image, part name, or assembly',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""Examples:
        - Search by image: python main.py retrieve --query path/to/image.png --k 5 --visualize
        - Search by part name: python main.py retrieve --part-name "screw" --k 5 --visualize
        - Advanced search: python main.py retrieve --part-name "bolt" --match-threshold 0.6 --rotation-invariant
        - Search for similar assemblies: python main.py retrieve-assembly --step-id STEP123 --k 5 --visualize
        - Search by part to find similar assemblies: python main.py retrieve-assembly --part-image path/to/part.png --k 5
        """
    )

    # Define commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest data from STEP output directories')
    ingest_parser.add_argument('--dataset_dir', type=str, required=True,
                               help='Directory containing STEP output directories')
    ingest_parser.add_argument('--ingest-assembly', action='store_true', 
                               help='Also ingest assembly graph data (requires --use-assembly)')
    ingest_parser.add_argument('--use-assembly', action='store_true', 
                              help='Enable assembly similarity support')

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

    # Train assembly GNN command
    train_assembly_parser = subparsers.add_parser('train-assembly-gnn', 
                                                help='Train GNN for assembly similarity search')
    train_assembly_parser.add_argument('--graph_dir', type=str, 
                                      help='Directory containing hierarchical graph files')
    train_assembly_parser.add_argument('--batch_size', type=int, help='Batch size for training')
    train_assembly_parser.add_argument('--epochs', type=int, help='Number of epochs to train')
    train_assembly_parser.add_argument('--lr', type=float, help='Learning rate')
    train_assembly_parser.add_argument('--evaluate', action='store_true', help='Evaluate GNN after training')
    train_assembly_parser.add_argument('--use-assembly', action='store_true', help='Enable assembly similarity')

    # Build assembly index command  
    build_assembly_parser = subparsers.add_parser('build-assembly', 
                                                 help='Build vector index from assembly graphs')
    build_assembly_parser.add_argument('--graph_dir', type=str, 
                                      help='Directory containing hierarchical graph files')
    build_assembly_parser.add_argument('--use-assembly', action='store_true', 
                                      help='Enable assembly similarity')

    # Ingest assembly command (standalone)
    ingest_assembly_parser = subparsers.add_parser('ingest-assembly', 
                                                 help='Ingest assembly graph data from STEP output directories')
    ingest_assembly_parser.add_argument('--dataset_dir', type=str, required=True,
                                      help='Directory containing STEP output directories')
    ingest_assembly_parser.add_argument('--use-assembly', action='store_true', 
                                      help='Enable assembly similarity')

    # Retrieve similar assemblies command
    retrieve_assembly_parser = subparsers.add_parser('retrieve-assembly', 
                                                   help='Retrieve similar assemblies')
    retrieve_assembly_parser.add_argument('--step-id', type=str, 
                                        help='STEP file ID to search for')
    retrieve_assembly_parser.add_argument('--step-name', type=str, 
                                        help='STEP file name to search for')
    retrieve_assembly_parser.add_argument('--part-image', type=str,
                                        help='Find assembly from part image, then search for similar assemblies')
    retrieve_assembly_parser.add_argument('--graph-file', type=str,
                                        help='Path to hierarchical graph file to search with')
    retrieve_assembly_parser.add_argument('--k', type=int, default=10, 
                                        help='Number of results to retrieve')
    retrieve_assembly_parser.add_argument('--visualize', action='store_true', 
                                        help='Visualize the results')
    retrieve_assembly_parser.add_argument('--use-assembly', action='store_true', 
                                        help='Enable assembly similarity')
    retrieve_assembly_parser.add_argument('--match-threshold', type=float, default=0.7,
                                       help='Threshold for STEP name matching (0-1)')

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
    # Add search level to retrieve command
    retrieve_parser.add_argument('--search-level', type=str, choices=['part', 'assembly'], default='part',
                              help='Whether to search for similar parts or assemblies')
    retrieve_parser.add_argument('--use-assembly', action='store_true', 
                              help='Enable assembly similarity (for assembly search level)')

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
        
    # Override assembly usage based on command line arguments
    if hasattr(args, 'use_assembly') and args.use_assembly:
        # Initialize assembly components if not already initialized
        if not hasattr(retrieval_system, 'use_assembly') or not retrieval_system.use_assembly:
            retrieval_system.init_assembly_components()
        retrieval_system.use_assembly = True
        print("Assembly similarity enabled via command line")

    if args.command == 'ingest':
        print(f"Ingesting data from {args.dataset_dir}")
        retrieval_system.ingest_data(args.dataset_dir)
        
        # If ingest-assembly flag is set, also ingest assembly data
        if args.ingest_assembly:
            if not hasattr(retrieval_system, 'use_assembly') or not retrieval_system.use_assembly:
                print("Warning: Assembly similarity is not enabled. Enabling it for ingestion.")
                if not hasattr(retrieval_system, 'init_assembly_components'):
                    print("Error: Assembly components not available.")
                    return
                retrieval_system.init_assembly_components()
                retrieval_system.use_assembly = True
            
            retrieval_system.ingest_assembly_data(args.dataset_dir)

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
            
    # Handle assembly-specific commands
    elif args.command == 'ingest-assembly':
        if not hasattr(retrieval_system, 'use_assembly') or not retrieval_system.use_assembly:
            print("Assembly similarity is not enabled. Initializing assembly components...")
            if not hasattr(retrieval_system, 'init_assembly_components'):
                print("Error: Assembly components not available.")
                return
            retrieval_system.init_assembly_components()
            retrieval_system.use_assembly = True
        
        print(f"Ingesting assembly data from {args.dataset_dir}")
        retrieval_system.ingest_assembly_data(args.dataset_dir)
    
    elif args.command == 'train-assembly-gnn':
        if not hasattr(retrieval_system, 'use_assembly') or not retrieval_system.use_assembly:
            print("Assembly similarity is not enabled. Initializing assembly components...")
            if not hasattr(retrieval_system, 'init_assembly_components'):
                print("Error: Assembly components not available.")
                return
            retrieval_system.init_assembly_components()
            retrieval_system.use_assembly = True
        
        print("Training assembly GNN...")
        
        # Get graph directory
        graph_dir = args.graph_dir or retrieval_system.config.get("assembly", {}).get("graph_dir", "data/output/hierarchical_graphs")
        
        # Get training parameters
        batch_size = args.batch_size or retrieval_system.config.get("assembly", {}).get("batch_size", 8)
        epochs = args.epochs or retrieval_system.config.get("assembly", {}).get("epochs", 5)
        lr = args.lr or retrieval_system.config.get("assembly", {}).get("learning_rate", 1e-4)
        
        # Train the GNN
        model_path = retrieval_system.config.get("assembly", {}).get("model_path", "models/assembly_gnn.pt")
        history = retrieval_system.assembly_encoder.train_gnn(
            graph_dir=graph_dir,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            save_path=model_path
        )
        
        print("Assembly GNN training complete!")
    
    elif args.command == 'build-assembly':
        if not hasattr(retrieval_system, 'use_assembly') or not retrieval_system.use_assembly:
            print("Assembly similarity is not enabled. Initializing assembly components...")
            if not hasattr(retrieval_system, 'init_assembly_components'):
                print("Error: Assembly components not available.")
                return
            retrieval_system.init_assembly_components()
            retrieval_system.use_assembly = True
        
        print("Building assembly vector index")
        try:
            retrieval_system.build_assembly_index(args.graph_dir)
        except ValueError as e:
            print(f"Error: {e}")
            return
    
    elif args.command == 'retrieve-assembly':
        if not hasattr(retrieval_system, 'use_assembly') or not retrieval_system.use_assembly:
            print("Assembly similarity is not enabled. Initializing assembly components...")
            if not hasattr(retrieval_system, 'init_assembly_components'):
                print("Error: Assembly components not available.")
                return
            retrieval_system.init_assembly_components()
            retrieval_system.use_assembly = True
        
        # Determine the query type (only one should be specified)
        query_types = [args.step_id, args.step_name, args.part_image, args.graph_file]
        valid_types = [q for q in query_types if q is not None]
        
        if not valid_types:
            print("Error: One of --step-id, --step-name, --part-image, or --graph-file must be specified")
            return
        
        if len(valid_types) > 1:
            print("Warning: Multiple query types specified. Using the first valid one.")
        
        # Determine which query to use
        query = None
        if args.step_id:
            query = args.step_id
            print(f"Searching for assemblies similar to STEP ID: {query}")
        elif args.step_name:
            print(f"Searching for STEP file by name: {args.step_name}")
            # Find the STEP by name first
            step_match = retrieval_system.find_step_by_name(
                args.step_name,
                threshold=args.match_threshold
            )
            if not step_match:
                print(f"Could not find a STEP file matching '{args.step_name}' with threshold {args.match_threshold}")
                return
            query = step_match['step_id']
            print(f"Found matching STEP: '{query}' with {step_match['similarity']:.2f} similarity")
        elif args.part_image:
            query = args.part_image
            print(f"Finding assembly from part image and then searching for similar assemblies: {query}")
        elif args.graph_file:
            query = args.graph_file
            print(f"Searching for assemblies similar to graph file: {query}")
        
        # Execute the query
        results = retrieval_system.retrieve_similar_assemblies(query, k=args.k)
        
        # Print results
        if results and "paths" in results and results["paths"]:
            print(f"Top {len(results['paths'])} results:")
            
            # Create a format string for consistent output
            format_str = "{:3} | {:30} | {:20} | {:8}"
            
            # Print header
            print(format_str.format("Rank", "STEP ID", "Graph File", "Similarity"))
            print("-" * 70)
            
            for i, (path, info, similarity) in enumerate(zip(
                    results["paths"],
                    results.get("assembly_info", [None] * len(results["paths"])),
                    results.get("similarities", [None] * len(results["paths"]))
                )):
                
                # Get STEP ID
                step_id = info["step_id"] if info and "step_id" in info else "unknown"
                
                # Get graph filename
                graph_filename = os.path.basename(path) if path else "N/A"
                
                # Format similarity
                similarity_str = f"{similarity:.1f}%" if similarity is not None else "N/A"
                
                # Print the result in a nicely formatted table
                print(format_str.format(
                    f"{i+1}.",
                    step_id,
                    graph_filename,
                    similarity_str
                ))
            
            # Visualize if requested
            if args.visualize:
                output_path = retrieval_system.visualize_assembly_results(query, results)
                print(f"Visualization saved to: {output_path}")
        else:
            print("No results found.")

    elif args.command == 'retrieve':
        # Handle search level
        if hasattr(args, 'search_level') and args.search_level == 'assembly':
            # If searching at assembly level, redirect to assembly search
            if not hasattr(retrieval_system, 'use_assembly') or not retrieval_system.use_assembly:
                print("Assembly similarity is not enabled. Initializing assembly components...")
                if not hasattr(retrieval_system, 'init_assembly_components'):
                    print("Error: Assembly components not available.")
                    return
                retrieval_system.init_assembly_components()
                retrieval_system.use_assembly = True
            
            # Convert part retrieval parameters to assembly parameters
            if args.query:
                print(f"Using part image {args.query} to find similar assemblies")
                query = args.query
            elif args.part_name:
                print(f"Using part name {args.part_name} to find step name")
                # Find the part by name first
                part_match = retrieval_system.find_part_by_name(
                    args.part_name,
                    threshold=args.match_threshold
                )
                if not part_match:
                    print(f"Could not find a part matching '{args.part_name}' with threshold {args.match_threshold}")
                    return
                
                # Use the found part's image as query
                query = part_match["path"]
                print(f"Found part image at {query}, using it to find the assembly")
            else:
                print("Error: Either --query or --part-name must be specified")
                return
            
            # Execute assembly search
            results = retrieval_system.retrieve_similar_assemblies(query, k=args.k)
            
            # Handle results display and visualization
            if results and "paths" in results and results["paths"]:
                print(f"Top {len(results['paths'])} results:")
                
                # Create a format string for consistent output
                format_str = "{:3} | {:30} | {:20} | {:8}"
                
                # Print header
                print(format_str.format("Rank", "STEP ID", "Graph File", "Similarity"))
                print("-" * 70)
                
                for i, (path, info, similarity) in enumerate(zip(
                        results["paths"],
                        results.get("assembly_info", [None] * len(results["paths"])),
                        results.get("similarities", [None] * len(results["paths"]))
                    )):
                    
                    # Get STEP ID
                    step_id = info["step_id"] if info and "step_id" in info else "unknown"
                    
                    # Get graph filename
                    graph_filename = os.path.basename(path) if path else "N/A"
                    
                    # Format similarity
                    similarity_str = f"{similarity:.1f}%" if similarity is not None else "N/A"
                    
                    # Print the result in a nicely formatted table
                    print(format_str.format(
                        f"{i+1}.",
                        step_id,
                        graph_filename,
                        similarity_str
                    ))
                
                # Visualize if requested
                if args.visualize:
                    output_path = retrieval_system.visualize_assembly_results(query, results)
                    print(f"Visualization saved to: {output_path}")
            else:
                print("No results found.")
                
        else:
            # Regular part search (existing functionality)
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
            
        # Display assembly info if available
        assembly_info = info.get('assembly', {})
        if assembly_info.get('enabled', False):
            print("\nAssembly Search Information:")
            print(f"Assembly similarity enabled: {assembly_info.get('enabled', False)}")
            print(f"Assembly graph directory: {assembly_info.get('graph_dir', 'N/A')}")
            print(f"Assembly GNN trained: {assembly_info.get('gnn_trained', False)}")
            print(f"Assembly embedding dimension: {assembly_info.get('embedding_dim', 'N/A')}")
            
            # Print index stats if available
            index_stats = assembly_info.get('index_stats', {})
            if index_stats:
                print(f"Assembly index type: {index_stats.get('index_type', 'N/A')}")
                print(f"Assemblies in index: {index_stats.get('num_vectors', 0)}")

    else:
        print("Please specify a command. Run with --help for options.")


if __name__ == "__main__":
    main()