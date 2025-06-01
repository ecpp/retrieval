# CAD Part Retrieval System

This project implements an advanced multi-modal retrieval system for 3D CAD parts and assemblies. It leverages deep learning models for visual and metadata feature extraction, FAISS for efficient similarity searching, and offers various query modalities including image-based search, part name search, and full assembly comparison. The system is designed to be configurable and extensible, with an optional GUI for ease of use.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
  - [Image Encoding](#image-encoding)
  - [Metadata Encoding](#metadata-encoding)
  - [Multi-modal Fusion](#multi-modal-fusion)
  - [Vector Database](#vector-database)
  - [Search Mechanisms](#search-mechanisms)
- [Project Structure](#project-structure)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
- [Usage](#usage)
  - [Command-Line Interface (CLI)](#command-line-interface-cli)
    - [Data Ingestion](#data-ingestion)
    - [Train Metadata Autoencoder](#train-metadata-autoencoder)
    - [Build Index](#build-index)
    - [Retrieve Similar Parts/Assemblies](#retrieve-similar-partsassemblies)
    - [Evaluate System](#evaluate-system)
    - [System Information](#system-information)
    - [List Assembly Parts](#list-assembly-parts)
  - [Graphical User Interface (GUI)](#graphical-user-interface-gui)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## Overview

The CAD Part Retrieval System aims to provide an efficient solution for finding visually and semantically similar CAD parts or assemblies from a large database. Users can query the system using a part image, a part name, or an entire assembly ID. The system then returns a ranked list of similar items. Advanced features include rotation-invariant search for visual queries and metadata integration for more context-aware retrieval.

## Features

- Multi-modal Retrieval: Combines visual features from images with textual/numeric features from Bill of Materials (BOM) metadata.
- Flexible Image Encoders: Supports state-of-the-art models like DINOv2 (default), ViT, and ResNet50 for image feature extraction.
- Metadata Integration: Utilizes an Autoencoder to learn compact representations of BOM metadata, which can be fused with visual features.
- Efficient Vector Search: Employs FAISS for fast similarity searches in the embedding space.
- Diverse Query Types:
  - Image Query: Find parts visually similar to a query image.
  - Part Name Query: Search for a part by its name, then find visually similar parts.
  - Full Assembly Query: Find assemblies similar to a query assembly based on constituent parts. Optionally select specific parts for comparison.
- Rotation-Invariant Search: Enhances visual search robustness by considering multiple orientations of the query image.
- Configurable System: Key parameters for models, data paths, indexing, search, and metadata are configurable via a YAML file.
- Data Processing Pipeline: Includes tools for ingesting CAD data (images and BOMs) and preparing it for the system.
- Evaluation Module: Provides tools for evaluating retrieval performance using metrics like Precision@K and Recall@K.
- Graphical User Interface (GUI): An optional PyQt5-based GUI for interactive use of the system's functionalities.
- Command-Line Interface (CLI): Comprehensive CLI for all system operations.

## System Architecture

The system is composed of several key modules that work together:

### Image Encoding

- The `ImageEncoder` (in `src/image_encoder.py`) is responsible for transforming part images into high-dimensional feature vectors (embeddings).
- It supports multiple architectures:
  - DINOv2: A self-supervised Vision Transformer, often providing excellent performance for visual similarity tasks, especially with engineering parts. This is the default.
  - ViT (Vision Transformer): A standard Transformer-based image model.
  - ResNet50: A widely-used Convolutional Neural Network.
- Images are preprocessed (resized, normalized) according to the requirements of the chosen model.
- The embedding dimension is configurable (e.g., 768 for DINOv2/ViT).

### Metadata Encoding

- When metadata integration is enabled (`metadata.enabled: true` in `config.yaml`), the `MetadataEncoder` (in `src/metadata_encoder.py`) processes BOM (Bill of Materials) data associated with CAD parts.
- Source Data: BOM data is expected in JSON format, typically extracted from STEP files.
- Feature Extraction: Relevant numerical and categorical features are extracted from the BOM (e.g., dimensions, volume, surface area, material properties, topological metrics, surface composition).
- Autoencoder: A neural network autoencoder is used to learn a compressed, dense representation (embedding) of this metadata.
  - The architecture (hidden layers, latent dimension) is configurable.
  - The autoencoder needs to be trained separately using the `train-autoencoder` command. The trained model is saved and loaded for indexing and retrieval.
- This allows the system to capture semantic information beyond visual appearance.

### Multi-modal Fusion

- If metadata is used, the `FusionModule` (in `src/fusion_module.py`) combines the image embedding and the metadata embedding into a single, unified embedding.
- Fusion Methods:
  - Concatenation (`concat`): Visual and metadata embeddings are concatenated, and a linear layer projects them to the final embedding dimension.
  - Weighted Sum (`weighted`): A weighted sum of visual and metadata embeddings is computed. The current implementation uses learnable scalar weights for visual and metadata embeddings which are normalized to sum to 1. Metadata embeddings are padded or truncated if their dimension doesn't match the visual embedding dimension for this method.
- The fused embedding is then used for indexing and similarity search.

### Vector Database

- The `VectorDatabase` (in `src/vector_database.py`) manages the storage and retrieval of embeddings.
- Technology: It uses FAISS, a library for efficient similarity search and clustering of dense vectors.
  - By default, it uses `IndexFlatL2`, which performs an exact search using L2 (Euclidean) distance.
- Functionality:
  - Stores the (potentially fused) embeddings.
  - Maintains a mapping between vector indices and their corresponding part image paths and metadata (like part name and parent STEP file).
  - Supports saving the index and metadata to disk for persistence and reloading them.

### Search Mechanisms

1. Visual Similarity Search:

   - The query image is encoded (and potentially fused with its metadata if available and enabled).
   - The resulting embedding is used to search the FAISS index for the k-nearest neighbors.
   - Rotation Invariance: For visual queries, multiple rotated versions of the query image can be generated and encoded. Their results are then intelligently combined to find parts similar regardless of orientation. This is handled by functions in `src/rotational_utils.py`.
2. Part Name Search:

   - This is a two-stage process detailed in `PART_NAME_SEARCH.md` and implemented in `RetrievalSystem.retrieve_by_part_name`.
   - Stage 1 (Text Match): The system searches its database of part names (derived from indexed items) for the best textual match to the query part name. This involves name normalization and similarity calculation using a weighted combination of Jaccard similarity (character-level), length ratio, and word-level Jaccard similarity. Weights and the matching threshold are configurable in `config.yaml`.
   - Stage 2 (Visual Search): The image of the best-matched part from Stage 1 is then used as the query for a standard visual similarity search.
3. Full Assembly Search:

   - Users provide an assembly ID. The system identifies all parts belonging to this query assembly.
   - Optionally, users can specify a subset of these parts using `--select-parts`.
   - For each part in the (selected) query assembly, the system performs a visual similarity search to find similar parts in other assemblies.
   - An assembly similarity score is then computed for each candidate assembly in the database. This score considers the average similarity of matched parts and the coverage ratio (how many parts of the query assembly found good matches in the candidate assembly). The score is significantly penalized for low coverage.
   - Results are ranked by this assembly similarity score. Full assembly images (if available in `data/output/full_assembly_images`) are used for visualization.

## Project Structure

```
.
├── config/
│   └── config.yaml                 # Main configuration file
├── data/                           # Default directory for input, output, database, models
│   ├── input/                      # Directory for raw STEP outputs
│   ├── output/                     # Processed data, evaluation results, visualizations
│   │   ├── images/                 # Flattened part images for indexing
│   │   ├── full_assembly_images/   # Full assembly images
│   │   ├── bom/                    # BOM JSON files for metadata encoder
│   │   ├── evaluation/             # Evaluation results and plots
│   │   └── results/                # Retrieval visualizations
│   └── database/                   # FAISS index and metadata storage (legacy, now in models/)
├── models/                         # Stores trained models and indexes
│   ├── faiss_index.bin             # Saved FAISS index
│   ├── index_metadata.pkl          # Metadata for the FAISS index
│   └── metadata_autoencoder.pt     # Trained metadata autoencoder model
├── src/
│   ├── __init__.py
│   ├── data_processor.py           # Processes raw CAD data (images, BOMs)
│   ├── image_encoder.py            # Encodes images into embeddings
│   ├── metadata_encoder.py         # Encodes BOM metadata using an autoencoder
│   ├── fusion_module.py            # Fuses visual and metadata embeddings
│   ├── vector_database.py          # Manages FAISS index for embeddings
│   ├── retrieval_system.py         # Core logic integrating all modules
│   ├── evaluator.py                # Evaluates retrieval performance
│   ├── rotational_utils.py         # Utilities for rotation-invariant search
│   └── gui_utils.py                # (If any utility functions for GUI exist)
├── main.py                         # CLI entry point for system operations
├── retrieval_gui.py                # PyQt5-based Graphical User Interface
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── USER_GUIDE.md                   # Guide for end-users
├── DEVELOPER_GUIDE.md              # Guide for developers
├── PART_NAME_SEARCH.md             # Details on the part name search feature
├── check_model.py                  # Utility to check current model and PyTorch setup
├── check_rotations.py              # Utility to visualize image rotations
├── debug_metadata.py               # Utility to debug metadata module imports
├── setup_env.py                    # Script to check and install dependencies
├── test_autoencoder.py             # Script to test the trained metadata autoencoder
├── test_retrieval.py               # Basic test script for the retrieval system
├── train_autoencoder.py            # Standalone script to train the metadata autoencoder
└── worker_thread.py                # QThread for running background processes in GUI
```

## Installation

### Prerequisites

- Python (3.10 recommended).
- Conda (recommended for managing environments).
- Access to a terminal or command prompt.
- For GPU acceleration: An NVIDIA GPU with appropriate CUDA drivers.

### Setup

1. Clone the Repository:

   ```bash
   git clone <your-repository-url>
   cd <repository-name>
   ```
2. Create and Activate Conda Environment:

   ```bash
   conda create -n cadretrieval python=3.10
   conda activate cadretrieval
   ```
3. Install Dependencies:
   It's recommended to install PyTorch first, matching your CUDA version if applicable. Check the [official PyTorch website](https://pytorch.org/get-started/locally/) for the correct command. For example, for CUDA 11.8:

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

   Then, install the remaining dependencies from `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

   Alternatively, you can try the `setup_env.py` script (ensure it's adapted for your environment if needed):

   ```bash
   python setup_env.py
   ```

## Usage

The system can be operated via a Command-Line Interface (CLI) or a Graphical User Interface (GUI).

### Command-Line Interface (CLI)

The main entry point for CLI operations is `main.py`.

#### Data Ingestion

Processes raw STEP file outputs (part images, BOM JSONs) from a dataset directory, organizes them, and copies BOMs for metadata processing.

```bash
python main.py ingest --dataset_dir /path/to/your/step_outputs
```

- `--dataset_dir`: Directory containing subdirectories, where each subdirectory is the output of processing a single STEP file (should contain an `images/` folder and a `*_bom.json` file).

#### Train Metadata Autoencoder

Trains the autoencoder on BOM data for metadata embedding. This is required if `metadata.enabled` is true.

```bash
python main.py train-autoencoder [--bom_dir /path/to/bom/files] [--batch_size 32] [--epochs 50] [--lr 0.0001] [--evaluate] [--use-metadata]
```

- `--bom_dir`: Directory containing BOM JSON files (defaults to `data/output/bom` as per `config.yaml`).
- `--batch_size`, `--epochs`, `--lr`: Training parameters.
- `--evaluate`: Evaluate the autoencoder after training.
- `--use-metadata`: Ensures metadata components are initialized; crucial if you intend to build an index with metadata. This flag should be consistent with the `metadata: enabled` setting in `config.yaml` for the `train-autoencoder`, `build`, and `retrieve` commands.

*Note:* The `train_autoencoder.py` script offers a standalone way to train with more specific parameter overrides if needed.

#### Build Index

Encodes part images (and metadata, if enabled) and builds/updates the FAISS vector index.

```bash
python main.py build [--image_dir /path/to/images] [--use-metadata]
```

- `--image_dir`: Directory containing part images to index (defaults to `data/output/images`).
- `--use-metadata`: If specified, enables metadata integration during indexing. The metadata autoencoder must be trained first.

#### Retrieve Similar Parts/Assemblies

Performs similarity search.

```bash
# By image
python main.py retrieve --query /path/to/query_image.png [--k 10] [--visualize] [--rotation-invariant] [--use-metadata]

# By part name
python main.py retrieve --part-name "bearing_housing" [--k 10] [--visualize] [--rotation-invariant] [--use-metadata] [--match-threshold 0.7]

# By full assembly
python main.py retrieve --full-assembly "assembly_id_123" [--k 5] [--visualize] [--select-parts "part_a.png" "part_b.png"] [--use-metadata]
```

- `--query`: Path to the query image.
- `--part-name`: Name of the part to search for.
- `--full-assembly`: ID of the assembly to query.
- `--select-parts`: (Used with `--full-assembly`) A list of specific part filenames from the query assembly to use for comparison.
- `--k`: Number of results to retrieve.
- `--visualize`: Generate and save a visualization of the results.
- `--rotation-invariant`: Enable rotation-invariant search for image/part name queries.
- `--num-rotations`: Number of rotations for rotation-invariant search (default 8).
- `--use-metadata`: Use metadata in the retrieval process (query encoding and/or reranking).
- `--match-threshold`: Similarity threshold (0-1) for part name matching.

#### Evaluate System

Evaluates retrieval performance against a ground truth dataset.

```bash
python main.py evaluate [--query_dir /path/to/query/images] [--ground_truth /path/to/ground_truth.json] [--use-metadata]
```

- `--query_dir`: Directory containing query images for evaluation.
- `--ground_truth`: JSON file mapping query image filenames to lists of relevant result image filenames.

#### System Information

Displays information about the current system configuration, model, and index.

```bash
python main.py info
```

#### List Assembly Parts

Lists all parts found for a given assembly ID. This helps identify part filenames for use with the `--select-parts` option in assembly search.

```bash
python main.py list-assembly-parts --assembly-id "assembly_id_123"
```

### Graphical User Interface (GUI)

Launch the GUI using:

```bash
python retrieval_gui.py
```

The GUI provides an interactive way to perform most of the CLI operations, including:

- Data Ingestion
- Training the Metadata Autoencoder
- Building the Index
- Retrieval (Image Query, Part Name Query, Full Assembly Query with part selection)
- Viewing output logs and result images.

## Configuration

The primary configuration for the system is managed in `config/config.yaml`. This file allows you to set:

- `model`: Image encoder settings (name, pretrained, embedding_dim, image_size).
- `data`: Default paths for input, output, and database directories.
- `indexing`: Paths for FAISS index and its metadata file.
- `text_search`: Parameters for part name search (default_threshold, normalize_names, similarity_weights for Jaccard, length ratio, word Jaccard).
- `training`: Default batch size and number of workers for image encoding during indexing.
- `evaluation`: Default `top_k` values for metrics.
- `metadata`:
  - `enabled`: Whether to use metadata integration.
  - `embedding_dim`: Output dimension of the metadata autoencoder.
  - `fusion_method`: How to combine visual and metadata embeddings ("concat" or "weighted").
  - `bom_dir`: Directory for BOM JSON files.
  - `size_weight`: Weight given to size similarity in reranking (0-1).
  - Autoencoder architecture (`hidden_dims`), model path (`model_path`), training parameters (`batch_size`, `epochs`, `learning_rate`).
