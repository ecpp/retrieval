# Developer Guide for CAD Part Retrieval System

This guide provides an in-depth technical overview of the CAD Part Retrieval System, including its architecture, core components, data flow, and guidelines for extension.

## Table of Contents

- [System Architecture](#system-architecture)
  - [Overall Flow](#overall-flow)
  - [Core Modules](#core-modules)
- [Code Structure](#code-structure)
- [Detailed Component Breakdown](#detailed-component-breakdown)
  - [1. Data Processor (`src/data_processor.py`)](#1-data-processor-srcdata_processorpy)
  - [2. Image Encoder (`src/image_encoder.py`)](#2-image-encoder-srcimage_encoderpy)
  - [3. Metadata Encoder (`src/metadata_encoder.py`)](#3-metadata-encoder-srcmetadata_encoderpy)
  - [4. Fusion Module (`src/fusion_module.py`)](#4-fusion-module-srcfusion_modulepy)
  - [5. Vector Database (`src/vector_database.py`)](#5-vector-database-srcvector_databasepy)
  - [6. Retrieval System (`src/retrieval_system.py`)](#6-retrieval-system-srcretrieval_systempy)
  - [7. Rotational Utilities (`src/rotational_utils.py`)](#7-rotational-utilities-srcrotational_utilspy)
  - [8. Evaluator (`src/evaluator.py`)](#8-evaluator-srcevaluatorpy)
  - [9. Main CLI (`main.py`)](#9-main-cli-mainpy)
  - [10. GUI (`retrieval_gui.py`)](#10-gui-retrieval_guipy)
- [Key Algorithms and Logic](#key-algorithms-and-logic)
  - [Part Name Matching](#part-name-matching)
  - [Rotation-Invariant Search](#rotation-invariant-search)
  - [Assembly Similarity Scoring](#assembly-similarity-scoring)
  - [Metadata Autoencoder Training](#metadata-autoencoder-training)
  - [Size-based Reranking](#size-based-reranking)
- [Configuration (`config/config.yaml`)](#configuration-configconfigyaml)
- [Extending the System](#extending-the-system)
  - [Adding a New Image Encoder](#adding-a-new-image-encoder)
  - [Modifying Metadata Features](#modifying-metadata-features)
  - [Implementing a New Fusion Method](#implementing-a-new-fusion-method)
  - [Using a Different FAISS Index](#using-a-different-faiss-index)
- [Development and Debugging Scripts](#development-and-debugging-scripts)
- [Troubleshooting for Developers](#troubleshooting-for-developers)

## System Architecture

### Overall Flow

The system operates in several main stages:

1. **Data Ingestion:** Raw CAD data (part images, assembly images, BOM JSON files) is processed. Part images are typically copied to a common directory, and BOM files are collected.
2. **Metadata Autoencoder Training (Optional):** If metadata integration is enabled, the `MetadataEncoder`'s autoencoder is trained on the collected BOM data to learn how to create compact embeddings from metadata features.
3. **Index Building:**
   a. Part images are fed through the `ImageEncoder` to generate visual embeddings.
   b. If metadata is enabled, corresponding BOM data for each part is fed through the trained `MetadataEncoder` to generate metadata embeddings.
   c. Visual and metadata embeddings are combined by the `FusionModule` to produce a final, multi-modal embedding for each part. If metadata is disabled, only visual embeddings are used.
   d. These final embeddings, along with paths and basic part information, are added to the `VectorDatabase` (FAISS index).
4. **Retrieval (Querying):**
   a. A query is provided (image, part name, or assembly ID).
   b. **For Image/Part Name Queries:** The query image (or an image found via part name text search) is processed through the same encoding (and fusion, if applicable) pipeline as during indexing to get a query embedding. Rotation invariance techniques may be applied here.
   c. **For Assembly Queries:** Parts of the query assembly are identified. Each part effectively becomes a sub-query.
   d. The query embedding(s) are searched against the FAISS index to find the k-nearest neighbors.
   e. **For Assembly Queries:** Results from part sub-queries are aggregated to score and rank other assemblies.
   f. Results can be optionally reranked using heuristics like size similarity if metadata is available.
   g. Results are presented to the user, potentially with visualizations.

### Core Modules

- **`RetrievalSystem`:** Orchestrates the overall process, integrating all other modules.
- **`ImageEncoder`:** Converts images to visual feature vectors.
- **`MetadataEncoder`:** Converts BOM data to metadata feature vectors using a trained autoencoder.
- **`FusionModule`:** Combines visual and metadata features.
- **`VectorDatabase`:** Stores and searches feature vectors using FAISS.
- **`DataProcessor`:** Handles initial data loading and preprocessing.
- **`RotationalUtils`:** Provides functions for rotation-invariant search.
- **`Evaluator`:** Calculates performance metrics.
- **`main.py`:** Provides the CLI.
- **`retrieval_gui.py`:** Provides the GUI.

## Code Structure

The project is primarily structured within the `src/` directory, with `main.py` and `retrieval_gui.py` as main entry points.

```
.
├── config/config.yaml          # System configuration
├── data/                       # Default data directories
├── models/                     # Default model/index storage
├── src/                        # Core source code
│   ├── data_processor.py
│   ├── image_encoder.py
│   ├── metadata_encoder.py
│   ├── fusion_module.py
│   ├── vector_database.py
│   ├── retrieval_system.py
│   ├── evaluator.py
│   ├── rotational_utils.py
│   └── __init__.py
├── main.py                     # CLI application
├── retrieval_gui.py            # GUI application
├── requirements.txt
├── *.md                        # Documentation
└── *.py                        # Utility/testing scripts
```

## Detailed Component Breakdown

### 1. Data Processor (`src/data_processor.py`)

- **Purpose:** Ingests and preprocesses CAD data from STEP file outputs.
- **`process_step_output(step_output_dir)`:** Processes a single STEP output directory, extracting part image paths and loading BOM data.
- **`process_dataset(dataset_dir)`:** Iterates through a dataset of STEP outputs, calling `process_step_output` for each.
- **`copy_to_flat_structure(all_parts, dest_dir)`:** Copies all part images and full assembly images into standardized flat directories (`data/output/images/` and `data/output/full_assembly_images/`) for easier access during indexing. Uses `ThreadPoolExecutor` for parallel copying.
- **`save_processed_data(all_parts, output_file)`:** Saves a summary of processed part information to a JSON file.

### 2. Image Encoder (`src/image_encoder.py`)

- **Purpose:** Generates dense vector embeddings from part images.
- **Models Supported:** DINOv2 (`facebook/dinov2-base`), ViT (`google/vit-base-patch16-224`), ResNet50. Configured via `config.yaml -> model -> name`.
- **`__init__(...)`:** Loads the specified pretrained model and corresponding image processor/transformations. Moves model to GPU if available.
- **`encode_image(image_path)`:** Encodes a single image. Loads image, applies transformations, passes through the model, and returns the CLS token embedding (for Transformers) or flattened feature map (for ResNet).
- **`encode_batch(image_paths, batch_size)`:** Encodes a list of images in batches for efficiency.

### 3. Metadata Encoder (`src/metadata_encoder.py`)

- **Purpose:** Learns to represent BOM metadata as dense embeddings using an autoencoder architecture.
- **`MetadataEncoder` Class:**
  - **`__init__(output_dim, hidden_dims)`:** Defines the encoder and decoder networks (fully connected layers). `output_dim` is the latent space dimension.
  - **`get_input_dim()`:** Returns the fixed number of features extracted from BOM.
  - **`extract_features(metadata)`:** Extracts a predefined set of ~46 numerical features from a part's BOM data. This includes dimensions (length, width, height, max/min/mid), ratios, volume, surface area, topological counts (faces, edges, vertices), surface composition counts (planes, cylinders, etc.), surface ratios, and a one-hot encoding of primary shape type. Handles missing values and potential capitalization differences in keys. Applies clipping to extreme values.
  - **`_preprocess_features(features_tensor)`:** Applies NaN/inf handling, clipping, and Z-score normalization (if `self.feature_means` and `self.feature_stds` are computed).
  - **`_compute_scaling_parameters(dataset)`:** Calculates mean and std for each feature across the training dataset for Z-score normalization.
  - **`encode_metadata(metadata, normalize)`:** Extracts features, preprocesses them, and passes them through the `self.encoder` network.
  - **`reconstruct(features_tensor)`:** Passes preprocessed features through encoder then decoder to get reconstructed features.
  - **`train_autoencoder(bom_dir, ...)`:** Manages the training loop for the autoencoder.
    - Uses `BomDataset` to load data.
    - Computes scaling parameters.
    - Uses Adam optimizer and MSELoss.
    - Saves the trained encoder, decoder, and scaling parameters.
  - **`load_trained_model(model_path)`:** Loads a saved autoencoder model and scaling parameters.
- **`BomDataset` Class:**
  - PyTorch `Dataset` for loading BOM JSON files.
  - **`__init__(bom_files_dir, metadata_encoder)`:** Walks `bom_files_dir`, loads each `*_bom.json` file, and uses `metadata_encoder.extract_features()` for every part within each BOM. Uses `ThreadPoolExecutor` for parallel file processing.
  - **`__getitem__(idx)`:** Returns the feature tensor for a given index.

### 4. Fusion Module (`src/fusion_module.py`)

- **Purpose:** Combines visual embeddings from the `ImageEncoder` and metadata embeddings from the `MetadataEncoder`.
- **`__init__(visual_dim, metadata_dim, output_dim, fusion_method)`:**
  - `fusion_method` is configurable ("concat" or "weighted").
- **`fuse(visual_embedding, metadata_embedding)`:**
  - **`concat`:** Concatenates the two embeddings and passes them through a linear projection layer to reach `output_dim`.
  - **`weighted`:** Computes a weighted sum. `self.visual_weight` and `self.metadata_weight` are learnable `nn.Parameter`s (initialized to 0.8 and 0.2). If dimensions don't match for weighted sum, metadata embedding is repeated/truncated to match visual embedding dimension.

### 5. Vector Database (`src/vector_database.py`)

- **Purpose:** Manages the FAISS index for efficient similarity search.
- **`__init__(embedding_dim, index_file, metadata_file)`:** Initializes a `faiss.IndexFlatL2` index. Loads an existing index/metadata if files are provided and exist.
- **`add_embeddings(embeddings, file_paths, metadata)`:** Adds a batch of embeddings to the FAISS index. Updates internal dictionaries (`id_to_path`, `path_to_id`, `part_info`) to map FAISS indices to image paths and their metadata (e.g., extracted part name, parent step).
- **`search(query_embedding, k)`:** Performs a k-nearest neighbor search in the FAISS index. Returns distances, indices, paths, associated part info, and recalibrated similarity scores.
  - **Similarity Recalibration:** Converts L2 distances to a more intuitive 0-100% similarity score using a non-linear mapping designed to better differentiate matches (very close distances get high similarity, with a steeper drop-off for larger distances).
- **`build_from_directory(image_encoder, directory, ...)`:** A utility to build the index directly from a directory of images by encoding them on the fly.
- **`save()` / `load()`:** Saves/loads the FAISS index (using `faiss.write_index`/`read_index`) and the metadata dictionary (using `pickle`).
- **`get_stats()`:** Returns statistics like the number of vectors and dimensions.

### 6. Retrieval System (`src/retrieval_system.py`)

- **Purpose:** The central class that integrates all modules and implements the core retrieval logic.
- **`__init__(config_path)`:** Loads configuration, initializes `ImageEncoder`, `VectorDatabase`, `DataProcessor`, `Evaluator`. If metadata is enabled in config, also initializes `MetadataEncoder` and `FusionModule`, and attempts to load a trained autoencoder model.
- **`extract_part_info(image_path)`:** Utility to guess parent STEP ID and part name from an image filename (e.g., "step_id_part_name.png").
- **`ingest_data(dataset_dir)`:** Orchestrates data ingestion using `DataProcessor`. Also copies BOM files to the configured `bom_dir` if metadata is enabled.
- **`build_index(image_dir)`:**
  - If `use_metadata` is true (and components are available and autoencoder is trained), calls `_build_index_with_metadata`.
  - Otherwise, calls `vector_db.build_from_directory` using only the `ImageEncoder`.
  - Saves the index.
- **`_build_index_with_metadata(image_dir)`:** Iterates through images, gets visual embeddings. For each image, tries to find its corresponding BOM, extracts and encodes metadata, fuses visual and metadata embeddings, and adds to `VectorDatabase`. Handles cases where metadata might be missing by using a zero metadata embedding.
- **`retrieve_similar(query_image_path, k, rotation_invariant, num_rotations)`:**
  - Handles single image queries.
  - If `rotation_invariant` is true, uses `rotational_utils.rotation_invariant_search`.
  - Otherwise, encodes the query image (fusing with its metadata if `use_metadata` is true and metadata can be found).
  - Searches the `VectorDatabase`.
  - If `use_metadata` is true and query metadata was found, calls `_rerank_by_size` to adjust scores based on dimensional similarity.
- **`_rerank_by_size(results, query_metadata, size_weight)`:**
  - Extracts size features (dimensions, volume, etc.) for the query part and for each result part (by loading their BOMs).
  - Calculates a `_calculate_size_similarity` score between the query and each result.
  - Combines this size similarity with the initial visual/fused similarity using `size_weight` from config. The combination logic gives boosts for very similar sizes and penalties for very dissimilar sizes.
  - Re-sorts results based on the new combined score.
- **`_extract_size_features(metadata)` / `_calculate_size_similarity(size1, size2)`:** Helper functions for size-based reranking. Similarity calculation uses a combination of volume ratio, average dimension ratio, and bounding box diagonal ratio, converted to a 0-100 score via an exponential decay function.
- **`visualize_results(query_image_path, results, output_path)`:** Generates and saves an image grid of the query and top-k results with similarity scores and part info.
- **`evaluate(query_dir, ground_truth)`:** Uses `Evaluator` to assess performance.
- **`find_part_by_name(part_name, threshold)`:** Implements Stage 1 of part name search: text-based matching using `_normalize_part_name` and `_calculate_name_similarity`.
- **`_normalize_part_name(part_name)` / `_calculate_name_similarity(name1, name2)`:** Helpers for text matching. Similarity uses a weighted sum of Jaccard (char-level), length ratio, and word-level Jaccard.
- **`retrieve_by_part_name(...)`:** Implements the two-stage part name search.
- **`retrieve_by_assembly(assembly_id, k, selected_parts)`:** Implements assembly similarity search logic.
- **`_visualize_assembly_results(...)`:** Custom visualization for assembly search results, showing matched parts comparison.
- **`get_system_info()`:** Returns a dictionary with current system status and configuration.

### 7. Rotational Utilities (`src/rotational_utils.py`)

- **Purpose:** Provides functions to support rotation-invariant visual search.
- **`generate_rotations(image, num_rotations, degrees)`:** Generates multiple PIL Image objects, each a rotated version of the input. Uses `generate_multi_view_angles` for angle generation.
- **`visualize_rotations(image, rotated_images, save_path)`:** Saves a plot of original and rotated images (used by `check_rotations.py`).
- **`generate_multi_view_angles(num_rotations)`:** Generates a set of angles (e.g., evenly spaced + standard orthographic views) to cover different viewpoints.
- **`rotation_invariant_search(...)`:**
  - Takes a query image path.
  - Generates `num_rotations` rotated versions of the image.
  - Encodes each rotated image (fusing with metadata if applicable and found for the original query part).
  - Performs a search with each rotated embedding.
  - Calls `combine_rotation_results` to merge the search results.
- **`combine_rotation_results(all_results, k)`:**
  - Aggregates results from multiple rotations.
  - Calculates an `adjusted_similarity` for each unique part found across rotations. This score considers:
    - Average L2 distance of the part across rotations where it was found.
    - Average rank of the part.
    - Frequency of appearance (how many rotations found this part).
  - The L2 distance is normalized and converted to a base similarity score via exponential decay. This base score is then modified by a rank factor and a frequency boost.
  - This aims to rank parts higher if they are consistently found with low distance, good rank, and across many rotations.
  - Returns the top-k results based on this adjusted similarity.

### 8. Evaluator (`src/evaluator.py`)

- **Purpose:** Evaluates the performance of the retrieval system.
- **`__init__(image_encoder, vector_db, output_dir)`:** Initializes with necessary components.
- **`evaluate_queries(query_images, ground_truth, top_k)`:**
  - For each query image, encodes it and searches the `vector_db`.
  - If `ground_truth` is provided (mapping query image to list of relevant result images), calculates Precision@K and Recall@K for each `k` in `top_k`.
  - Stores per-query and average metrics, and query times.
- **`visualize_retrieval(query_results, max_queries, max_results)`:** Generates image grids for a subset of queries, showing query image and top results with similarity scores and part info.
- **`plot_metrics(results)`:** Generates a plot of Precision@K and Recall@K curves.
- **`save_results(results, output_file)`:** Saves detailed evaluation results to a JSON file.

### 9. Main CLI (`main.py`)

- **Purpose:** Provides the command-line interface to all system functionalities.
- **`parse_args()`:** Uses `argparse` to define commands (`ingest`, `train-autoencoder`, `build`, `retrieve`, `evaluate`, `info`, `list-assembly-parts`) and their respective arguments.
- **`main()`:**
  - Parses arguments.
  - Initializes `RetrievalSystem`.
  - Overrides `use_metadata` in `RetrievalSystem` instance if the command-line flag is passed.
  - Calls the appropriate `RetrievalSystem` methods based on the command.
  - Handles specific logic for `list-assembly-parts` (listing files) and `train-autoencoder` (setting up paths, calling `metadata_encoder.train_autoencoder`).
  - Formats and prints retrieval results to the console.

### 10. GUI (`retrieval_gui.py`)

- **Purpose:** Provides a PyQt5-based graphical interface for the system.
- **`RetrievalGUI` (QMainWindow):** Main application window.
  - Uses a `QSplitter` to create resizable left (controls), center (tabbed image viewer), and right (result thumbnails) panels.
- **`LeftPanel`:** Contains `QComboBox` for operation selection (ingest, train, build, retrieve), dynamic forms for operation parameters, "Run" button, and a `QTextEdit` for logging output.
- **`RightPanel`:** Displays `ResultThumbnail` widgets for retrieved items. Clicking a thumbnail opens it in the `CenterPanel`.
- **`CenterPanel` (`QTabWidget`):** Shows a "Welcome" tab and dynamically adds `ResultTab` widgets for viewed images.
- **`ResultTab`:** Displays a single image using `ZoomableGraphicsView` with zoom controls.
- **`ResultThumbnail`:** A clickable frame showing a small image preview and basic info.
- **`PartsSelectionDialog`:** A dialog that appears during "Full Assembly Query" after listing parts, allowing the user to visually select/deselect parts (with image previews if found) for the query.
- **`ProcessWorker` (from `worker_thread.py`):** A `QThread` used to run `main.py` commands as subprocesses in the background to prevent the GUI from freezing. Emits signals for stdout, stderr, and process completion.
- **Styling:** Uses a custom stylesheet for a more modern look.
- **Functionality:**
  - Constructs and runs `python main.py ...` commands based on GUI inputs.
  - Displays output from `main.py` in the log window.
  - Populates the right panel with result thumbnails by scanning pre-defined result directories (e.g., `data/output/results/image_queries`).
  - Handles part selection for assembly queries by first calling `main.py list-assembly-parts` and then parsing its output to populate the `PartsSelectionDialog`.

## Key Algorithms and Logic

### Part Name Matching
- **Normalization:** Part names are lowercased. Common prefixes and special characters/numbers are removed/standardized.
- **Similarity Metrics:** A weighted combination of:
  1. **Character-level Jaccard Similarity:** Measures overlap in character sets.
  2. **Length Ratio Similarity:** Ratio of shorter name length to longer name length.
  3. **Word-level Jaccard Similarity:** Measures overlap in word sets (for multi-word names).
- **Configuration:** Weights for these metrics and the overall match threshold are in `config.yaml -> text_search`.

### Rotation-Invariant Search
- The query image is rotated multiple times (e.g., 8 rotations, 360/8 = 45-degree steps, or using angles from `generate_multi_view_angles`).
- Each rotated image is encoded and searched against the database.
- `combine_rotation_results` aggregates these multiple result sets.
- **Scoring Logic:** For each unique part found across rotations, an `adjusted_similarity` is calculated based on:
  - **Average Distance:** Lower average L2 distance across rotations is better.
  - **Average Rank:** Lower average rank (appearing higher in lists) is better.
  - **Frequency:** Appearing in more rotation results is better.
  - Distance is normalized and converted to a base similarity (exponential decay). This is then modified by a rank penalty and a frequency boost.

### Assembly Similarity Scoring
- For a query assembly (ID provided by user, parts optionally filtered):
  1. Each part of the query assembly acts as an individual query against the database parts.
  2. For each candidate assembly in the database, the system finds the best matching part for each part of the query assembly.
  3. A `raw_score` (average similarity of these best part-to-part matches) is calculated.
  4. A `coverage_ratio` (fraction of query assembly parts that found a match in the candidate assembly) is calculated.
  5. The final `score = raw_score * (coverage_ratio^2)`. Squaring the coverage ratio heavily penalizes assemblies that only match a few parts of the query assembly.
- Results are ranked by this final score.

### Metadata Autoencoder Training
- The `MetadataEncoder` uses a simple feed-forward autoencoder.
- **Input:** A flat vector of ~46 numerical features extracted from BOM JSONs.
- **Preprocessing:** Features are preprocessed using:
  - NaN/infinity handling.
  - Clipping values to a predefined range (e.g., `min_clip_value = -1e4`, `max_clip_value = 1e4`).
  - Z-score normalization (subtract mean, divide by standard deviation). Means and stds are computed once from the entire training dataset.
- **Loss:** Mean Squared Error (MSE) between the preprocessed input features and the reconstructed features from the decoder.
- **Optimizer:** Adam.
- The trained encoder part is then used to generate metadata embeddings.

### Size-based Reranking
- Applicable if `use_metadata` is true and metadata for the query part is found.
- **Feature Extraction:** Extracts `length`, `width`, `height`, `volume`, `surface_area`, `max_dimension`, `min_dimension`, `mid_dimension` from BOM properties for query and result parts.
- **Size Similarity Calculation (`_calculate_size_similarity`):**
  - Calculates volume ratio (larger/smaller).
  - Calculates dimensional ratios for length, width, height, etc.
  - Calculates bounding box diagonal ratio.
  - A `combined_ratio` is formed (weighted sum, e.g., 0.5 * volume_ratio + 0.3 * avg_dim_ratio + 0.2 * diagonal_ratio).
  - This ratio is converted to a 0-100 similarity score using `100 * math.exp(-0.3 * (combined_ratio - 1))`. Perfect match (ratio 1) gives 100.
  - Boosts for very close matches (ratio < 1.05 ensures >= 95% similarity).
- **Reranking Logic:**
  - The initial similarity (visual or fused) is combined with the size similarity score.
  - The `size_weight` (from `config.yaml`) controls the influence of size similarity.
  - The formula varies:
    - Very similar size (>=95%): Strong boost, minimal penalty for visual score.
    - Good size match (50-95%): Standard weighted average: `original_score * (1 - size_weight) + size_sim * size_weight`.
    - Poor size match (<=5%): Heavier penalty to visual score.
    - Moderate match (5-50%): Standard weighted average.
- Results are re-sorted based on this adjusted similarity.

## Configuration (`config/config.yaml`)

This YAML file is central to customizing the system's behavior without code changes. Key sections:

- **`model`**: Defines the image encoder model (`name`: "dinov2", "vit", "resnet50"), whether to use `pretrained` weights, target `embedding_dim`, and `image_size`.
- **`data`**: Default directories for `input_dir` (raw data), `output_dir` (processed data, results), and `database_dir` (legacy, now `models/` is used more).
- **`indexing`**: Paths for the saved FAISS `index_file` and `metadata_file` (pickle for id-to-path mappings).
- **`text_search`**:
  - `default_threshold`: For part name matching (0-1).
  - `normalize_names`: Boolean.
  - `similarity_weights`: Weights for `jaccard`, `length_ratio`, `word_jaccard` in name similarity calculation.
- **`training`**: (Primarily for batch processing during indexing) `batch_size`, `num_workers`.
- **`evaluation`**: `top_k` values for precision/recall.
- **`metadata`**:
  - `enabled`: Global switch for metadata integration.
  - `embedding_dim`: Latent dimension for the metadata autoencoder.
  - `fusion_method`: "concat" or "weighted".
  - `bom_dir`: Location of processed BOM JSON files.
  - `size_weight`: Importance of size similarity in reranking.
  - Autoencoder settings: `hidden_dims` (list of layer sizes), `model_path` (for saved autoencoder), training `batch_size`, `epochs`, `learning_rate`.

## Extending the System

### Adding a New Image Encoder

1. Modify `src/image_encoder.py`:
   - Add a new `elif self.model_name == 'your_new_model':` block in `__init__`.
   - Load your model, its specific preprocessor/tokenizer, and define its specific `self.transform`.
   - Ensure the forward pass logic correctly extracts the desired embedding (e.g., CLS token, pooled features).
2. Update `config/config.yaml` to allow selecting `'your_new_model'` under `model: name:`.
3. Ensure the `embedding_dim` in config matches your new model's output.

### Modifying Metadata Features

1. Edit `src/metadata_encoder.py`:
   - Change `get_input_dim()` to reflect the new total number of features.
   - Modify `extract_features(metadata)` to extract your new features or change existing ones. Ensure it always returns a list of the correct dimension.
2. If you change the number of input features or their meaning significantly, you **must** retrain the metadata autoencoder (`python main.py train-autoencoder --use-metadata`).

### Implementing a New Fusion Method

1. Edit `src/fusion_module.py`:
   - Add a new `elif self.fusion_method == "your_new_method":` block in `__init__` to define any necessary layers or parameters for your method.
   - Implement the fusion logic within the `fuse(...)` method under a corresponding `elif`.
2. Add `"your_new_method"` as an option to `config.yaml -> metadata -> fusion_method`.

### Using a Different FAISS Index

1. Modify `src/vector_database.py`:
   - In `__init__`, change `self.index = faiss.IndexFlatL2(embedding_dim)` to your desired FAISS index type (e.g., `faiss.IndexHNSWFlat`, `faiss.IndexIVFFlat`).
   - If using an index that requires training (like `IndexIVFFlat`), you'll need to add a training step before adding embeddings. This typically involves:
     ```python
     # Example for IndexIVFFlat
     # quantizer = faiss.IndexFlatL2(embedding_dim)
     # self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist, faiss.METRIC_L2)
     # ...
     # if not self.index.is_trained:
     #     self.index.train(training_vectors_numpy_array)
     ```
   - Adjust search parameters if your new index type requires them.

## Development and Debugging Scripts

Several utility scripts are provided:

- **`check_model.py`:** Prints current image encoder details, PyTorch version, CUDA availability, and DINOv2 availability in transformers. Useful for verifying environment setup.
- **`check_rotations.py --image /path/to/image.png --num-rotations 8`:** Generates and saves a visualization of an image rotated multiple times. Helps debug `rotational_utils.py`.
- **`debug_metadata.py`:** Attempts to import metadata components and initialize them. Helps diagnose import errors or basic instantiation issues with `MetadataEncoder` and `FusionModule`.
- **`setup_env.py`:** Checks for required Python packages and attempts to install them.
- **`test_autoencoder.py [--config ... --model ... --bom_dir ... --visualize]`:** Loads a trained metadata autoencoder, calculates reconstruction loss on test samples, and can visualize the latent space using PCA/t-SNE and analyze feature reconstruction errors.
- **`test_retrieval.py`:** A very basic script to initialize `RetrievalSystem` and perform a sample query if an index and sample query image exist.
- **`train_autoencoder.py [--config ... --bom_dir ... --epochs ...]`:** Standalone script for training the metadata autoencoder. Provides more direct control over training parameters than `main.py train-autoencoder`.

## Troubleshooting for Developers

- **Metadata Issues:**
  - Use `debug_metadata.py` to check for import problems.
  - When training the autoencoder (`train_autoencoder.py` or `main.py train-autoencoder`), check the console output for "Processing BOM files" progress. If it finds 0 files or 0 parts, your `bom_dir` or BOM file structure might be incorrect.
  - The `MetadataEncoder.extract_features()` method is critical. Ensure it correctly parses your BOM JSON structure and handles missing data. Add print statements there to debug.
  - `test_autoencoder.py --visualize` can reveal if the latent space is poorly structured or if certain features are badly reconstructed, indicating issues with feature extraction or autoencoder capacity.
- **Embedding Dimension Mismatches:** Errors like "mat1 and mat2 shapes cannot be multiplied" often point to an embedding dimension mismatch between modules (e.g., `ImageEncoder` output vs. `FusionModule` input vs. `VectorDatabase` dimension). Check `embedding_dim` in `config.yaml` for all relevant components.
- **FAISS Errors:**
  - Ensure `embedding_dim` passed to `VectorDatabase` matches the actual dimension of embeddings being added.
  - If loading an index, ensure it was created with the same FAISS version and dimension.
- **Rotation Invariance Not Working as Expected:**
  - Use `check_rotations.py` to see how your images are being rotated.
  - Examine the `combine_rotation_results` logic in `rotational_utils.py`. Print the scores, ranks, and frequencies for parts to understand how the final `adjusted_similarity` is derived.
- **Slow Performance:**
  - For indexing large datasets, `IndexFlatL2` will become slow for search. Consider switching to an approximate nearest neighbor (ANN) index in FAISS like `IndexHNSWFlat` or `IndexIVFFlat` (requires adding training step for IVF).
  - Ensure batch processing is used where possible (e.g., `image_encoder.encode_batch`).

This guide should provide a solid foundation for understanding and working with the CAD Part Retrieval System.
```