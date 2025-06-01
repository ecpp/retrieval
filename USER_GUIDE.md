# User Guide for CAD Part Retrieval System

Welcome to the CAD Part Retrieval System! This guide will help you set up and use the system to find similar CAD parts and assemblies.

## What is the Assembly Retrieval System?

The CAD Part Retrieval System is a powerful tool designed to help engineers, designers, and researchers quickly locate relevant CAD models from a database. You can search for parts based on their visual appearance (by providing an image), their name, or even find similar entire assemblies. The system uses artificial intelligence to understand and compare the features of CAD models.

## Key Concepts

- **Ingestion:** This is the first step where you feed the system your CAD data. The system processes part images and associated information (like Bill of Materials or BOMs).
- **Indexing (`build`):** After ingesting data, the system creates an "index." Think of this like an intelligent catalog. It analyzes the visual features (and optionally, metadata from BOMs) of each part and stores them in a way that allows for very fast searching. If you're using metadata, you'll also need to train a special "metadata autoencoder" model first.
- **Querying (`retrieve`):** Once the index is built, you can ask the system to find parts or assemblies. This is called querying. You can query with:
  - An image of a part.
  - The name of a part.
  - The ID of an assembly.
- **Metadata:** Additional information about parts, often found in BOM files (e.g., dimensions, material, function). Using metadata can make searches more accurate.
- **Rotation Invariance:** This feature allows the system to find a part even if your query image shows it from a different angle than the images in the database.
- **GUI (Graphical User Interface):** A user-friendly window-based application to interact with the system instead of typing commands.

## Prerequisites

- **Python:** Version 3.10 is recommended. You can download it from the [official Python website](https://www.python.org/downloads/).
- **Conda:** A package manager that simplifies installing Python and its libraries. Download it from the [Anaconda website](https://www.anaconda.com/products/distribution).
- **CAD Data:** You'll need a collection of CAD parts you want to index. This typically involves:
  - Images of individual parts (e.g., PNG, JPG).
  - Optionally, Bill of Materials (BOM) files in JSON format for each assembly, containing metadata about the parts.

## Installation

1. **Download the System:**
   If you have git, clone the repository. Otherwise, download the source code.

   ```bash
   # Example using git:
   git clone <your-repository-url>
   cd <repository-name>
   ```
2. **Set up Conda Environment:**
   Open your terminal or Anaconda Prompt and run:

   ```bash
   conda create -n cadretrieval python=3.10
   conda activate cadretrieval
   ```

   This creates a dedicated environment named `cadretrieval` and activates it.
3. **Install Libraries:**
   First, install PyTorch. If you have an NVIDIA GPU and want to use it for faster processing, find the correct PyTorch installation command for your CUDA version on the [PyTorch website](https://pytorch.org/get-started/locally/).
   Example for CUDA 11.8:

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

   If you don't have an NVIDIA GPU or don't want to use it, you can install the CPU-only version:

   ```bash
   pip install torch torchvision torchaudio
   ```

   Then, install the other required libraries:

   ```bash
   pip install -r requirements.txt
   ```

   You can also try running the provided environment setup script:

   ```bash
   python setup_env.py
   ```

## How to Use the System

You can use the system through the Command-Line Interface (CLI) by typing commands in your terminal, or through the Graphical User Interface (GUI).

### Using the Command-Line Interface (CLI)

Navigate to the system's directory in your terminal. All commands start with `python main.py`.

**Step 1: Ingest Your Data**
This step processes your raw CAD data (images and BOMs).

```bash
python main.py ingest --dataset_dir /path/to/your/step_outputs
```

- Replace `/path/to/your/step_outputs` with the path to your main data folder. This folder should contain subfolders, each representing a processed STEP file and containing part images and a `*_bom.json` file.
- This command will copy images to `data/output/images/` and BOMs to `data/output/bom/` (by default).

**Step 2 (Optional, for Metadata): Train Metadata Autoencoder**
If you want to use metadata from BOM files for more accurate searches, you need to train the metadata autoencoder.

```bash
python main.py train-autoencoder --use-metadata
```

- This will use BOM files from the directory specified in `config/config.yaml` (default: `data/output/bom`).
- You can specify training parameters like `--epochs`, `--batch_size`, `--lr` if needed.
- The trained model will be saved (default: `models/metadata_autoencoder.pt`).

**Step 3: Build the Index**
This creates the searchable database of your parts.

```bash
python main.py build
```

- If you trained the metadata autoencoder and want to use metadata in the index, add the `--use-metadata` flag:
  ```bash
  python main.py build --use-metadata
  ```
- This command will process images from `data/output/images/` (by default) and create/update `models/faiss_index.bin` and `models/index_metadata.pkl`.

**Step 4: Retrieve Similar Parts or Assemblies**

Now you can search!

- **Search by Image:**

  ```bash
  python main.py retrieve --query /path/to/your/query_image.png --k 5 --visualize
  ```

  - Replace `/path/to/your/query_image.png` with your image.
  - `--k 5`: Retrieve top 5 results.
  - `--visualize`: Show a window with the query and results.
  - Add `--rotation-invariant` for better matching if the part's orientation might be different.
  - Add `--use-metadata` if you built the index with metadata and want to leverage it for retrieval.
- **Search by Part Name:**

  ```bash
  python main.py retrieve --part-name "fastener" --k 5 --visualize
  ```

  - Replace `"fastener"` with the name of the part you're looking for.
  - `--match-threshold 0.7`: Adjust how strictly the name should match (0.0 to 1.0).
  - Also supports `--rotation-invariant` and `--use-metadata`.
- **Search for Similar Assemblies:**

  ```bash
  python main.py retrieve --full-assembly "ASSEMBLY_ID_001" --k 3 --visualize
  ```

  - Replace `"ASSEMBLY_ID_001"` with the ID of your query assembly.
  - To use only specific parts from "ASSEMBLY_ID_001" for comparison:

    ```bash
    python main.py retrieve --full-assembly "ASSEMBLY_ID_001" --select-parts "part_x.png" "part_y.png" --k 3 --visualize
    ```

    (Use `python main.py list-assembly-parts --assembly-id "ASSEMBLY_ID_001"` to see available part filenames).

**Other Useful Commands:**

- **Evaluate System Performance:**

  ```bash
  python main.py evaluate --query_dir /path/to/test_queries --ground_truth /path/to/ground_truth.json
  ```

  (Requires a set of test queries and a ground truth file defining correct matches).
- **Get System Information:**

  ```bash
  python main.py info
  ```

  (Shows current model, index size, etc.).
- **List Parts in an Assembly:**

  ```bash
  python main.py list-assembly-parts --assembly-id "ASSEMBLY_ID_001"
  ```

### Using the Graphical User Interface (GUI)

For a more visual approach, you can use the GUI.

1. **Launch the GUI:**
   In your terminal (with the `cadretrieval` conda environment activated), run:

   ```bash
   python retrieval_gui.py
   ```
2. **Using the GUI:**

   - **Left Panel (Operations):**
     - Select an operation (e.g., "ingest", "train autoencoder", "build", "retrieve").
     - Fill in the required parameters (e.g., dataset directory for ingest, query image for retrieve).
     - Click "Run".
   - **Output Log:** Shows messages and progress from the running operations.
   - **Center Panel (Results Viewer):** When you click on a result thumbnail, the image will be displayed here in a new tab with zoom controls.
   - **Right Panel (Result Thumbnails):** Shows thumbnails of retrieved parts/assemblies. Click a thumbnail to view it larger in the center panel.

   **GUI Operations:**

   - **Ingest:** Select "ingest", specify the "Dataset Directory", click "Run".
   - **Train Autoencoder:** Select "train autoencoder", specify "BOM Directory" (if not default), adjust parameters if needed, check "Enable metadata integration" if you want to use it, click "Run".
   - **Build Index:** Select "build", specify "Image Directory" (if not default), check "Use metadata for indexing" if applicable, click "Run".
   - **Retrieve:**
     - Select "retrieve".
     - Choose "Query Type" (Image, Part Name, or Full Assembly).
     - **Image Query:** Provide "Query Image" path.
     - **Part Name Query:** Enter "Part Name".
     - **Full Assembly Query:** Enter "Assembly ID".
       - Click "List Assembly Parts" to see parts in that assembly. A dialog will appear allowing you to select which parts to include in the query.
     - Adjust common options like "Number of Results (k)", "Visualize Results", "Enable Rotation-Invariant Search", "Use Metadata for Retrieval", "Match Threshold".
     - Click "Run".

## Example Output (CLI)

When you run a retrieval query, the output in the terminal might look like this:

```
Top 5 results:
Rank | Part                           | STEP File          | Part Name          | Similarity
------------------------------------------------------------------------------------------
1.   | assemblyX_widget_modelA.png    | assemblyX          | widget_modelA      | 98.7%
2.   | assemblyY_widget_variantB.png  | assemblyY          | widget_variantB    | 95.2%
...
```

If `--visualize` is used, an image window will also pop up showing the query and the results.

For more advanced issues, refer to the `README.md` or `DEVELOPER_GUIDE.md`.
