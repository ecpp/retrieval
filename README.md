# CAD Part Retrieval System

A visual similarity-based retrieval system for 3D CAD parts.

## Overview

This system processes images of CAD parts from STEP files and builds a retrieval system for finding visually similar parts. It uses deep learning models to extract visual features from part images and FAISS for efficient similarity search.

## Features

- Ingests processed STEP file outputs (BOM JSONs and part images)
- Encodes part images into embeddings using state-of-the-art models (ViT, ResNet50, DINOv2 when available)
- Builds an efficient vector index for fast similarity search
- Retrieves visually similar parts given a query image
- Evaluates retrieval performance using standard metrics

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/cad-part-retrieval.git
cd cad-part-retrieval
```

2. Install dependencies:
```
pip install -r requirements.txt
```

## Usage

### 1. Data Ingestion

Process STEP file outputs and prepare them for indexing:

```
python main.py ingest --dataset_dir /path/to/step/outputs
```

### 2. Train Metadata Autoencoder (Optional)

If you plan to use metadata integration, train the autoencoder on BOM files:

```
python main.py train-autoencoder --evaluate
```

You can customize training with additional parameters:

```
python main.py train-autoencoder --bom_dir /path/to/bom/files --batch_size 64 --epochs 100 --lr 0.0001
```

### 3. Build Index

Build the vector index from processed part images:

```
python main.py build
```

To enable metadata integration (requires trained autoencoder):

```
python main.py build --use-metadata
```

### 4. Retrieve Similar Parts

Find parts similar to a query image:

```
python main.py retrieve --query /path/to/query/image.png --k 10 --visualize
```

### 5. Evaluate System

Evaluate the retrieval system's performance:

```
python main.py evaluate --query_dir /path/to/query/images --ground_truth /path/to/ground_truth.json
```

### 6. System Information

Display information about the retrieval system:

```
python main.py info
```

## Configuration

The system can be configured by editing the `config/config.yaml` file:

- Model settings (model type, embedding dimensions, etc.)
- Data paths
- Indexing parameters
- Training and evaluation settings

## Project Structure

```
cad-part-retrieval/
├── config/
│   └── config.yaml
├── data/
│   ├── input/
│   ├── output/
│   └── database/
├── models/
├── src/
│   ├── __init__.py
│   ├── image_encoder.py
│   ├── vector_database.py
│   ├── data_processor.py
│   ├── evaluator.py
│   └── retrieval_system.py
├── main.py
├── requirements.txt
└── README.md
```

## Phase 1 Implementation

The current implementation focuses on visual similarity using image embeddings:

1. Image encoding using DINOv2/ViT/ResNet50
2. Vector database using FAISS for efficient similarity search
3. Basic evaluation metrics (precision, recall)
4. Visualization of retrieval results

Future phases will integrate BOM metadata and implement multimodal retrieval.
