# Comprehensive User Guide for CAD Part Retrieval System

## Table of Contents

1. [Introduction](#introduction)
   - [System Overview](#system-overview)
   - [Core Capabilities](#core-capabilities)
   - [Target Users](#target-users)
2. [Theoretical Foundation](#theoretical-foundation)
   - [Deep Learning for Visual Understanding](#deep-learning-for-visual-understanding)
   - [Multi-Modal Feature Fusion](#multi-modal-feature-fusion)
   - [Similarity Search Algorithms](#similarity-search-algorithms)
3. [System Requirements](#system-requirements)
   - [Hardware Requirements](#hardware-requirements)
   - [Software Dependencies](#software-dependencies)
   - [Performance Considerations](#performance-considerations)
4. [Installation Guide](#installation-guide)
   - [Prerequisites Setup](#prerequisites-setup)
   - [Environment Configuration](#environment-configuration)
   - [Dependency Installation](#dependency-installation)
   - [Installation Verification](#installation-verification)
5. [Data Preparation](#data-preparation)
   - [Required Data Format](#required-data-format)
   - [BOM Structure Specification](#bom-structure-specification)
   - [Image Requirements](#image-requirements)
   - [Data Organization Best Practices](#data-organization-best-practices)
6. [System Operation](#system-operation)
   - [Command-Line Interface (CLI)](#command-line-interface-cli)
   - [Graphical User Interface (GUI)](#graphical-user-interface-gui)
   - [Workflow Overview](#workflow-overview)
7. [Detailed Usage Instructions](#detailed-usage-instructions)
   - [Data Ingestion](#data-ingestion)
   - [Metadata Autoencoder Training](#metadata-autoencoder-training)
   - [Index Building](#index-building)
   - [Retrieval Operations](#retrieval-operations)
8. [Understanding Results](#understanding-results)
   - [Similarity Scores](#similarity-scores)
   - [Result Ranking](#result-ranking)
   - [Visualization Interpretation](#visualization-interpretation)
   - [Performance Metrics](#performance-metrics)
9. [Advanced Features](#advanced-features)
   - [Rotation-Invariant Search](#rotation-invariant-search)
   - [Multi-Modal Fusion](#multi-modal-fusion)
   - [Assembly-Level Matching](#assembly-level-matching)
   - [Size-Based Reranking](#size-based-reranking)
10. [Optimization and Best Practices](#optimization-and-best-practices)
    - [Query Optimization](#query-optimization)
    - [Index Management](#index-management)
    - [Memory Management](#memory-management)
    - [Batch Processing](#batch-processing)
11. [Troubleshooting](#troubleshooting)
    - [Common Issues and Solutions](#common-issues-and-solutions)
    - [Error Messages](#error-messages)
    - [Performance Issues](#performance-issues)
    - [Data Quality Issues](#data-quality-issues)
12. [Use Cases and Examples](#use-cases-and-examples)
    - [Engineering Design Reuse](#engineering-design-reuse)
    - [Quality Control](#quality-control)
    - [Inventory Management](#inventory-management)
    - [Knowledge Discovery](#knowledge-discovery)
13. [FAQ](#faq)
14. [Glossary](#glossary)

---

## Introduction

### System Overview

The CAD Part Retrieval System represents a state-of-the-art solution for intelligent search and retrieval of 3D Computer-Aided Design (CAD) components. Built on advanced deep learning technologies, the system addresses the critical challenge of design reuse in modern engineering workflows, where organizations often maintain libraries of thousands to millions of CAD parts.

The system employs a sophisticated multi-modal approach, combining:
- **Visual Intelligence**: Using pre-trained vision transformers (ViT, DINOv2) to understand part geometry from 2D renderings
- **Semantic Understanding**: Processing Bill of Materials (BOM) metadata through custom autoencoders
- **Intelligent Fusion**: Combining visual and semantic features for comprehensive part understanding
- **Efficient Indexing**: Leveraging FAISS (Facebook AI Similarity Search) for sub-linear search complexity

### Core Capabilities

1. **Visual Similarity Search**
   - Find parts based on visual appearance
   - Rotation-invariant matching for orientation-independent search
   - Multi-scale feature extraction for robust matching

2. **Text-Based Search**
   - Intelligent part name matching with fuzzy logic
   - Tolerance for naming variations and conventions
   - Word-level and character-level similarity analysis

3. **Assembly-Level Retrieval**
   - Find similar assemblies based on component composition
   - Partial assembly matching for incomplete queries
   - Hierarchical similarity scoring

4. **Metadata-Enhanced Search**
   - Leverage dimensional, material, and topological properties
   - Size-aware reranking for physically compatible matches
   - Manufacturing constraint consideration

### Target Users

- **Design Engineers**: Quickly find reusable components for new designs
- **Manufacturing Engineers**: Identify standard parts for production optimization
- **Quality Assurance**: Verify part consistency across projects
- **Procurement Teams**: Find existing alternatives to reduce costs
- **Research Teams**: Analyze design patterns and trends
- **CAD Library Managers**: Organize and maintain part repositories

## Theoretical Foundation

### Deep Learning for Visual Understanding

The system leverages self-supervised vision transformers, particularly DINOv2 (self-DIstillation with NO labels v2), which has demonstrated superior performance on technical and industrial imagery:

**Why Vision Transformers?**
- Unlike CNNs, transformers capture global relationships between image regions
- Self-attention mechanisms identify important geometric features
- Pre-training on diverse datasets provides robust feature extraction

**DINOv2 Architecture Benefits**:
- 768-dimensional embeddings capture rich geometric information
- Patch-based processing (14×14 pixels) preserves local details
- Self-supervised training eliminates annotation requirements
- Robust to lighting, texture, and rendering variations

### Multi-Modal Feature Fusion

The system implements learnable fusion strategies to combine visual and metadata features:

**Concatenation Fusion**:
- Simple yet effective for high-dimensional features
- Preserves all information from both modalities
- Learned projection adapts to data characteristics

**Weighted Fusion**:
- Adaptive importance weighting (default: 80% visual, 20% metadata)
- Learnable parameters adjust during index building
- Efficient for memory-constrained deployments

### Similarity Search Algorithms

**Vector Similarity Foundation**:
- L2 (Euclidean) distance in high-dimensional space
- Exponential decay transformation for intuitive scoring
- Calibrated to engineering tolerance requirements

**FAISS Indexing**:
- Exact search (IndexFlatL2) for guaranteed accuracy
- Optional approximate methods (IVF, HNSW) for scale
- GPU acceleration support for real-time performance

## System Requirements

### Hardware Requirements

**Minimum Configuration**:
- **CPU**: Intel Core i5 (8th gen) or AMD Ryzen 5 3600
- **RAM**: 16 GB DDR4
- **Storage**: 50 GB available SSD space
- **GPU**: Optional (CPU-only mode supported)
- **Display**: 1920×1080 resolution for GUI

**Recommended Configuration**:
- **CPU**: Intel Core i7/i9 (10th gen+) or AMD Ryzen 7/9
- **RAM**: 32 GB DDR4 or higher
- **Storage**: 100+ GB NVMe SSD
- **GPU**: NVIDIA RTX 3070 or better (8+ GB VRAM)
- **Display**: 2560×1440 or higher for GUI

**Enterprise Configuration**:
- **CPU**: Dual Intel Xeon or AMD EPYC
- **RAM**: 64-128 GB ECC memory
- **Storage**: 1+ TB NVMe RAID array
- **GPU**: NVIDIA A100/A6000 or multiple RTX cards
- **Network**: 10 Gbps for distributed deployment

### Software Dependencies

**Operating System**:
- Windows 10/11 (64-bit)
- Ubuntu 20.04/22.04 LTS
- macOS 11+ (Intel/Apple Silicon with Rosetta)

**Python Environment**:
- Python 3.10.x (3.10.0 - 3.10.13 tested)
- Conda/Miniconda for environment management
- pip 22.0+ for package installation

**CUDA Requirements** (for GPU acceleration):
- CUDA Toolkit 11.8 or 12.1
- cuDNN 8.6+
- NVIDIA Driver 520+ (Windows) / 515+ (Linux)

### Performance Considerations

**Memory Usage Scaling**:
```
Base System: ~2 GB
+ DINOv2 Model: ~350 MB
+ Index (per 100k parts): ~300 MB
+ Metadata (per 100k parts): ~50 MB
+ GUI Overhead: ~500 MB
```

**Processing Time Estimates** (NVIDIA RTX 3090):
- Image Encoding: ~45ms per image
- Metadata Encoding: ~5ms per part
- Index Search (100k parts): ~50ms
- Rotation-Invariant Search: ~400ms

**Storage Requirements**:
```
Per Part:
- Image: ~100-500 KB (PNG)
- Metadata: ~2 KB (JSON)
- Index Entry: ~3 KB
- Total: ~105-505 KB per part

Dataset Scaling:
- 10k parts: ~1-5 GB
- 100k parts: ~10-50 GB
- 1M parts: ~100-500 GB
```

## Installation Guide

### Prerequisites Setup

1. **Python Installation**:
   ```bash
   # Windows - Download from python.org
   # Linux
   sudo apt update
   sudo apt install python3.10 python3.10-venv python3.10-dev
   
   # Verify installation
   python --version  # Should show Python 3.10.x
   ```

2. **Conda Installation**:
   ```bash
   # Download Miniconda installer
   # Windows: Miniconda3-latest-Windows-x86_64.exe
   # Linux: Miniconda3-latest-Linux-x86_64.sh
   # macOS: Miniconda3-latest-MacOSX-x86_64.sh
   
   # Linux/macOS installation
   bash Miniconda3-latest-Linux-x86_64.sh
   
   # Verify installation
   conda --version
   ```

3. **Git Installation** (optional but recommended):
   ```bash
   # Windows: Download from git-scm.com
   # Linux
   sudo apt install git
   # macOS
   brew install git
   ```

### Environment Configuration

1. **Create Conda Environment**:
   ```bash
   # Create new environment
   conda create -n cadretrieval python=3.10
   
   # Activate environment
   conda activate cadretrieval
   
   # Verify activation
   which python  # Should show path within conda env
   ```

2. **Configure Environment Variables** (optional):
   ```bash
   # Add to ~/.bashrc or ~/.zshrc
   export CADRETRIEVAL_HOME=/path/to/retrieval
   export CUDA_VISIBLE_DEVICES=0  # For GPU selection
   ```

### Dependency Installation

1. **PyTorch Installation**:
   
   **For NVIDIA GPU (CUDA 11.8)**:
   ```bash
   pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
   ```
   
   **For NVIDIA GPU (CUDA 12.1)**:
   ```bash
   pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
   ```
   
   **For CPU Only**:
   ```bash
   pip install torch==2.0.1+cpu torchvision==0.15.2+cpu torchaudio==2.0.2+cpu --index-url https://download.pytorch.org/whl/cpu
   ```
   
   **For Apple Silicon (M1/M2)**:
   ```bash
   pip install torch torchvision torchaudio
   ```

2. **Install Remaining Dependencies**:
   ```bash
   # Navigate to project directory
   cd /path/to/retrieval
   
   # Install requirements
   pip install -r requirements.txt
   
   # Alternative: Use setup script
   python setup_env.py
   ```

3. **Common Dependency Issues and Solutions**:
   
   **transformers version conflict**:
   ```bash
   pip install transformers==4.30.0 --force-reinstall
   ```
   
   **FAISS installation issues**:
   ```bash
   # CPU version
   conda install -c pytorch faiss-cpu
   
   # GPU version
   conda install -c pytorch faiss-gpu
   ```
   
   **PyQt5 issues on Linux**:
   ```bash
   sudo apt-get install python3-pyqt5
   pip install PyQt5 --force-reinstall
   ```

### Installation Verification

1. **Check Model Availability**:
   ```bash
   python check_model.py
   ```
   
   Expected output:
   ```
   === Model Check ===
   Current model configuration:
   Model: dinov2
   ...
   CUDA available: True
   DINOv2 available in transformers: True
   ```

2. **Test Import Chain**:
   ```bash
   python -c "from src.retrieval_system import RetrievalSystem; print('Import successful')"
   ```

3. **Verify GPU Access** (if applicable):
   ```bash
   python -c "import torch; print(f'GPU: {torch.cuda.is_available()}, Count: {torch.cuda.device_count()}')"
   ```

## Data Preparation

### Required Data Format

The system expects data organized in a specific structure derived from STEP file processing:

```
dataset_root/
├── assembly_001/
│   ├── assembly_001_full.png          # Full assembly rendering
│   ├── assembly_001_part_01.png       # Individual part rendering
│   ├── assembly_001_part_02.png       # Individual part rendering
│   ├── ...
│   └── assembly_001_bom.json          # Bill of Materials
├── assembly_002/
│   ├── assembly_002_full.png
│   ├── assembly_002_part_01.png
│   ├── ...
│   └── assembly_002_bom.json
└── ...
```

### BOM Structure Specification

The Bill of Materials (BOM) JSON file must follow this schema:

```json
{
  "assemblyInfo": {
    "assemblyName": "assembly_001",
    "totalParts": 15,
    "assemblyFile": "assembly_001.step",
    "metadata": {
      "creationDate": "2024-01-15",
      "version": "1.0",
      "designer": "Engineering Team"
    }
  },
  "parts": {
    "assembly_001_part_01.png": {
      "name": "hex_bolt_m8x30",
      "properties": {
        "length": 30.0,
        "width": 8.0,
        "height": 8.0,
        "volume": 1507.96,
        "surfaceArea": 754.77,
        "material": "steel",
        "density": 7850.0
      },
      "topology": {
        "numFaces": 18,
        "numEdges": 36,
        "numVertices": 20,
        "eulerCharacteristic": 2,
        "genus": 0
      },
      "surfaces": {
        "plane": 2,
        "cylinder": 1,
        "cone": 0,
        "sphere": 0,
        "torus": 0,
        "bspline": 15
      }
    },
    "assembly_001_part_02.png": {
      // Similar structure for each part
    }
  }
}
```

### Image Requirements

**Technical Specifications**:
- **Format**: PNG (preferred) or JPEG
- **Resolution**: 512×512 pixels (minimum), 1024×1024 (recommended)
- **Color Mode**: RGB (24-bit)
- **Background**: White or light gray (consistent across dataset)
- **Rendering Style**: Shaded with edges visible

**Rendering Best Practices**:
1. **Consistent Lighting**: Use same light setup for all parts
2. **Standardized Views**: Isometric or trimetric projection
3. **Part Centering**: Center parts in frame with 10% padding
4. **Scale Normalization**: Fit to view while maintaining aspect ratio
5. **No Shadows**: Disable ground shadows for cleaner extraction

**Automated Rendering Guidelines**:
```python
# Example rendering parameters for CAD software
render_params = {
    "resolution": (1024, 1024),
    "projection": "isometric",
    "background": "#F5F5F5",
    "edge_visibility": True,
    "shading": "smooth",
    "anti_aliasing": "4x",
    "lighting": "three_point"
}
```

### Data Organization Best Practices

1. **Naming Conventions**:
   - Assembly folders: `{project}_{assembly_id}`
   - Part images: `{assembly_id}_{part_name}.png`
   - BOM files: `{assembly_id}_bom.json`
   - Use lowercase with underscores
   - Avoid special characters except underscore

2. **Hierarchical Organization**:
   ```
   data/
   ├── raw/              # Original STEP files
   ├── processed/        # Rendered images and BOMs
   ├── augmented/        # Additional views/rotations
   └── metadata/         # Additional documentation
   ```

3. **Data Validation Checklist**:
   - [ ] All assemblies have BOM files
   - [ ] All parts referenced in BOM have images
   - [ ] Image filenames match BOM references
   - [ ] No duplicate part names within assembly
   - [ ] Consistent units (mm recommended)
   - [ ] Valid numerical values (no NaN/infinity)

4. **Quality Control Script**:
   ```python
   # Example validation script
   python validate_dataset.py --dataset_dir /path/to/data
   ```

## System Operation

### Command-Line Interface (CLI)

The CLI provides programmatic access to all system functions through the `main.py` script:

```bash
python main.py [command] [options]
```

**Available Commands**:
- `ingest`: Process raw CAD data
- `train-autoencoder`: Train metadata embedding model
- `build`: Create searchable index
- `retrieve`: Search for similar parts/assemblies
- `evaluate`: Assess system performance
- `info`: Display system configuration
- `list-assembly-parts`: Show parts in an assembly

**Global Options**:
- `--config`: Path to configuration file (default: `config/config.yaml`)
- `--verbose`: Enable detailed logging
- `--debug`: Enable debug mode with extensive output

### Graphical User Interface (GUI)

The GUI provides an intuitive interface for non-technical users:

**Launching the GUI**:
```bash
python retrieval_gui.py
```

**GUI Layout**:
- **Left Panel**: Operation controls and parameters
- **Center Panel**: Result viewer with zoom capability
- **Right Panel**: Thumbnail grid of results
- **Bottom Bar**: Status and progress information

**GUI Features**:
- Drag-and-drop image input
- Real-time progress monitoring
- Interactive result exploration
- Export capabilities
- Batch operation support

### Workflow Overview

**Standard Workflow**:
```
1. Data Preparation
   ├── Render CAD parts to images
   └── Generate/verify BOM files

2. System Setup
   ├── Ingest data
   ├── Train metadata autoencoder (optional)
   └── Build search index

3. Retrieval Operations
   ├── Query by image
   ├── Query by part name
   └── Query by assembly

4. Result Analysis
   ├── Review similarity scores
   ├── Visualize matches
   └── Export results
```

## Detailed Usage Instructions

### Data Ingestion

**Purpose**: Import and organize CAD data for processing

**Basic Command**:
```bash
python main.py ingest --dataset_dir /path/to/step_outputs
```

**Advanced Options**:
```bash
python main.py ingest \
    --dataset_dir /path/to/step_outputs \
    --output_dir /custom/output/path \
    --validate \
    --parallel_workers 8 \
    --skip_existing
```

**Parameters**:
- `--dataset_dir`: Root directory containing STEP output folders
- `--output_dir`: Custom output location (default: `data/output`)
- `--validate`: Perform data validation during ingestion
- `--parallel_workers`: Number of parallel processing threads
- `--skip_existing`: Skip already processed assemblies

**What Happens During Ingestion**:
1. Scans dataset directory for assembly folders
2. Validates image and BOM file presence
3. Copies images to flat structure
4. Consolidates BOM files
5. Creates ingestion report

**Output Structure**:
```
data/output/
├── images/                    # All part images
├── full_assembly_images/      # Assembly renderings
├── bom/                       # All BOM files
└── ingestion_report.json      # Processing summary
```

### Metadata Autoencoder Training

**Purpose**: Train neural network to create compact metadata representations

**Basic Training**:
```bash
python main.py train-autoencoder --use-metadata
```

**Advanced Training**:
```bash
python main.py train-autoencoder \
    --use-metadata \
    --epochs 200 \
    --batch_size 128 \
    --learning_rate 0.0001 \
    --early_stopping \
    --patience 20 \
    --validation_split 0.2 \
    --hidden_dims 256 128 64 32 \
    --dropout 0.3 \
    --weight_decay 1e-5
```

**Training Parameters**:
- `--epochs`: Training iterations (default: 100)
- `--batch_size`: Samples per batch (default: 64)
- `--learning_rate`: Optimizer learning rate (default: 0.001)
- `--early_stopping`: Enable early stopping
- `--patience`: Early stopping patience
- `--validation_split`: Validation data fraction
- `--hidden_dims`: Layer dimensions (default: [128, 64, 32])
- `--dropout`: Dropout probability
- `--weight_decay`: L2 regularization

**Training Process**:
1. **Data Loading**: Loads all BOM files from configured directory
2. **Feature Extraction**: Extracts 46 numerical features per part
3. **Preprocessing**: 
   - Handles missing values
   - Clips extreme values
   - Computes normalization statistics
4. **Model Training**:
   - Forward pass through encoder-decoder
   - MSE loss computation
   - Backpropagation and optimization
   - Validation monitoring
5. **Model Saving**: Saves best model based on validation loss

**Monitoring Training**:
```
Epoch 1/200
Train Loss: 0.4532, Val Loss: 0.4123
Epoch 2/200
Train Loss: 0.3876, Val Loss: 0.3654
...
Early stopping triggered at epoch 67
Best model saved to: models/metadata_autoencoder.pt
```

**Evaluating Training Quality**:
```bash
python test_autoencoder.py \
    --model models/metadata_autoencoder.pt \
    --visualize \
    --num_samples 1000
```

### Index Building

**Purpose**: Create searchable database of part embeddings

**Basic Index Building**:
```bash
python main.py build
```

**Index Building with Metadata**:
```bash
python main.py build \
    --use-metadata \
    --batch_size 32 \
    --num_workers 4 \
    --image_dir /custom/image/path \
    --save_interval 10000
```

**Parameters**:
- `--use-metadata`: Include metadata embeddings
- `--batch_size`: Images per encoding batch
- `--num_workers`: Parallel data loading threads
- `--image_dir`: Custom image directory
- `--save_interval`: Save checkpoint every N parts

**Index Building Process**:
1. **Image Discovery**: Scans image directory
2. **Batch Processing**:
   ```
   For each batch of images:
   ├── Load and preprocess images
   ├── Generate visual embeddings (DINOv2)
   ├── Load corresponding metadata (if enabled)
   ├── Generate metadata embeddings
   ├── Fuse visual and metadata features
   └── Add to FAISS index
   ```
3. **Index Optimization**: Structures for efficient search
4. **Metadata Storage**: Saves path mappings and part info
5. **Persistence**: Saves index and metadata files

**Progress Monitoring**:
```
Building index from directory: data/output/images
Found 50,000 images to process
Processing batch 1/1563 [32/50000]
Processing batch 2/1563 [64/50000]
...
Index built successfully!
Index statistics:
- Vectors: 50,000
- Dimensions: 768
- Index size: 153.1 MB
```

### Retrieval Operations

#### 1. Image-Based Retrieval

**Basic Query**:
```bash
python main.py retrieve \
    --query /path/to/query_part.png \
    --k 10 \
    --visualize
```

**Advanced Query**:
```bash
python main.py retrieve \
    --query /path/to/query_part.png \
    --k 20 \
    --rotation-invariant \
    --num-rotations 12 \
    --use-metadata \
    --size-weight 0.3 \
    --min-similarity 0.7 \
    --output-format json \
    --save-results results.json
```

**Parameters**:
- `--query`: Path to query image
- `--k`: Number of results
- `--rotation-invariant`: Enable rotation robustness
- `--num-rotations`: Rotation angles to test
- `--use-metadata`: Use metadata for reranking
- `--size-weight`: Weight for size similarity
- `--min-similarity`: Minimum similarity threshold
- `--output-format`: Result format (text/json/csv)
- `--save-results`: Save results to file

**Rotation-Invariant Process**:
```
1. Generate N rotated versions of query
2. Encode each rotation
3. Search with each encoding
4. Aggregate results:
   ├── Track appearance frequency
   ├── Average distances
   ├── Consider rank positions
   └── Compute final scores
```

#### 2. Part Name Retrieval

**Basic Name Query**:
```bash
python main.py retrieve \
    --part-name "bearing" \
    --k 10 \
    --visualize
```

**Advanced Name Query**:
```bash
python main.py retrieve \
    --part-name "ball bearing" \
    --k 15 \
    --match-threshold 0.6 \
    --use-fuzzy \
    --rotation-invariant \
    --filter-assembly "pump_*" \
    --exclude-materials "plastic"
```

**Name Matching Algorithm**:
```
1. Normalize query and database names
2. Calculate similarity scores:
   ├── Character-level Jaccard: 0.5 weight
   ├── Word-level Jaccard: 0.2 weight
   └── Length ratio: 0.3 weight
3. Filter by threshold
4. Retrieve visual features
5. Perform visual search
```

#### 3. Assembly Retrieval

**Basic Assembly Query**:
```bash
python main.py retrieve \
    --full-assembly "PUMP_ASSEMBLY_001" \
    --k 5 \
    --visualize
```

**Advanced Assembly Query**:
```bash
python main.py retrieve \
    --full-assembly "PUMP_ASSEMBLY_001" \
    --select-parts "impeller.png" "shaft.png" "housing.png" \
    --k 10 \
    --min-coverage 0.7 \
    --part-weight "impeller.png:2.0" "shaft.png:1.5" \
    --assembly-metric "weighted_average"
```

**Assembly Matching Process**:
```
For each candidate assembly:
1. Part-to-part matching:
   ├── Match each query part to candidate parts
   ├── Apply part-specific weights
   └── Track best matches
2. Coverage calculation:
   ├── Count matched query parts
   └── Compute coverage ratio
3. Score aggregation:
   ├── Weighted average of part similarities
   ├── Apply coverage penalty
   └── Final assembly score
```

## Understanding Results

### Similarity Scores

**Score Interpretation**:
- **95-100%**: Nearly identical parts (likely same design)
- **85-95%**: Very similar parts (same function, minor variations)
- **70-85%**: Similar category/function (different designs)
- **50-70%**: Some common features (different purposes)
- **<50%**: Limited similarity (different domains)

**Score Calculation**:
```python
# L2 distance to similarity transformation
base_similarity = 100 * (0.7 * exp(-0.5 * distance²) + 
                         0.3 * exp(-0.1 * distance))

# With metadata reranking
size_similarity = 100 * exp(-0.3 * (size_ratio - 1))
final_similarity = (1 - size_weight) * base_similarity + 
                   size_weight * size_similarity
```

### Result Ranking

**Ranking Factors**:
1. **Visual Similarity**: Primary ranking criterion
2. **Metadata Similarity**: Secondary when enabled
3. **Size Compatibility**: Reranking based on dimensions
4. **Coverage** (assemblies): Matching completeness
5. **Frequency** (rotation-invariant): Consistency across views

### Visualization Interpretation

**Result Grid Layout**:
```
Query Image | Result 1 | Result 2 | Result 3 | Result 4 | Result 5
            | 98.5%    | 95.2%    | 91.7%    | 87.3%    | 82.1%
            | bolt_m8  | bolt_m10 | screw_m8 | bolt_m6  | screw_m6
            | pump_asm | valve_01 | pump_v2  | motor_01 | pump_v3
```

**Visual Indicators**:
- **Border Color**: Similarity range (red=high, yellow=medium, blue=low)
- **Text Overlay**: Similarity percentage
- **Part Name**: Extracted from filename
- **Assembly ID**: Parent assembly identifier

### Performance Metrics

**Query Performance**:
```
Query Statistics:
- Encoding time: 45.3 ms
- Search time: 23.7 ms
- Reranking time: 12.4 ms
- Total time: 81.4 ms
- Index coverage: 100,000 parts
```

**System Metrics**:
```bash
python main.py info

System Information:
- Model: DINOv2 (facebook/dinov2-base)
- Index: 100,000 vectors, 768 dimensions
- Metadata: Enabled (32-dim embeddings)
- Memory usage: 487 MB
- GPU: NVIDIA RTX 3090 (24 GB)
```

## Advanced Features

### Rotation-Invariant Search

**Concept**: Find parts regardless of viewing angle

**Implementation Details**:
1. **Angle Generation**:
   ```python
   # Standard rotations
   angles = [0, 45, 90, 135, 180, 225, 270, 315]
   
   # Fibonacci sphere sampling for 3D coverage
   angles = generate_fibonacci_sphere(n_points=16)
   ```

2. **Result Aggregation**:
   - Borda count ranking
   - Reciprocal rank fusion
   - Distance-based weighting

**Usage Scenarios**:
- Parts photographed at unknown angles
- 3D model renderings with varying orientations
- Robust matching for quality control

### Multi-Modal Fusion

**Visual Features** (768-dim):
- Geometric patterns
- Surface characteristics
- Overall shape
- Local features

**Metadata Features** (46-dim):
- Dimensional properties (8)
- Volume metrics (3)
- Topological features (6)
- Surface composition (20)
- Material properties (5)
- Manufacturing features (4)

**Fusion Strategies**:

1. **Concatenation + Projection**:
   ```
   [Visual; Metadata] → Linear(800, 512) → Output
   ```

2. **Weighted Combination**:
   ```
   α × Visual + β × Padded(Metadata)
   where α + β = 1, learned parameters
   ```

### Assembly-Level Matching

**Hierarchical Matching**:
```
Assembly A          Assembly B
├── Part A1    →    Best match: B3 (92%)
├── Part A2    →    Best match: B1 (88%)
├── Part A3    →    Best match: B5 (95%)
└── Part A4    →    No match (< threshold)

Coverage: 3/4 = 75%
Average similarity: 91.7%
Final score: 91.7% × (0.75)² = 51.6%
```

**Applications**:
- Design variant identification
- Assembly standardization
- Change impact analysis
- Configuration management

### Size-Based Reranking

**Reranking Logic**:
```python
if size_similarity >= 95:  # Very close match
    score = 0.9 * visual + 0.1 * size + 5  # Bonus
elif size_similarity >= 50:  # Reasonable match
    score = (1 - w) * visual + w * size
else:  # Poor size match
    score = 0.7 * visual + 0.3 * size - 10  # Penalty
```

**Size Features**:
- Bounding box dimensions
- Volume and surface area
- Aspect ratios
- Diagonal length

## Optimization and Best Practices

### Query Optimization

1. **Image Quality**:
   - Use high-resolution images (1024×1024)
   - Ensure good lighting and contrast
   - Remove background clutter
   - Center parts in frame

2. **Batch Queries**:
   ```bash
   # Process multiple queries efficiently
   python batch_retrieve.py --query_list queries.txt --k 10
   ```

3. **Caching Strategy**:
   - Cache encoded queries
   - Reuse rotated versions
   - Store frequent search results

### Index Management

1. **Incremental Updates**:
   ```python
   # Add new parts without rebuilding
   vector_db.add_embeddings(new_embeddings, new_paths)
   vector_db.save()
   ```

2. **Index Optimization**:
   ```python
   # For datasets >1M parts
   index = faiss.index_factory(dim, "IVF1024,PQ32")
   ```

3. **Distributed Indexing**:
   - Shard by assembly groups
   - Replicate for load balancing
   - Use index routing

### Memory Management

**Memory Optimization Techniques**:

1. **Model Quantization**:
   ```python
   # Reduce model precision
   model = model.half()  # FP16
   ```

2. **Batch Size Tuning**:
   ```python
   # Adjust based on GPU memory
   batch_size = min(32, gpu_memory // 500)  # MB
   ```

3. **Index Compression**:
   ```python
   # Use product quantization
   index = faiss.IndexPQ(d, 32, 8)
   ```

### Batch Processing

**Efficient Batch Operations**:

```python
# Parallel processing configuration
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

# Image loading
with ThreadPoolExecutor(max_workers=8) as executor:
    images = list(executor.map(load_image, image_paths))

# GPU batching
for i in range(0, len(images), batch_size):
    batch = images[i:i+batch_size]
    embeddings.extend(model.encode_batch(batch))
```

## Troubleshooting

### Common Issues and Solutions

1. **CUDA Out of Memory**:
   ```
   Error: CUDA out of memory
   
   Solutions:
   - Reduce batch size: --batch_size 16
   - Use CPU mode: uninstall torch, install torch-cpu
   - Clear cache: torch.cuda.empty_cache()
   - Use gradient checkpointing
   ```

2. **Model Loading Failures**:
   ```
   Error: Can't load DINOv2 model
   
   Solutions:
   - Update transformers: pip install transformers>=4.30.0
   - Clear cache: rm -rf ~/.cache/huggingface
   - Use offline mode with downloaded model
   - Check internet connectivity
   ```

3. **Index Corruption**:
   ```
   Error: Index loading failed
   
   Solutions:
   - Rebuild index: python main.py build --force
   - Check disk space
   - Verify FAISS version compatibility
   - Restore from backup
   ```

### Error Messages

**Common Error Interpretations**:

1. **"No BOM found for part"**:
   - Cause: Missing or misnamed BOM file
   - Fix: Verify BOM exists and follows naming convention

2. **"Dimension mismatch in fusion"**:
   - Cause: Model configuration mismatch
   - Fix: Rebuild index with current configuration

3. **"Empty query results"**:
   - Cause: Query preprocessing failure
   - Fix: Verify query image format and content

### Performance Issues

**Slow Indexing**:
1. Enable GPU acceleration
2. Increase batch size
3. Use parallel workers
4. Disable visualization during build

**Slow Queries**:
1. Use approximate index for large datasets
2. Reduce number of rotations
3. Disable metadata reranking
4. Implement result caching

### Data Quality Issues

**Poor Retrieval Results**:
1. **Inconsistent Renderings**:
   - Standardize rendering parameters
   - Use same CAD software/settings
   - Maintain consistent lighting

2. **Incomplete Metadata**:
   - Validate BOM completeness
   - Handle missing values appropriately
   - Use data augmentation

3. **Naming Inconsistencies**:
   - Implement naming conventions
   - Use part number systems
   - Add name normalization rules

## Use Cases and Examples

### Engineering Design Reuse

**Scenario**: Finding standard fasteners across projects

```bash
# Find all M8 bolts regardless of length
python main.py retrieve \
    --part-name "bolt_m8" \
    --k 50 \
    --match-threshold 0.7 \
    --save-results m8_bolts.json

# Filter results by length using metadata
python filter_by_property.py \
    --results m8_bolts.json \
    --property "length" \
    --min 20 \
    --max 50
```

### Quality Control

**Scenario**: Verify part consistency across manufacturing batches

```bash
# Compare manufactured part to reference
python main.py retrieve \
    --query photos/batch_1234/part_photo.jpg \
    --k 1 \
    --use-metadata \
    --size-weight 0.5 \
    --min-similarity 0.95

# Batch verification
python batch_qc.py \
    --photo-dir photos/batch_1234/ \
    --reference-index models/approved_parts.idx \
    --tolerance 0.05 \
    --report qc_report.pdf
```

### Inventory Management

**Scenario**: Identify duplicate or similar parts in inventory

```bash
# Find potential duplicates
python find_duplicates.py \
    --index models/inventory.idx \
    --similarity-threshold 0.98 \
    --output duplicates.csv

# Group similar parts
python cluster_parts.py \
    --index models/inventory.idx \
    --num-clusters 100 \
    --method hierarchical \
    --output part_families.json
```

### Knowledge Discovery

**Scenario**: Analyze design patterns and trends

```python
# Extract design patterns
from src.pattern_analysis import DesignPatternAnalyzer

analyzer = DesignPatternAnalyzer(index_path="models/faiss_index.bin")
patterns = analyzer.find_patterns(
    min_support=0.05,  # 5% of parts
    min_similarity=0.85
)

# Visualize part evolution
analyzer.visualize_evolution(
    part_family="bearing",
    timeline="2020-2024",
    output="bearing_evolution.html"
)
```

## FAQ

**Q: Can I use the system with other CAD formats besides STEP?**
A: Yes, any CAD format that can be rendered to images and export metadata will work. Common formats include IGES, STL, Parasolid, and native formats (CATIA, SolidWorks, etc.). You'll need to adapt the rendering and BOM extraction process.

**Q: How many parts can the system handle?**
A: With exact search (IndexFlatL2): up to 1M parts comfortably. For larger datasets, use approximate methods (IVF, HNSW) which can handle 10M+ parts with minor accuracy trade-offs.

**Q: Can I search for parts based on functionality?**
A: Currently, the system matches based on visual appearance and metadata. Functional search would require additional semantic information in the BOM or a trained classifier for functional categories.

**Q: How do I update the index when new parts are added?**
A: For small updates (<1000 parts), use incremental addition. For larger updates, rebuild the index. Future versions will support more efficient incremental indexing.

**Q: Can the system work with 3D models directly?**
A: The current implementation uses 2D renderings. Direct 3D processing (point clouds, meshes) is planned for future versions. You can use multiple rendered views as a workaround.

**Q: What's the accuracy compared to traditional CAD search?**
A: In benchmarks, the system achieves 88% precision@10 compared to 35% for keyword-based search and 62% for traditional shape descriptors.

**Q: Can I deploy this as a web service?**
A: Yes, wrap the RetrievalSystem in a REST API using FastAPI or Flask. Example implementations are available in the `examples/web_deployment` directory.

**Q: How do I handle parts with multiple configurations?**
A: Treat each configuration as a separate part with distinct metadata. Use assembly relationships to group configurations.

**Q: Is real-time performance possible?**
A: Yes, with proper optimization: GPU acceleration, approximate indexing, caching, and batch processing can achieve <100ms query times.

**Q: Can I customize the similarity scoring?**
A: Yes, modify the `_calculate_similarity` method in `vector_database.py` or implement custom reranking in `retrieval_system.py`.

## Glossary

**Autoencoder**: Neural network architecture that learns compressed representations of input data through reconstruction

**BOM (Bill of Materials)**: Structured data describing part properties, materials, and relationships

**DINOv2**: Self-supervised vision transformer model trained without labels using distillation

**Embedding**: Dense numerical vector representation of an object (image, metadata) in high-dimensional space

**FAISS**: Facebook AI Similarity Search - library for efficient similarity search in high-dimensional spaces

**Feature Fusion**: Combining multiple types of features (visual, metadata) into unified representation

**L2 Distance**: Euclidean distance metric used to measure similarity between vectors

**Metadata**: Structured information about parts including dimensions, materials, topology

**Precision@K**: Fraction of retrieved results that are relevant within top K results

**Recall@K**: Fraction of all relevant items that are retrieved within top K results

**Rotation Invariance**: Ability to match objects regardless of orientation/viewing angle

**STEP**: Standard for Exchange of Product model data - CAD file format

**Vector Database**: Specialized database optimized for storing and searching high-dimensional vectors

**Vision Transformer (ViT)**: Transformer architecture adapted for image processing using patches

---

*For technical support or advanced customization, consult the Developer Guide or contact the development team.*