# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a multi-modal CAD part retrieval system that combines visual AI (DINOv2), metadata processing, and vector search. It enables engineers to find similar 3D CAD parts and assemblies using image queries, part name searches, or full assembly comparisons.

## Core Architecture

The system follows a multi-stage pipeline:

1. **Data Ingestion**: Processes STEP file outputs containing part images and BOM JSON files
2. **Metadata Training**: Optional autoencoder training on BOM features (46-dimensional metadata)  
3. **Index Building**: Visual encoding + metadata fusion → FAISS vector database
4. **Retrieval**: Multi-modal search with optional rotation invariance and size-based reranking

**Key Components:**
- `src/retrieval_system.py`: Central orchestrator integrating all modules
- `src/image_encoder.py`: DINOv2/ViT/ResNet50 visual feature extraction  
- `src/metadata_encoder.py`: Autoencoder for BOM metadata (dimensions, topology, materials)
- `src/fusion_module.py`: Combines visual + metadata embeddings (concat/weighted methods)
- `src/vector_database.py`: FAISS-based similarity search with L2 distance
- `src/rotational_utils.py`: Rotation-invariant search across multiple orientations

## Primary Interface - GUI Application

**The main way to interact with the system is through the graphical interface:**

```bash
# Launch the GUI (primary interface)
python retrieval_gui.py
```

The GUI provides all core functionality:
- **Data Ingestion**: Process STEP file outputs
- **Metadata Training**: Train autoencoder on BOM data  
- **Index Building**: Build vector search index
- **Multi-modal Retrieval**: Image query, part name search, full assembly comparison
- **Part Selection**: Interactive part selection for assembly queries
- **Result Visualization**: View and analyze retrieval results

## System Evaluation

### Comprehensive Evaluation Framework
For thesis documentation and publication-ready analysis:

```bash
# RECOMMENDED: Use the wrapper script to handle environment setup
./run_comprehensive_evaluation.sh --thesis-mode --full-benchmark --scalability-test --dataset-dir /path/to/data

# Quick retrieval-only evaluation
./run_comprehensive_evaluation.sh --retrieval-only --part-queries 20 --name-queries 15

# Custom scalability analysis
./run_comprehensive_evaluation.sh --scalability-test --max-queries 50 --k-values 1 5 10 20

# Alternative: Direct Python execution (requires manual environment setup)
MKL_THREADING_LAYER=GNU /home/ngin/miniconda3/envs/f_r/bin/python comprehensive_evaluation.py --thesis-mode
```

**Key Features:**
- **System Benchmarks**: Measures ingest, training, and build performance
- **Retrieval Analysis**: Part image and name search evaluation
- **Scalability Testing**: Performance vs. query load analysis
- **Publication-Ready Graphs**: Automated thesis figure generation (separate PNG files)
- **Statistical Analysis**: Comprehensive metrics and comparisons
- **Figure Explanations**: Automated documentation for each generated figure

### Individual Evaluation Scripts
```bash
# Evaluate part name retrieval performance
python evaluate_name_retrieval.py --num-queries 10 --k 10 --threshold 0.7

# Evaluate part image retrieval performance  
python evaluate_part_retrieval.py --num-queries 5 --k 10 --rotation-invariant
```

### Evaluation Methodology
See `EVALUATION_METHODOLOGY.md` for detailed academic evaluation approach, including:
- Research questions and objectives
- Statistical analysis methods
- Reproducibility measures
- Expected outcomes for thesis documentation

### Generated Thesis Figures
The comprehensive evaluation generates separate PNG files for thesis inclusion:
- `visual_retrieval_similarity_vs_k.png`: DINOv2 visual similarity effectiveness
- `visual_retrieval_time_vs_k.png`: Visual search response time scalability
- `textual_retrieval_score_vs_k.png`: Text matching accuracy analysis
- `textual_retrieval_time_vs_k.png`: Textual search performance characteristics
- `retrieval_performance_comparison.png`: Multi-modal effectiveness comparison

Each figure includes detailed explanations in `figure_explanations/` directory.
See `THESIS_FIGURES_SUMMARY.md` for complete analysis of what each figure proves.

### CLI Commands (For Advanced Users)
```bash
# Environment setup
conda create -n cadretrieval python=3.10
conda activate cadretrieval
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Core operations (typically done via GUI)
python main.py ingest --dataset_dir /path/to/step_outputs
python main.py train-autoencoder --use-metadata --epochs 50
python main.py build --use-metadata
python main.py info  # System status
```

## Configuration

System behavior is controlled via `config/config.yaml`:

- **Model settings**: Image encoder choice (dinov2/vit/resnet50), embedding dimensions
- **Metadata settings**: Autoencoder architecture, fusion method (concat/weighted), feature dimensions
- **Search parameters**: Rotation invariance, part name matching thresholds, similarity weights
- **Data paths**: Input/output directories, model storage locations

## Key Development Patterns

**Multi-modal Pipeline**: Visual embeddings are optionally fused with metadata embeddings before indexing. The `FusionModule` handles dimension alignment and combination strategies.

**Rotation Invariance**: Implemented by generating multiple rotated views of query images, encoding each separately, then combining results with frequency/rank-based scoring in `rotational_utils.py`.

**Assembly Search**: Hierarchical matching where each part in a query assembly becomes a sub-query. Results are aggregated using coverage-weighted similarity scoring.

**Size-based Reranking**: When metadata is available, retrieval results can be reranked using dimensional similarity (volume, bounding box ratios) to improve accuracy for geometrically similar parts.

## File Structure Patterns

- `data/output/images/`: Flattened part images for indexing
- `data/output/bom/`: BOM JSON files for metadata training
- `models/`: Saved FAISS indexes and trained autoencoder models
- Part naming: `{assembly_id}_{part_name}.png` format enables assembly grouping

## Testing and Validation Strategy

**IMPORTANT**: Always create and run comprehensive tests after making changes to ensure new implementations work correctly.

### Testing Approach for Changes
When making modifications or adding features:

1. **Create Custom Tests**: Don't rely on existing test scripts as they may be outdated
2. **Component-Level Testing**: Test individual modules in isolation
3. **Integration Testing**: Verify end-to-end workflows work correctly
4. **Real Data Validation**: Test with actual CAD data and BOM files
5. **Performance Validation**: Ensure changes don't break performance characteristics

### Essential Test Categories

**System Health Checks:**
```bash
# Verify environment setup
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from src.retrieval_system import RetrievalSystem; print('System imports OK')"

# Check model availability
python -c "from transformers import Dinov2Model; print('DINOv2 available')"
```

**Component Testing Framework:**
```bash
# Test image encoder
python -c "from src.image_encoder import ImageEncoder; e = ImageEncoder(); print('Image encoder OK')"

# Test metadata encoder (if using metadata)
python -c "from src.metadata_encoder import MetadataEncoder; print('Metadata encoder OK')"

# Test vector database
python -c "from src.vector_database import VectorDatabase; print('Vector DB OK')"
```

**End-to-End Workflow Testing:**
```bash
# Test minimal workflow with sample data
python main.py info  # System status
python main.py retrieve --query test_image.png --k 5  # Basic retrieval
```

### Legacy Test Scripts (Use with Caution)
Existing scripts may be outdated but can provide reference:
- `check_model.py`: PyTorch/DINOv2 setup verification
- `check_rotations.py`: Rotation visualization debugging  
- `test_retrieval.py`: Basic retrieval system testing
- `test_autoencoder.py`: Metadata autoencoder analysis

## Troubleshooting

### MKL Threading Issues
If you encounter "MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1" errors:

```bash
# Use GNU threading layer
export MKL_THREADING_LAYER=GNU

# Or use the provided wrapper script which handles this automatically
./run_comprehensive_evaluation.sh
```

### Environment Setup
The system requires the `f_r` conda environment:

```bash
# Check available environments
conda env list

# Use correct Python path
/home/ngin/miniconda3/envs/f_r/bin/python
```

### Testing Environment Fix
Use the provided test script to verify setup:

```bash
/home/ngin/miniconda3/envs/f_r/bin/python test_mkl_fix.py
```

## Performance Considerations

- Use batch encoding (`image_encoder.encode_batch`) for large datasets
- FAISS IndexFlatL2 provides exact search but consider IndexHNSWFlat for large-scale approximate search
- Metadata integration adds ~52ms query overhead but improves precision from 82% to 88%
- GPU acceleration provides 10-50× speedup for encoding operations