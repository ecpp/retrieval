# Retrieval Evaluation Scripts

This directory contains scripts for automated evaluation of the retrieval system. The evaluation approach focuses on selecting a variety of inputs, running retrieval queries, and saving the results for manual inspection.

## Overview

Two types of retrieval are evaluated:

1. **Part Retrieval**: Queries using random part images
2. **Part Name Retrieval**: Queries using random part names

Each evaluation script runs queries and saves both the visualizations and structured results in JSON format to allow for manual inspection.

## Scripts

- `evaluate_part_retrieval.py`: Evaluates part image retrieval
- `evaluate_name_retrieval.py`: Evaluates part name retrieval

## How to run

Run individual evaluations:

```
python evaluate_part_retrieval.py --num-queries 10 --k 5
python evaluate_name_retrieval.py --custom-queries "bolt" "gear" "housing"
```

## Common Parameters

All scripts support these common parameters:
- `--seed`: Random seed for reproducibility
- `--output-dir`: Directory to store results

## Part Retrieval Parameters

- `--num-queries`: Number of random part images to query
- `--k`: Number of results per query
- `--rotation-invariant`: Enable rotation-invariant search

## Part Name Retrieval Parameters

- `--num-queries`: Number of random part names to query
- `--k`: Number of results per query
- `--threshold`: Threshold for part name matching (0-1)
- `--rotation-invariant`: Enable rotation-invariant search
- `--custom-queries`: Custom part names to query

## Output Format

Results are stored in the specified output directory, with each run in a timestamped folder:

```
data/evaluation/
├── part_retrieval/
│   └── run_20230521_120000/
│       ├── query_1_results.png
│       ├── query_2_results.png
│       └── evaluation_summary.json
└── name_retrieval/
    └── run_20230521_120030/
        ├── query_1_results.png
        ├── query_2_results.png
        └── evaluation_summary.json
```

The `evaluation_summary.json` file contains structured information about the queries, results, and basic metrics.

## Example Usage

Run part name queries with custom part names:

```
python evaluate_name_retrieval.py --custom-queries "hex bolt" "bearing" "shaft" --k 15
```

Run part image evaluations with rotation invariance:

```
python evaluate_part_retrieval.py --num-queries 20 --k 10 --rotation-invariant
```

## Manual Inspection

After running the evaluation, you can manually inspect the results by:

1. Opening the visualization images in the output directory
2. Reviewing the `evaluation_summary.json` file for metrics
3. Comparing different runs to assess retrieval quality

Since automatic evaluation is challenging for this type of system, the focus is on providing a structured way to run and visualize results for human evaluation.