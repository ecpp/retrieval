# Part Name Search Feature

This feature allows you to search for CAD parts by their name and then retrieve visually similar parts.

## Overview

The part name search feature works in two steps:
1. First, it searches the database for a part that best matches the provided part name
2. Then, it uses the found part's image as the query for visual similarity search

This approach allows you to find parts even when you don't have a query image available.

## How It Works

### Part Name Matching

The system uses a sophisticated text similarity algorithm to find the best match for a given part name:

- **Name normalization**: Removes special characters, numbers, and common prefixes to standardize part names
- **Multiple similarity metrics**:
  - Jaccard similarity (character level)
  - Length ratio
  - Word-level Jaccard similarity
- **Configurable similarity threshold**: Control how strict the matching should be

### Usage Examples

```bash
# Basic part name search
python main.py retrieve --part-name "screw" --visualize

# More specific part name with lower threshold
python main.py retrieve --part-name "m6_bolt" --match-threshold 0.5 --k 10

# Advanced search with rotation invariance
python main.py retrieve --part-name "circular connector" --rotation-invariant --num-rotations 12
```

## Configuration

The part name search is configurable in the `config/config.yaml` file:

```yaml
text_search:
  default_threshold: 0.7  # Default similarity threshold for text search (0-1)
  normalize_names: true  # Whether to normalize part names before matching
  # Weights for different similarity metrics
  similarity_weights:
    jaccard: 0.4         # Weight for character-level Jaccard similarity
    length_ratio: 0.3    # Weight for length ratio similarity
    word_jaccard: 0.3    # Weight for word-level Jaccard similarity
```

## Tips for Effective Searching

1. **Start with generic terms**: Begin with general part categories like "bolt", "gear", or "connector"
2. **Adjust the threshold**: Lower the threshold (e.g., 0.5) for more results, raise it (e.g., 0.8) for stricter matching
3. **Use descriptive part names**: More specific queries like "hex bolt" or "planetary gear" can yield more targeted results
4. **Check visualization**: Always use the `--visualize` flag to see the results

## Troubleshooting

If you're not getting expected results:

- Try different variations of the part name
- Lower the match threshold
- Check if the part exists in your database
- Review the console output for details about the matching process