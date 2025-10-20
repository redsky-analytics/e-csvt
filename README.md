# ecsvt - Entity Resolution and CSV Transformation Tool

A Python package for entity resolution, duplicate detection, and CSV data transformation using hybrid matching techniques including fuzzy string matching, semantic embeddings, and blocking strategies.

## Features

### Entity Resolution
- **Hybrid Matching**: Combines fuzzy string matching, semantic embeddings, and rule-based matching
- **Blocking Strategies**: 7 types of blocks to reduce O(n²) comparisons to ~O(n)
  - Exact email, email domain, name key, Soundex, company word, name combo
  - Optional embedding-based blocking with FAISS for semantic similarity
- **Semantic Matching**: Uses SentenceTransformers (`all-MiniLM-L6-v2`) for nickname detection
- **Pre-computed Embeddings**: 10-40x performance improvement through batch encoding
- **Clustering**: Groups duplicate pairs into connected components using DFS

### CSV Transformation
- **YAML-based Configuration**: Define transformations declaratively
- **Built-in Transformations**: Name parsing, email extraction, copying with normalization
- **Post-processing**: Hash ID generation, duplicate removal
- **Flexible Column Management**: Keep original columns, transformed columns, or both

### CLI Tools
- **ecsvt-transform**: Apply transformations from YAML config
- **ecsvt-similarity**: Interactive REPL for computing semantic similarity

## Installation

```bash
# From PyPI (when published)
pip install ecsvt

# From source
git clone https://github.com/redsky-analytics/e-csvt.git
cd e-csvt
pip install -e .
```

## Quick Start

### Python API

```python
from ecsvt import HybridEntityResolver

# Initialize resolver
resolver = HybridEntityResolver(
    model_name='all-MiniLM-L6-v2',
    use_semantic=True,
    use_embedding_blocking=False  # Set True for semantic blocking
)

# Find duplicates
duplicates = resolver.find_duplicates(
    'data.csv',
    threshold=0.75
)

# Cluster duplicates
clusters = resolver.cluster_duplicates(duplicates)

# Save results
duplicates.to_csv('duplicates.csv', index=False)
clusters.to_csv('clusters.csv', index=False)
```

### CLI Tools

#### CSV Transformation

```bash
# Apply transformations from YAML config
ecsvt-transform --config config.yml --input data.csv --output transformed.csv

# With uv (for development)
uv run ecsvt-transform --config examples/configs/n1.yml --input data.csv --output output.csv
```

Example configuration (`config.yml`):

```yaml
transformations:
  - source_column: name
    function: split_lastname_firstname
    output_columns:
      - LastName
      - FirstName
    description: "Split 'LASTNAME, FIRSTNAME' format"

post_processing:
  add_hash_id:
    - column_name: "id"
      algorithm: "join"
      prefix: "user_"
      subset:
        - FirstName
        - LastName
```

#### Semantic Similarity REPL

```bash
# Interactive similarity comparison
ecsvt-similarity

# Or with uv
uv run ecsvt-similarity
```

```
Enter string 1: Bill Smith
Enter string 2: William Smith

Similarity: 0.8349
Interpretation: High similarity (>=0.75) - likely match
```

## Configuration

### Field Mapping

Map CSV columns to expected fields:

```python
resolver = HybridEntityResolver(
    field_mapping={
        'email': 'email_address',    # Map to different column name
        'first_name': 'fname',
        'last_name': 'lname',
        'company': None               # Skip this field
    }
)
```

### Field Weights

Customize scoring weights:

```python
resolver = HybridEntityResolver(
    field_weights={
        'email': 0.50,          # Increased emphasis on email
        'last_name': 0.20,
        'first_name': 0.15,
        'company': 0.10,
        'semantic_name': 0.05   # Reduced semantic weight
    }
)
```

### Embedding-Based Blocking

Enable semantic blocking with FAISS:

```python
resolver = HybridEntityResolver(
    use_embedding_blocking=True,
    embedding_block_k=50,                    # K nearest neighbors
    embedding_similarity_threshold=0.75      # Minimum similarity
)
```

## Performance

### Blocking Efficiency
- **Without blocking**: n² comparisons (5B for 100K records)
- **With blocking**: ~O(n) comparisons (~9.5M for 100K records)
- **Reduction ratio**: Typically 526x speedup

### Embedding Optimization
- **Pre-computed embeddings**: 10-40x faster than on-demand encoding
- **Batch encoding**: Processes thousands of records efficiently
- **Memory usage**: ~15 MB for 10K records (float32, 384 dimensions)

## Documentation

- [Blocking Strategy](docs/blocking.md) - Detailed explanation of all 7 blocking types
- [Clustering Algorithm](docs/clustering.md) - How connected components are formed
- [Performance Guide](docs/clustering_performance_guide.md) - Evaluation metrics and interpretation

## Project Structure

```
e-csvt/
├── src/
│   └── ecsvt/                  # Main package
│       ├── __init__.py         # Package exports
│       ├── ers.py              # Entity resolution core
│       ├── transformations.py  # Transform functions
│       ├── apply_transformations.py  # Transform application
│       └── cli/                # CLI tools
│           ├── main.py
│           ├── ers_all_names.py
│           ├── analyze_clustering.py
│           ├── similarity_repl.py
│           └── join_csv.py
├── examples/
│   ├── configs/                # Example YAML configs
│   └── experiments/            # Experimental scripts
├── docs/                       # Documentation
└── README.md
```

## Development

```bash
# Clone repository
git clone https://github.com/redsky-analytics/e-csvt.git
cd e-csvt

# Install with development dependencies
pip install -e .

# Run tests
python -m pytest

# Run examples
cd examples
uv run python experiments/basic.py
```

## Match Types

The resolver classifies matches into 5 types:

1. **exact_email**: Identical email addresses (score = 1.0)
2. **fuzzy_email**: Very similar emails (score > 0.9)
3. **strong_name**: High name similarity (last > 0.9, first > 0.8)
4. **semantic_name**: High semantic similarity (score > 0.85)
5. **weak_match**: Above threshold but doesn't meet other criteria

## License

MIT License

## Credits

Developed by [Redsky Analytics](https://github.com/redsky-analytics)

Built with:
- [sentence-transformers](https://www.sbert.net/) - Semantic embeddings
- [RapidFuzz](https://github.com/maxbachmann/RapidFuzz) - Fuzzy string matching
- [FAISS](https://github.com/facebookresearch/faiss) - Similarity search
- [pandas](https://pandas.pydata.org/) - Data manipulation
