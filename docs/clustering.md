# Clustering in Entity Resolution

## Overview

Clustering is the process of grouping duplicate records together after entity matching. When the system identifies pairs of records that likely represent the same entity, clustering groups all related records into unified clusters.

## Why Clustering?

Entity matching produces **pairs** of potential duplicates:
- Record A matches Record B (score: 0.85)
- Record B matches Record C (score: 0.90)
- Record D matches Record E (score: 0.88)

But we need **groups** (clusters) to understand which records represent the same entity:
- Cluster 1: {A, B, C}
- Cluster 2: {D, E}

Clustering enables:
- **Deduplication**: Identify all variations of the same entity
- **Record linking**: Connect records across different systems
- **Data quality**: Understand the extent of duplicates in your data

## Algorithm: Connected Components

The clustering implementation uses a **graph-based connected components** algorithm:

### 1. Graph Construction

Each record is a **node**, and each match is an **edge**:

```
If matching pairs are:
  (A, B), (B, C), (D, E)

Then the graph looks like:
  A --- B --- C    D --- E
```

### 2. Finding Connected Components

The algorithm uses **Depth-First Search (DFS)** to find all connected components:

```python
def dfs(node, cluster):
    visited.add(node)
    cluster.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(neighbor, cluster)
```

**How it works:**
1. Start at an unvisited node
2. Mark it as visited and add to current cluster
3. Recursively visit all connected neighbors
4. Repeat until all nodes are visited

**Result:** Each connected component becomes a cluster

### 3. Transitive Closure

An important property: **transitivity**

If A matches B, and B matches C, then A, B, and C are all in the same cluster, even if A and C were never directly compared.

**Example:**
```
Matches found:
  - william.smith@company.com matches bill.smith@company.com (0.85)
  - bill.smith@company.com matches w.smith@company.com (0.82)

Clustering result:
  Cluster 1: {william.smith@company.com, bill.smith@company.com, w.smith@company.com}
```

Even though "william.smith" and "w.smith" were never directly compared, they're clustered together through their mutual connection to "bill.smith".

## Implementation Details

### Location: `ers.py:570-647`

The `cluster_duplicates()` method performs clustering:

```python
def cluster_duplicates(self, matches_df: pd.DataFrame, original_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Group duplicate pairs into clusters

    Args:
        matches_df: DataFrame of matches from find_duplicates()
        original_df: Original DataFrame with 'id' column (optional)

    Returns:
        DataFrame with cluster_id for each record
    """
```

### Step-by-Step Process

#### Step 1: Build the Graph

```python
# Build graph of connections
graph = defaultdict(set)
for _, row in matches_df.iterrows():
    graph[row[id_col1]].add(row[id_col2])
    graph[row[id_col2]].add(row[id_col1])
```

Creates an adjacency list representation where:
- Keys = record indices
- Values = sets of connected record indices

#### Step 2: Find Connected Components

```python
# Find connected components (clusters)
visited = set()
clusters = []

def dfs(node, cluster):
    visited.add(node)
    cluster.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(neighbor, cluster)

for node in list(graph.keys()):
    if node not in visited:
        cluster = set()
        dfs(node, cluster)
        clusters.append(cluster)
```

Traverses the graph to identify all connected components.

#### Step 3: Assign Cluster IDs

```python
# Create cluster mapping
cluster_map = {}
for cluster_id, cluster in enumerate(clusters):
    for record_id in cluster:
        cluster_map[record_id] = cluster_id
```

Assigns a unique `cluster_id` to each group of connected records.

#### Step 4: Build Result DataFrame

```python
result_data = []
for record_index, cluster_id in cluster_map.items():
    row_data = {
        'record_index': record_index,
        'cluster_id': cluster_id
    }
    # Add original ID and fields if available
    if original_df is not None and 'id' in original_df.columns:
        row_data['record_id'] = original_df.loc[record_index, 'id']
        # Include active fields for reference
    result_data.append(row_data)
```

Creates a DataFrame mapping each record to its cluster.

## Usage Example

```python
# Step 1: Find duplicate pairs
resolver = HybridEntityResolver()
matches = resolver.find_duplicates('data.csv', threshold=0.75)

# Matches output:
#   id1    id2    composite_score
#   100    200    0.85
#   200    300    0.82
#   400    500    0.90

# Step 2: Cluster the duplicates
clusters = resolver.cluster_duplicates(matches, original_df=df)

# Clusters output:
#   record_index  cluster_id  record_id  first_name  last_name
#   100           0           usr_abc    William     Smith
#   200           0           usr_def    Bill        Smith
#   300           0           usr_ghi    W.          Smith
#   400           1           usr_jkl    John        Doe
#   500           1           usr_mno    Jon         Doe

# Result: 2 clusters identified
```

## Key Features

### 1. Transitive Closure
Records are grouped together through shared connections, even without direct comparison.

### 2. Efficient Graph Traversal
Uses DFS for O(V + E) time complexity:
- V = number of records in matches
- E = number of matching pairs

### 3. Singleton Handling
Records that don't match anything are not included in the clusters output. They remain as standalone records.

### 4. ID Preservation
The clustering maintains both:
- `record_index`: Internal pandas DataFrame index
- `record_id`: Original ID from the 'id' column (if present)

## Output Format

The `cluster_duplicates()` method returns a DataFrame with:

| Column | Description |
|--------|-------------|
| `record_index` | Pandas DataFrame index of the record |
| `cluster_id` | Cluster identifier (0, 1, 2, ...) |
| `record_id` | Original record ID from data (if available) |
| `first_name`, `last_name`, etc. | Active fields for reference (if available) |

**Example:**
```csv
record_index,cluster_id,record_id,first_name,last_name,email
99,0,1f3b097252bc21a3,michael,bahmasell,m.bahmasell@co.com
35,0,428401aaafd6adc1,michael,bahrmasel,m.bahrmasel@co.com
100,0,d9320f7f63d64903,michael,bahmasel,michael.b@co.com
```

This shows three records (indices 99, 35, 100) clustered together as cluster 0, representing the same person "Michael Bahmasell" with name variations.

## Performance Evaluation

See `analyze_clustering_performance.py` for evaluating clustering quality.

### Metrics

The clustering performance is evaluated using:

- **True Positives (TP)**: Correct clusters (same cluster, ground truth = match)
- **False Negatives (FN)**: Missed matches (different clusters, ground truth = match)
- **True Negatives (TN)**: Correctly separated (different clusters, ground truth = no match)
- **False Positives (FP)**: Cannot be computed (incomplete ground truth)

**Important Note:** FP cannot be reliably computed because ground truth only contains a subset of all possible pairs. Pairs created by clustering that aren't in ground truth cannot be verified.

### Key Metric: Recall

**Recall** is the most reliable metric:

```
Recall = TP / (TP + FN)
```

This measures: *"Of all actual matches, what percentage did we successfully cluster together?"*

- **High recall (>0.7)**: Clustering finds most true matches
- **Low recall (<0.5)**: Clustering misses many true matches

### Evaluation Example

```bash
uv run analyze_clustering_performance.py \
  --clusters all_names_clusters.csv \
  --ground-truth nexp.csv
```

**Sample Output:**
```
CONFUSION MATRIX
===================================================================
                     Predicted: Same Cluster    Predicted: Diff Cluster
Actual: Match (Y)           TP = 42                   FN = 8
Actual: No Match (N)        FP = N/A                  TN = 27

PERFORMANCE METRICS
===================================================================
True Positives (TP):    42  - Correctly clustered together
False Negatives (FN):   8   - Incorrectly separated
True Negatives (TN):    27  - Correctly separated

Recall:                 0.8400  (TP / (TP + FN))
```

This means the clustering found 84% of actual matches.

## Tuning Clustering Performance

Clustering quality depends on the matching threshold used in `find_duplicates()`:

### Low Threshold (e.g., 0.70)
- **More matches** → Larger clusters
- **Higher recall** (fewer false negatives)
- **More false positives** (risk of merging different entities)

### High Threshold (e.g., 0.90)
- **Fewer matches** → Smaller clusters
- **Lower recall** (more false negatives)
- **Fewer false positives** (more conservative)

### Optimal Threshold

Find the threshold that balances recall and precision for your use case:

1. Run clustering with different thresholds (0.70, 0.75, 0.80, 0.85, 0.90)
2. Evaluate each with `analyze_clustering_performance.py`
3. Review false negatives to understand missed matches
4. Choose the threshold that achieves your target recall

**Recommendation:** Start with 0.75-0.80 for most use cases.

## Common Clustering Issues

### Issue 1: Large "Megaclusters"

**Symptom:** One cluster contains hundreds/thousands of records

**Cause:**
- Threshold too low
- Common values (e.g., "John Smith", "info@company.com")
- Weak blocking strategy

**Solution:**
- Increase matching threshold
- Improve blocking to reduce false matches
- Add additional matching criteria

### Issue 2: Over-fragmentation

**Symptom:** Many small clusters that should be merged

**Cause:**
- Threshold too high
- Missing name variations
- Poor name normalization

**Solution:**
- Decrease matching threshold
- Improve transformation rules (see `transformations.py`)
- Enable semantic matching for nickname handling

### Issue 3: Missed Connections

**Symptom:** High false negatives

**Cause:**
- Records not in same block (never compared)
- Threshold too high
- Missing fields prevent matching

**Solution:**
- Review blocking strategy (see `ers.py:192-243`)
- Add more blocking keys
- Adjust field weights to emphasize available fields

## Best Practices

1. **Always evaluate clustering quality** using ground truth data
2. **Review false negatives** to understand what matches are being missed
3. **Tune matching threshold** based on your precision/recall requirements
4. **Use semantic matching** when dealing with nicknames and variations
5. **Monitor cluster sizes** to detect megaclusters early
6. **Save clustering results** for reproducibility and auditing

## Related Files

- `ers.py`: Main entity resolution and clustering implementation
- `ers_all_names.py`: Specialized clustering for name datasets
- `analyze_clustering_performance.py`: Clustering evaluation tool
- `clustering_performance_guide.md`: Performance evaluation guide
- `README.md`: Workflow and usage examples
