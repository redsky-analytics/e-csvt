create all names

```
uv run apply_transformations.py --input n.csv --config n1.yml --output t50_names.csv
uv run apply_transformations.py --input n.csv --config n2.yml --output crm_names.csv
uv run join_csv_files.py t50_names.csv crm_names.csv -o all_names.csv --drop-duplicates
```

create nexp.csv for analyzing the performance

```
uv run apply_transformations.py --input n.csv --config nexp.yml --output nexp.csv
```

```
uv run apply_transformations.py --input n.csv --config nexp2.yml --output nexp2.csv --all-columns
```


process all_names and creates all_names_clusters.csv

```
uv run ers_all_names.py
````

analyze performance

```
uv run analyze_clustering_performance.py --clusters all_names_clusters.csv --ground-truth nexp.csv
```

## Confusion Matrix Rules

The clustering performance evaluation compares clustering results against ground truth pairs using the following rules:

### Definitions

For each pair (id1, id2) in the ground truth:
- **Ground Truth = Y**: The pair should be in the same cluster (they are a match)
- **Ground Truth = N**: The pair should be in different clusters (they are not a match)

### Confusion Matrix Categories

| Predicted | Ground Truth | Metric | Description |
|-----------|--------------|--------|-------------|
| Same cluster | Y (Match) | **TP** (True Positive) | Correctly clustered together |
| Different clusters | Y (Match) | **FN** (False Negative) | Incorrectly separated |
| Different clusters | N (No Match) | **TN** (True Negative) | Correctly separated |
| Same cluster | N (No Match) | **FP** (False Positive) | **NOT COMPUTED** |

### Important Limitation: False Positives Cannot Be Computed

**False Positives (FP) are NOT computed** because:

1. **Incomplete Ground Truth**: The ground truth file only contains a subset of all possible pairs
2. **Cluster Pairs**: When clustering puts multiple records together, it creates many new pairs
3. **Unverifiable Pairs**: Clustered pairs that are not in the ground truth cannot be evaluated
4. **No Negative Evidence**: We cannot determine if a clustered pair is a false positive without explicit ground truth

**Example**: If records A, B, and C are clustered together, this creates pairs (A,B), (A,C), and (B,C). If only (A,B) exists in ground truth as "Y", we cannot evaluate (A,C) and (B,C) - they might be correct matches or false positives, but we have no ground truth to verify.

### Metrics Calculation

Given this limitation:

- **Precision**: Set to 1.0 (optimistic assumption: FP = 0)
  - Formula: `TP / (TP + FP)` → `TP / TP` = 1.0 (if TP > 0)
  - Interpretation: Upper bound on precision

- **Recall**: Can be computed accurately
  - Formula: `TP / (TP + FN)`
  - Interpretation: What fraction of true matches did we find?

- **F1 Score**: Based on optimistic precision
  - Formula: `2 * (Precision * Recall) / (Precision + Recall)`
  - Interpretation: Harmonic mean with optimistic precision

- **Accuracy**: Can be computed for evaluated pairs
  - Formula: `(TP + TN) / (TP + TN + FN + FP)`
  - Since FP = 0: `(TP + TN) / (TP + TN + FN)`

### Interpreting Results

**Focus on Recall** as the most reliable metric:
- High recall (>0.7): Clustering finds most true matches
- Low recall (<0.5): Clustering misses many true matches

**Precision is optimistic** (assumes no false positives):
- Real precision is likely lower
- Use as an upper bound

**False Negatives** indicate clustering is too conservative:
- Review FN examples to identify patterns
- Consider lowering similarity thresholds