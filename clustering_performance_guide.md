# Clustering Performance Analysis Guide

## Overview

The `analyze_clustering_performance.py` script evaluates how well your clustering algorithm performs by comparing it against ground truth data.

## Input Files

### 1. Clusters File (`all_names_clusters.csv`)
Contains the clustering results:
```csv
record_index,cluster_id,record_id,first_name,last_name
99,0,1f3b097252bc21a32c91bd5224b641b0,michael,bahmasell
35,0,428401aaafd6adc11a3ffc6dcc07a53f,michael,bahrmasel
100,0,d9320f7f63d64903559bff4ff1f60036,michael,bahmasel
```

- `cluster_id`: Groups of records the algorithm thinks are the same person
- `record_id`: Unique identifier for each record

### 2. Ground Truth File (`nexp.csv`)
Contains known matches/non-matches:
```csv
t50_id,crm_id,Is Match
ed41e65cdd8969bb8cca77e37ad72d81,a4b6eb1d8d8a9a6eb5ad9c8a13273c38,Y
36c93c7b6c34aff5d3aafb7a658115cc,f981f8cee6f29d8c50883915806e8c4f,Y
d72939e565e375e15915f99e242d794e,22ed62ff6b9e860ffc1c3b967d448c48,N
```

- `t50_id`, `crm_id`: IDs from different systems
- `Is Match`: Ground truth indicator
  - `Y` = These IDs represent the same person
  - `N` or empty = These IDs represent different people

## Usage

```bash
# Basic usage
python analyze_clustering_performance.py \
  --clusters all_names_clusters.csv \
  --ground-truth nexp.csv

# With custom output prefix
python analyze_clustering_performance.py \
  --clusters all_names_clusters.csv \
  --ground-truth nexp.csv \
  --output my_analysis
```

## Understanding the Metrics

### Confusion Matrix

The script evaluates clustering as a binary classification problem:
- **Prediction**: Same cluster (positive) or Different clusters (negative)
- **Ground Truth**: Match/Y (positive) or No Match/N (negative)

```
                      Predicted: Same Cluster    Predicted: Diff Cluster
Actual: Match (Y)            TP                         FN
Actual: No Match (N)         FP                         TN
```

### Metrics Explained

**True Positives (TP)**
- **What**: Pairs that ARE in the same cluster AND ground truth = 'Y'
- **Meaning**: Correctly identified matches
- **Example**: Michael Bahmasell and Michael Bahrmasel are clustered together, and they ARE the same person ✓

**False Positives (FP)**
- **What**: Pairs that ARE in the same cluster BUT ground truth = 'N'
- **Meaning**: Incorrectly clustered different people together
- **Example**: Two different people named "Alex" are clustered together, but they're NOT the same person ✗

**True Negatives (TN)**
- **What**: Pairs that are NOT in the same cluster AND ground truth = 'N'
- **Meaning**: Correctly kept separate
- **Example**: John Smith and Jane Doe are in different clusters, and they ARE different people ✓

**False Negatives (FN)**
- **What**: Pairs that are NOT in the same cluster BUT ground truth = 'Y'
- **Meaning**: Failed to cluster actual matches together
- **Example**: "Bill" and "William" are in different clusters, but they ARE the same person ✗

### Performance Metrics

**Precision = TP / (TP + FP)**
- Of all pairs we clustered together, what percentage were actually matches?
- High precision = Few false alarms (low FP)

**Recall = TP / (TP + FN)**
- Of all actual matches, what percentage did we successfully cluster together?
- High recall = Few missed matches (low FN)

**F1 Score = 2 × (Precision × Recall) / (Precision + Recall)**
- Harmonic mean of precision and recall
- Balances both metrics into a single score

**Accuracy = (TP + TN) / Total**
- Overall percentage of correct classifications
- Can be misleading if classes are imbalanced

## Output Files

The script generates three CSV files:

1. **`{prefix}_false_positives.csv`**
   - Details of all false positives
   - Use this to identify why different people were incorrectly clustered together

2. **`{prefix}_false_negatives.csv`**
   - Details of all false negatives
   - Use this to identify why matching people were incorrectly separated

3. **`{prefix}_summary.csv`**
   - Summary of all metrics
   - Easy to import into spreadsheets for comparison

## Interpreting Results

### Good Performance
```
Precision: 0.95+  (Few false alarms)
Recall: 0.90+     (Caught most matches)
F1 Score: 0.92+   (Good balance)
```

### What to Improve

**Low Precision (High FP)**
- Algorithm is too aggressive
- Clustering different people together
- **Solution**: Increase matching threshold, tighten blocking rules

**Low Recall (High FN)**
- Algorithm is too conservative
- Missing valid matches
- **Solution**: Decrease matching threshold, improve name normalization, use semantic matching

## Example Output

```
CONFUSION MATRIX
===================================================================
                     Predicted: Same Cluster    Predicted: Diff Cluster
Actual: Match (Y)           TP = 42                   FN = 8
Actual: No Match (N)        FP = 3                    TN = 27

PERFORMANCE METRICS
===================================================================
Total Pairs Evaluated:  80
Skipped (missing IDs):  4

True Positives (TP):    42  - Correctly clustered together
False Positives (FP):   3   - Incorrectly clustered together
True Negatives (TN):    27  - Correctly separated
False Negatives (FN):   8   - Incorrectly separated

Precision:              0.9333  (TP / (TP + FP))
Recall:                 0.8400  (TP / (TP + FN))
F1 Score:               0.8842
Accuracy:               0.8625  ((TP + TN) / Total)
```

This shows:
- **Precision 93.3%**: When we cluster pairs together, we're right 93% of the time
- **Recall 84%**: We found 84% of all actual matches
- **F1 Score 88.4%**: Good overall balance
- **8 False Negatives**: We missed 8 actual matches that should have been clustered

## Next Steps

1. **Review false positives** to understand why non-matches were clustered
2. **Review false negatives** to understand why matches were missed
3. **Tune your clustering parameters** based on findings
4. **Re-run evaluation** to measure improvement
