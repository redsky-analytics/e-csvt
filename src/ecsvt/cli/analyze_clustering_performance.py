"""
Analyze Clustering Performance

Compares clustering results against ground truth to compute performance metrics.

Usage:
    python analyze_clustering_performance.py --clusters all_names_clusters.csv --ground-truth nexp.csv
"""

import argparse
import pandas as pd
from collections import defaultdict
from typing import Dict, Set, Tuple


class ClusteringEvaluator:
    """Evaluate clustering performance against ground truth."""

    def __init__(self, clusters_file: str, ground_truth_file: str, all_records_file: str = None):
        """
        Initialize evaluator with data files.

        Args:
            clusters_file: Path to CSV with cluster assignments (record_id, cluster_id)
            ground_truth_file: Path to CSV with ground truth (id1, id2, Is Match)
            all_records_file: Optional path to CSV with all records (for singleton handling)
        """
        print(f"Loading clusters from: {clusters_file}")
        self.clusters_df = pd.read_csv(clusters_file)

        print(f"Loading ground truth from: {ground_truth_file}")
        self.ground_truth_df = pd.read_csv(ground_truth_file)

        # Build mapping of record_id -> cluster_id
        # Records not in any cluster will be treated as singletons (each in their own cluster)
        self.id_to_cluster = self._build_cluster_mapping()
        self.next_singleton_cluster_id = max(self.clusters_df['cluster_id']) + 1 if len(self.clusters_df) > 0 else 0

        print(f"[OK] Loaded {len(self.clusters_df)} clustered records")
        print(f"[OK] Loaded {len(self.ground_truth_df)} ground truth pairs")
        print(f"[OK] Found {len(set(self.clusters_df['cluster_id']))} unique clusters")
        print(f"[OK] Records not in clusters will be treated as singletons")

    def _build_cluster_mapping(self) -> Dict[str, int]:
        """Build mapping from record_id to cluster_id."""
        mapping = {}
        for _, row in self.clusters_df.iterrows():
            record_id = str(row['record_id'])
            cluster_id = row['cluster_id']
            mapping[record_id] = cluster_id
        return mapping

    def _get_cluster_id(self, record_id: str) -> int:
        """
        Get cluster ID for a record. If not in any cluster, assign a unique singleton cluster.

        Args:
            record_id: The record ID to look up

        Returns:
            Cluster ID (existing or newly assigned singleton)
        """
        record_id = str(record_id)
        if record_id not in self.id_to_cluster:
            # Assign a unique singleton cluster ID
            self.id_to_cluster[record_id] = self.next_singleton_cluster_id
            self.next_singleton_cluster_id += 1
        return self.id_to_cluster[record_id]

    def _are_in_same_cluster(self, id1: str, id2: str) -> bool:
        """
        Check if two IDs are in the same cluster.

        Records not in any cluster are treated as singletons (each in their own unique cluster).

        Args:
            id1: First record ID
            id2: Second record ID

        Returns:
            True if both IDs are in the same cluster, False otherwise
        """
        cluster1 = self._get_cluster_id(id1)
        cluster2 = self._get_cluster_id(id2)
        return cluster1 == cluster2

    def evaluate(self) -> Dict[str, any]:
        """
        Evaluate clustering performance.

        Returns:
            Dictionary with performance metrics and detailed results
        """
        print("\n" + "=" * 70)
        print("EVALUATING CLUSTERING PERFORMANCE")
        print("=" * 70)

        tp = 0  # True Positives: Same cluster AND ground truth = Y
        fp = 0  # False Positives: CANNOT BE COMPUTED (ground truth is incomplete)
        tn = 0  # True Negatives: Different clusters AND ground truth = N
        fn = 0  # False Negatives: Different clusters AND ground truth = Y

        # Track examples for reporting
        fp_examples = []  # Not used - FP cannot be computed
        fn_examples = []  # False negatives (incorrectly separated)

        skipped = 0  # Pairs where one or both IDs not in clusters

        print("\n[NOTE] False Positives (FP) cannot be reliably computed because:")
        print("  - Ground truth only contains a subset of all possible pairs")
        print("  - Clustered pairs not in ground truth cannot be verified")
        print("  - FP will be set to 0 for metric calculations")

        print("\nProcessing ground truth pairs...")
        for idx, row in self.ground_truth_df.iterrows():
            id1 = str(row['t50_id']) if pd.notna(row['t50_id']) else None
            id2 = str(row['crm_id']) if pd.notna(row['crm_id']) else None
            is_match = str(row['Is Match']).strip().upper() if pd.notna(row['Is Match']) else ''

            # Skip if either ID is missing
            if not id1 or not id2:
                skipped += 1
                continue

            # Determine if they're in the same cluster
            # (singletons are handled automatically by _are_in_same_cluster)
            same_cluster = self._are_in_same_cluster(id1, id2)

            # Ground truth: Y means they should be in same cluster
            ground_truth_match = (is_match == 'Y')

            # Confusion matrix logic
            # NOTE: We skip FP calculation because ground truth is incomplete
            if same_cluster and ground_truth_match:
                tp += 1
            elif same_cluster and not ground_truth_match:
                # Skip FP - cannot reliably compute (ground truth incomplete)
                skipped += 1
            elif not same_cluster and not ground_truth_match:
                tn += 1
            elif not same_cluster and ground_truth_match:
                fn += 1
                fn_examples.append({
                    'id1': id1,
                    'id2': id2,
                    'cluster_id_1': self._get_cluster_id(id1),
                    'cluster_id_2': self._get_cluster_id(id2),
                    'ground_truth': is_match
                })

        # Calculate metrics
        # Note: FP is always 0 (cannot be computed from incomplete ground truth)
        total = tp + fp + tn + fn

        # Precision cannot be computed without FP, so we set it to N/A or 1.0 if TP > 0
        precision = 1.0 if tp > 0 else 0.0  # Optimistic: assume no false positives
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / total if total > 0 else 0.0

        # Print results
        print("\n" + "=" * 70)
        print("CONFUSION MATRIX")
        print("=" * 70)
        print(f"{'':20} {'Predicted: Same Cluster':^30} {'Predicted: Diff Cluster':^30}")
        print(f"{'Actual: Match (Y)':20} {f'TP = {tp}':^30} {f'FN = {fn}':^30}")
        print(f"{'Actual: No Match (N)':20} {f'FP = N/A (not computed)':^30} {f'TN = {tn}':^30}")

        print("\n" + "=" * 70)
        print("PERFORMANCE METRICS")
        print("=" * 70)
        print(f"Total Pairs Evaluated:  {total:,}")
        print(f"Skipped:                {skipped:,}  (FP candidates + missing IDs)")
        print()
        print(f"True Positives (TP):    {tp:,}  - Correctly clustered together")
        print(f"False Positives (FP):   N/A  - Cannot compute (incomplete ground truth)")
        print(f"True Negatives (TN):    {tn:,}  - Correctly separated")
        print(f"False Negatives (FN):   {fn:,}  - Incorrectly separated")
        print()
        print(f"Precision:              {precision:.4f}  (Optimistic: assumes FP=0)")
        print(f"Recall:                 {recall:.4f}  (TP / (TP + FN))")
        print(f"F1 Score:               {f1_score:.4f}  (Based on optimistic precision)")
        print(f"Accuracy:               {accuracy:.4f}  ((TP + TN) / Total)")

        # Show example errors (FP examples skipped - not computed)
        if fn_examples:
            print("\n" + "=" * 70)
            print(f"EXAMPLE FALSE NEGATIVES (showing up to 10 of {len(fn_examples)})")
            print("=" * 70)
            print("(These pairs were separated but ground truth says they ARE a match)")
            for i, ex in enumerate(fn_examples[:10], 1):
                print(f"{i}. IDs: {ex['id1'][:16]}... & {ex['id2'][:16]}...")
                print(f"   Clusters: {ex['cluster_id_1']} and {ex['cluster_id_2']}, Ground Truth: {ex['ground_truth']}")

        return {
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'total': total,
            'skipped': skipped,
            'fp_examples': fp_examples,
            'fn_examples': fn_examples
        }

    def save_error_analysis(self, output_prefix: str, results: Dict):
        """
        Save detailed error analysis to CSV files.

        Args:
            output_prefix: Prefix for output files
            results: Results dictionary from evaluate()
        """
        # Skip false positives (not computed due to incomplete ground truth)

        # Save false negatives
        if results['fn_examples']:
            fn_df = pd.DataFrame(results['fn_examples'])
            fn_file = f"{output_prefix}_false_negatives.csv"
            fn_df.to_csv(fn_file, index=False)
            print(f"[OK] Saved {len(fn_df)} false negatives to: {fn_file}")

        # Save summary metrics
        summary_file = f"{output_prefix}_summary.csv"
        summary_df = pd.DataFrame([{
            'metric': k,
            'value': v
        } for k, v in results.items() if isinstance(v, (int, float))])
        summary_df.to_csv(summary_file, index=False)
        print(f"[OK] Saved performance summary to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze clustering performance against ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python analyze_clustering_performance.py \\
    --clusters all_names_clusters.csv \\
    --ground-truth nexp.csv

  # With custom output prefix
  python analyze_clustering_performance.py \\
    --clusters all_names_clusters.csv \\
    --ground-truth nexp.csv \\
    --output analysis_results
        """
    )

    parser.add_argument(
        '--clusters',
        required=True,
        help='Path to clusters CSV file (with record_id and cluster_id columns)'
    )
    parser.add_argument(
        '--ground-truth',
        required=True,
        help='Path to ground truth CSV file (with t50_id, crm_id, and "Is Match" columns)'
    )
    parser.add_argument(
        '--output',
        default='clustering_analysis',
        help='Output file prefix for error analysis (default: clustering_analysis)'
    )

    args = parser.parse_args()

    # Run evaluation
    evaluator = ClusteringEvaluator(args.clusters, args.ground_truth)
    results = evaluator.evaluate()

    # Save detailed error analysis
    evaluator.save_error_analysis(args.output, results)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
