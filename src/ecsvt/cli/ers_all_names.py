"""
Entity Resolution for all_names.csv
Uses only FirstName and LastName fields with equal weights
"""

from ecsvt import HybridEntityResolver
import pandas as pd


def main():
    print("=" * 70)
    print("ENTITY RESOLUTION FOR all_names.csv")
    print("=" * 70)
    print("Using only FirstName and LastName with equal weights")
    print()

    # Initialize resolver with custom configuration
    resolver = HybridEntityResolver(
        model_name='all-MiniLM-L6-v2',
        use_semantic=True,  # Set to False for faster processing without ML
        batch_size=1000,

        # Map to actual column names in all_names.csv
        # Set email and company to None to skip them
        field_mapping={
            'email': None,        # Skip email field
            'first_name': 'FirstName',  # Map to 'FirstName' column
            'last_name': 'LastName',    # Map to 'LastName' column
            'company': None       # Skip company field
        },

        # Set equal weights for first_name and last_name
        # Since only first_name, last_name, and semantic_name are active,
        # these will be normalized to sum to 1.0
        field_weights={
            'first_name': 0.33,
            'last_name': 0.33,
            'semantic_name': 0.34  # Semantic similarity of full name
        }
    )

    print(f"Active fields: {list(resolver.active_fields.keys())}")
    print(f"Normalized weights: {resolver.weights}")
    print()

    # Find duplicates
    duplicates = resolver.find_duplicates(
        'all_names.csv',
        threshold=0.75,  # Adjust threshold as needed (0-1)
        max_comparisons=None  # Remove limit for full processing
    )

    # Save and display results
    if len(duplicates) > 0:
        print("\n" + "=" * 60)
        print("TOP MATCHES")
        print("=" * 60)

        # Display relevant columns
        display_cols = ['id1', 'id2', 'first_name_1', 'last_name_1',
                       'first_name_2', 'last_name_2', 'composite_score',
                       'first_name_score', 'last_name_score', 'semantic_score',
                       'match_type']

        print(duplicates[display_cols].head(20))

        # Save results
        duplicates.to_csv('all_names_duplicates.csv', index=False)
        print("\n✓ Full results saved to: all_names_duplicates.csv")

        # Cluster duplicates
        print("\n" + "=" * 60)
        print("CLUSTERING DUPLICATES")
        print("=" * 60)

        # Load original data for clustering
        df_original = pd.read_csv('all_names.csv')
        clusters = resolver.cluster_duplicates(duplicates, original_df=df_original)

        clusters.to_csv('all_names_clusters.csv', index=False)
        print("✓ Clusters saved to: all_names_clusters.csv")

        print("\nSample clusters:")
        print(clusters.head(20))

        # Summary statistics
        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)
        print(f"Total records processed: {len(df_original):,}")
        print(f"Duplicate pairs found: {len(duplicates):,}")
        print(f"Unique records in clusters: {len(clusters):,}")
        print(f"Number of clusters: {clusters['cluster_id'].nunique():,}")

        # Show cluster size distribution
        cluster_sizes = clusters.groupby('cluster_id').size()
        print(f"\nCluster size distribution:")
        print(f"  Average cluster size: {cluster_sizes.mean():.1f}")
        print(f"  Largest cluster: {cluster_sizes.max()} records")
        print(f"  Clusters with 2 records: {(cluster_sizes == 2).sum()}")
        print(f"  Clusters with 3+ records: {(cluster_sizes >= 3).sum()}")

    else:
        print("\nNo matches found above threshold")


if __name__ == "__main__":
    main()
