"""
Apply transformations defined in YAML to CSV data.

Usage:
    # Basic usage
    python apply_transformations.py --config n.yml --input n.csv --output output.csv

    # Keep all columns (original + transformed)
    python apply_transformations.py --input n.csv --include-original-columns

    # Keep only transformed columns (override config)
    python apply_transformations.py --input n.csv --exclude-original-columns

    # Short aliases
    python apply_transformations.py --input n.csv --all-columns
    python apply_transformations.py --input n.csv --transformed-only
"""

import argparse
import pandas as pd
import yaml
from pathlib import Path
import hashlib
import transformations


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_transformations(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, list[str]]:
    """
    Apply all transformations defined in the config to the dataframe.

    Args:
        df: Input dataframe
        config: Configuration dictionary from YAML

    Returns:
        Tuple of (transformed dataframe, list of output column names)
    """
    result_df = df.copy()
    all_output_columns = []

    for transform_config in config.get("transformations", []):
        source_col = transform_config["source_column"]
        func_name = transform_config["function"]
        output_cols = transform_config["output_columns"]
        description = transform_config.get("description", "")

        # Check if source column exists
        if source_col not in result_df.columns:
            print(f"Warning: Source column '{source_col}' not found in dataframe. Skipping.")
            continue

        # Get the transformation function
        if not hasattr(transformations, func_name):
            print(f"Warning: Transformation function '{func_name}' not found. Skipping.")
            continue

        transform_func = getattr(transformations, func_name)

        print(f"Applying: {func_name} on '{source_col}' -> {output_cols}")
        if description:
            print(f"  Description: {description}")

        # Apply transformation
        transformed = transform_func(result_df[source_col])

        # Handle single vs multiple output columns
        if isinstance(transformed, pd.DataFrame):
            # Multiple outputs - must match output_columns list
            if len(transformed.columns) != len(output_cols):
                print(
                    f"Warning: Function returned {len(transformed.columns)} columns "
                    f"but config specifies {len(output_cols)} output columns. Using defaults."
                )
                for col in transformed.columns:
                    result_df[col] = transformed[col]
                    all_output_columns.append(col)
            else:
                # Rename to match config
                for old_col, new_col in zip(transformed.columns, output_cols):
                    result_df[new_col] = transformed[old_col]
                    all_output_columns.append(new_col)
        else:
            # Single output
            if len(output_cols) != 1:
                print(
                    f"Warning: Function returned single column but config specifies "
                    f"{len(output_cols)} output columns. Using first column name."
                )
            result_df[output_cols[0]] = transformed
            all_output_columns.append(output_cols[0])

    return result_df, all_output_columns


def apply_post_processing(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Apply post-processing steps defined in the config.

    Post-processing happens BEFORE column selection, allowing columns added here
    (like hash IDs) to be included in output_columns_to_keep.

    Args:
        df: Dataframe after transformations
        config: Configuration dictionary from YAML

    Returns:
        Post-processed dataframe
    """
    result_df = df.copy()
    post_processing = config.get("post_processing", {})

    if not post_processing:
        return result_df

    print("\nApplying post-processing:")
    print("-" * 60)

    # Add hash ID(s)
    if "add_hash_id" in post_processing:
        hash_config = post_processing["add_hash_id"]

        # Normalize to list format (supports string, dict, or list of dicts)
        hash_configs = []
        if isinstance(hash_config, str):
            # Simple format: just column name
            hash_configs.append({
                "column_name": hash_config,
                "algorithm": "md5",
                "subset": None
            })
        elif isinstance(hash_config, dict):
            # Single hash configuration
            hash_configs.append({
                "column_name": hash_config.get("column_name", "row_hash_id"),
                "algorithm": hash_config.get("algorithm", "md5").lower(),
                "subset": hash_config.get("subset", None),
                "prefix": hash_config.get("prefix", "")
            })
        elif isinstance(hash_config, list):
            # Multiple hash configurations
            for cfg in hash_config:
                if isinstance(cfg, dict):
                    hash_configs.append({
                        "column_name": cfg.get("column_name", f"hash_id_{len(hash_configs)}"),
                        "algorithm": cfg.get("algorithm", "md5").lower(),
                        "subset": cfg.get("subset", None),
                        "prefix": cfg.get("prefix", "")
                    })
        else:
            # Fallback
            hash_configs.append({
                "column_name": "row_hash_id",
                "algorithm": "md5",
                "subset": None
            })

        # Process each hash configuration
        for hash_cfg in hash_configs:
            column_name = hash_cfg["column_name"]
            algorithm = hash_cfg["algorithm"]
            subset = hash_cfg["subset"]
            prefix = hash_cfg.get("prefix", "")

            prefix_msg = f" with prefix '{prefix}'" if prefix else ""
            print(f"Adding hash ID column '{column_name}' using {algorithm.upper()}{prefix_msg}")

            # Determine which columns to hash
            columns_to_hash = subset if subset else result_df.columns.tolist()
            available_hash_cols = [col for col in columns_to_hash if col in result_df.columns]

            if subset and len(available_hash_cols) < len(columns_to_hash):
                missing = [col for col in columns_to_hash if col not in result_df.columns]
                print(f"  Warning: Some columns for hashing not found: {missing}")

            if not available_hash_cols:
                print(f"  Warning: No valid columns to hash for '{column_name}', skipping")
                continue

            def compute_hash(row, cols, algo, pfx):
                # Concatenate all values in the row (for specified columns)
                row_string = "|".join(str(row[col]) if pd.notna(row[col]) else "" for col in cols)

                # Compute hash or join
                if algo == "join":
                    # Join values with "-" separator (no cryptographic hash)
                    hash_value = "-".join(str(row[col]) if pd.notna(row[col]) else "" for col in cols)
                elif algo == "md5":
                    hash_value = hashlib.md5(row_string.encode("utf-8")).hexdigest()
                elif algo == "sha256":
                    hash_value = hashlib.sha256(row_string.encode("utf-8")).hexdigest()
                elif algo == "sha1":
                    hash_value = hashlib.sha1(row_string.encode("utf-8")).hexdigest()
                else:
                    print(f"  Warning: Unknown hash algorithm '{algo}', using MD5")
                    hash_value = hashlib.md5(row_string.encode("utf-8")).hexdigest()

                # Prepend prefix if provided (lowercase per CLAUDE.md instructions)
                if pfx:
                    return f"{pfx.lower()}{hash_value.lower()}"
                return hash_value.lower()

            result_df[column_name] = result_df.apply(lambda row: compute_hash(row, available_hash_cols, algorithm, prefix), axis=1)
            print(f"  Created {len(result_df)} hash IDs based on {len(available_hash_cols)} columns")

    # Drop duplicates
    if "drop_duplicates" in post_processing:
        dup_config = post_processing["drop_duplicates"]
        initial_count = len(result_df)

        # Support both boolean and dict configuration
        if isinstance(dup_config, bool) and dup_config:
            # Drop duplicates based on all columns
            result_df = result_df.drop_duplicates()
            print(f"Dropping duplicates based on all columns")
        elif isinstance(dup_config, dict):
            subset = dup_config.get("subset", None)  # Columns to consider
            keep = dup_config.get("keep", "first")  # 'first', 'last', or False

            if subset:
                available_subset = [col for col in subset if col in result_df.columns]
                missing_subset = [col for col in subset if col not in result_df.columns]

                if missing_subset:
                    print(f"  Warning: Some columns for duplicate detection not found: {missing_subset}")

                result_df = result_df.drop_duplicates(subset=available_subset, keep=keep)
                print(f"Dropping duplicates based on columns: {available_subset} (keep={keep})")
            else:
                result_df = result_df.drop_duplicates(keep=keep)
                print(f"Dropping duplicates based on all columns (keep={keep})")

        duplicates_removed = initial_count - len(result_df)
        print(f"  Removed {duplicates_removed} duplicate rows ({len(result_df)} rows remaining)")

    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Apply transformations defined in YAML to CSV data"
    )
    parser.add_argument(
        "--config", default="n.yml", help="YAML configuration file (default: n.yml)"
    )
    parser.add_argument(
        "--input", default="n.csv", help="Input CSV file (default: n.csv)"
    )
    parser.add_argument(
        "--output",
        default="n_transformed.csv",
        help="Output CSV file (default: n_transformed.csv)",
    )

    # Column control switches (override config file settings)
    column_group = parser.add_mutually_exclusive_group()
    column_group.add_argument(
        "--include-original-columns",
        "--all-columns",
        action="store_true",
        help="Keep all original + transformed columns (overrides config)",
    )
    column_group.add_argument(
        "--exclude-original-columns",
        "--transformed-only",
        action="store_true",
        help="Keep only transformed columns (overrides config)",
    )

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Override config with command-line flags
    if args.include_original_columns:
        config["include_original_columns"] = True
        print("Command-line override: include_original_columns = True")
    elif args.exclude_original_columns:
        config["include_original_columns"] = False
        # Also clear output_columns_to_keep to force default behavior
        if "output_columns_to_keep" in config:
            config.pop("output_columns_to_keep")
        print("Command-line override: include_original_columns = False")

    # Load input CSV
    print(f"Loading input CSV from: {args.input}")
    df = pd.read_csv(args.input)
    print(f"  Loaded {len(df)} rows and {len(df.columns)} columns")

    # Apply transformations
    print("\nApplying transformations:")
    print("-" * 60)
    transformed_df, output_columns = apply_transformations(df, config)

    # Apply post-processing FIRST (adds hash columns, drops duplicates)
    # This happens before column selection so that hash columns can be included in output_columns_to_keep
    transformed_df = apply_post_processing(transformed_df, config)

    # Handle output column selection AFTER post-processing
    # Priority order:
    # 1. If output_columns_to_keep is specified, use those (explicit control)
    # 2. If include_original_columns is true, keep all columns
    # 3. If column_order is specified, use those columns as the output selection
    # 4. Otherwise, only keep transformed output columns (default)
    if "output_columns_to_keep" in config:
        cols_to_keep = config["output_columns_to_keep"]
        available_cols = [col for col in cols_to_keep if col in transformed_df.columns]
        missing_cols = [col for col in cols_to_keep if col not in transformed_df.columns]

        if missing_cols:
            print(f"\nWarning: Some columns specified in output_columns_to_keep not found: {missing_cols}")

        transformed_df = transformed_df[available_cols]
        print(f"\nKept {len(available_cols)} specified columns from config")
    elif config.get("include_original_columns", False):
        # Keep all columns (original + transformed + post-processed)
        print(f"\nKept all {len(transformed_df.columns)} columns (original + transformed + post-processed)")
    elif "column_order" in config:
        # Use column_order to determine which columns to keep
        cols_to_keep = config["column_order"]
        available_cols = [col for col in cols_to_keep if col in transformed_df.columns]
        missing_cols = [col for col in cols_to_keep if col not in transformed_df.columns]

        if missing_cols:
            print(f"\nWarning: Some columns specified in column_order not found: {missing_cols}")

        transformed_df = transformed_df[available_cols]
        print(f"\nKept {len(available_cols)} columns based on column_order")
    else:
        # Default: only keep the output columns from transformations
        available_cols = [col for col in output_columns if col in transformed_df.columns]
        transformed_df = transformed_df[available_cols]
        print(f"\nKept {len(available_cols)} transformed output columns (default behavior)")

    # Handle column ordering (after post-processing and column selection)
    # Only reorder if we used output_columns_to_keep or include_original_columns for selection
    # (If we used column_order for selection, columns are already in the correct order)
    if "column_order" in config and ("output_columns_to_keep" in config or config.get("include_original_columns", False)):
        specified_order = config["column_order"]
        # Only include columns that exist in the dataframe
        ordered_cols = [col for col in specified_order if col in transformed_df.columns]
        missing_order_cols = [col for col in specified_order if col not in transformed_df.columns]

        if missing_order_cols:
            print(f"\nWarning: Some columns specified in column_order not found: {missing_order_cols}")

        # Add any remaining columns not in the order specification (at the end)
        remaining_cols = [col for col in transformed_df.columns if col not in ordered_cols]
        final_order = ordered_cols + remaining_cols

        transformed_df = transformed_df[final_order]
        print(f"\nReordered columns according to column_order (+ {len(remaining_cols)} unspecified columns at end)")

    # Save output
    print("-" * 60)
    print(f"\nSaving output to: {args.output}")
    transformed_df.to_csv(args.output, index=False)
    print(f"  Saved {len(transformed_df)} rows and {len(transformed_df.columns)} columns")
    print("\nDone!")


if __name__ == "__main__":
    main()
