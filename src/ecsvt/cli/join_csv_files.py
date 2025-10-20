"""
Join/concatenate multiple CSV files into a single output CSV file.

Usage:
    # Using command line arguments:
    python join_csv_files.py file1.csv file2.csv file3.csv --output combined.csv

    # Using YAML config:
    python join_csv_files.py --config join_config.yml
"""

import argparse
import pandas as pd
import yaml
from pathlib import Path
from typing import List


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def join_csv_files(
    input_files: List[str],
    output_file: str,
    drop_duplicates: bool = False,
    drop_duplicates_subset: List[str] = None,
    drop_duplicates_keep: str = "first",
    ignore_index: bool = True,
    add_source_column: bool = False,
    source_column_name: str = "source_file",
) -> pd.DataFrame:
    """
    Join multiple CSV files into a single dataframe.

    Args:
        input_files: List of CSV file paths to join
        output_file: Path for output CSV file
        drop_duplicates: Whether to drop duplicate rows
        drop_duplicates_subset: Columns to consider for duplicate detection
        drop_duplicates_keep: Which duplicates to keep ('first', 'last', or False)
        ignore_index: Whether to reset index in combined dataframe
        add_source_column: Whether to add a column indicating source file
        source_column_name: Name of the source column if add_source_column is True

    Returns:
        Combined dataframe
    """
    print(f"Joining {len(input_files)} CSV files:")
    print("-" * 60)

    dataframes = []
    total_rows = 0

    for file_path in input_files:
        try:
            df = pd.read_csv(file_path)
            rows = len(df)
            total_rows += rows
            print(f"  {file_path}: {rows} rows, {len(df.columns)} columns")

            # Add source column if requested
            if add_source_column:
                # Use just the filename without path
                source_name = Path(file_path).name
                df[source_column_name] = source_name

            dataframes.append(df)
        except Exception as e:
            print(f"  Error reading {file_path}: {e}")
            continue

    if not dataframes:
        print("Error: No valid CSV files to join")
        return None

    print(f"\nTotal rows before join: {total_rows}")

    # Concatenate all dataframes
    print("\nConcatenating dataframes...")
    combined_df = pd.concat(dataframes, ignore_index=ignore_index)
    print(f"  Combined dataframe: {len(combined_df)} rows, {len(combined_df.columns)} columns")

    # Drop duplicates if requested
    if drop_duplicates:
        print("\nDropping duplicates:")
        initial_count = len(combined_df)

        if drop_duplicates_subset:
            # Check which columns exist
            available_subset = [col for col in drop_duplicates_subset if col in combined_df.columns]
            missing_subset = [col for col in drop_duplicates_subset if col not in combined_df.columns]

            if missing_subset:
                print(f"  Warning: Some columns not found: {missing_subset}")

            if available_subset:
                combined_df = combined_df.drop_duplicates(subset=available_subset, keep=drop_duplicates_keep)
                print(f"  Based on columns: {available_subset} (keep={drop_duplicates_keep})")
            else:
                print("  Warning: No valid columns specified, dropping based on all columns")
                combined_df = combined_df.drop_duplicates(keep=drop_duplicates_keep)
        else:
            combined_df = combined_df.drop_duplicates(keep=drop_duplicates_keep)
            print(f"  Based on all columns (keep={drop_duplicates_keep})")

        duplicates_removed = initial_count - len(combined_df)
        print(f"  Removed {duplicates_removed} duplicate rows ({len(combined_df)} rows remaining)")

    # Save output
    print("-" * 60)
    print(f"\nSaving output to: {output_file}")
    combined_df.to_csv(output_file, index=False)
    print(f"  Saved {len(combined_df)} rows and {len(combined_df.columns)} columns")
    print("\nDone!")

    return combined_df


def main():
    parser = argparse.ArgumentParser(
        description="Join multiple CSV files into a single output CSV file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Join multiple files
  python join_csv_files.py file1.csv file2.csv file3.csv -o combined.csv

  # Join with duplicate removal
  python join_csv_files.py file1.csv file2.csv -o combined.csv --drop-duplicates

  # Join with source tracking
  python join_csv_files.py file1.csv file2.csv -o combined.csv --add-source

  # Use YAML config
  python join_csv_files.py --config join_config.yml
        """,
    )

    parser.add_argument(
        "input_files",
        nargs="*",
        help="CSV files to join (not needed if using --config)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="joined_output.csv",
        help="Output CSV file (default: joined_output.csv)",
    )
    parser.add_argument(
        "--config",
        help="YAML configuration file (alternative to command-line args)",
    )
    parser.add_argument(
        "--drop-duplicates",
        action="store_true",
        help="Drop duplicate rows after joining",
    )
    parser.add_argument(
        "--drop-duplicates-subset",
        nargs="+",
        help="Columns to consider for duplicate detection (default: all columns)",
    )
    parser.add_argument(
        "--drop-duplicates-keep",
        choices=["first", "last", "false"],
        default="first",
        help="Which duplicate to keep (default: first)",
    )
    parser.add_argument(
        "--add-source",
        action="store_true",
        help="Add a column indicating the source file for each row",
    )
    parser.add_argument(
        "--source-column-name",
        default="source_file",
        help="Name of source column if --add-source is used (default: source_file)",
    )
    parser.add_argument(
        "--keep-index",
        action="store_true",
        help="Keep original indices from input files (default: reset index)",
    )

    args = parser.parse_args()

    # Load configuration from YAML if provided
    if args.config:
        print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)

        input_files = config.get("input_files", [])
        output_file = config.get("output_file", "joined_output.csv")

        # Drop duplicates configuration
        drop_dup_config = config.get("drop_duplicates", {})
        if isinstance(drop_dup_config, bool):
            drop_duplicates = drop_dup_config
            drop_duplicates_subset = None
            drop_duplicates_keep = "first"
        elif isinstance(drop_dup_config, dict):
            drop_duplicates = True
            drop_duplicates_subset = drop_dup_config.get("subset", None)
            drop_duplicates_keep = drop_dup_config.get("keep", "first")
        else:
            drop_duplicates = False
            drop_duplicates_subset = None
            drop_duplicates_keep = "first"

        # Source column configuration
        source_config = config.get("add_source_column", {})
        if isinstance(source_config, bool):
            add_source_column = source_config
            source_column_name = "source_file"
        elif isinstance(source_config, dict):
            add_source_column = True
            source_column_name = source_config.get("column_name", "source_file")
        else:
            add_source_column = False
            source_column_name = "source_file"

        ignore_index = not config.get("keep_index", False)

    else:
        # Use command-line arguments
        if not args.input_files:
            parser.error("input_files are required when not using --config")

        input_files = args.input_files
        output_file = args.output
        drop_duplicates = args.drop_duplicates
        drop_duplicates_subset = args.drop_duplicates_subset
        drop_duplicates_keep = args.drop_duplicates_keep if args.drop_duplicates_keep != "false" else False
        add_source_column = args.add_source
        source_column_name = args.source_column_name
        ignore_index = not args.keep_index

    # Validate input files
    if not input_files:
        print("Error: No input files specified")
        return

    # Join the files
    join_csv_files(
        input_files=input_files,
        output_file=output_file,
        drop_duplicates=drop_duplicates,
        drop_duplicates_subset=drop_duplicates_subset,
        drop_duplicates_keep=drop_duplicates_keep,
        ignore_index=ignore_index,
        add_source_column=add_source_column,
        source_column_name=source_column_name,
    )


if __name__ == "__main__":
    main()
