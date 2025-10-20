"""
Semantic Similarity REPL
Interactive tool to compute semantic similarity between two strings
Uses the same method as the entity resolution system (ers.py)

Usage:
    python semantic_similarity_repl.py
    python semantic_similarity_repl.py --model all-MiniLM-L6-v2

    Or with uv:
    uv run python semantic_similarity_repl.py
    uv run ecsvt-similarity --model paraphrase-MiniLM-L6-v2

Example:
    Enter string 1: Bill Smith
    Enter string 2: William Smith

    Similarity: 0.8349
    Interpretation: High similarity (>0.75)
"""

import argparse
import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer

# Fix for multiprocessing issues with sentence-transformers on all platforms
# Prevents semaphore leaks and crashes on Windows/macOS
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class SemanticSimilarityCalculator:
    """
    Calculate semantic similarity between text strings using sentence transformers.
    Uses the same method as the entity resolution system.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the similarity calculator.

        Args:
            model_name: Name of the sentence transformer model to use
        """
        print(f"Loading model '{model_name}'...", end=' ', flush=True)
        self.model = SentenceTransformer(model_name)
        print("Done!")
        print()

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two text strings.

        Args:
            text1: First text string
            text2: Second text string

        Returns:
            Similarity score between 0 and 1 (higher = more similar)
        """
        # Encode both texts with normalized embeddings
        # This matches the method in ers.py:378-384
        embeddings = self.model.encode(
            [text1, text2],
            convert_to_numpy=True,
            normalize_embeddings=True  # Pre-normalize for cosine similarity
        )

        # Convert to float32 (matching ers.py:387)
        embeddings = embeddings.astype(np.float32)

        # For normalized vectors, cosine similarity = dot product
        # This matches the method in ers.py:454
        similarity = np.dot(embeddings[0], embeddings[1])

        return float(similarity)

    @staticmethod
    def interpret_score(score: float) -> str:
        """
        Provide a human-readable interpretation of the similarity score.

        Args:
            score: Similarity score (0-1)

        Returns:
            Interpretation string
        """
        if score >= 0.85:
            return "Very high similarity (>=0.85) - semantic_name match"
        elif score >= 0.75:
            return "High similarity (>=0.75) - likely match"
        elif score >= 0.60:
            return "Moderate similarity (>=0.60) - possible match"
        elif score >= 0.40:
            return "Low similarity (>=0.40) - weak match"
        else:
            return "Very low similarity (<0.40) - unlikely match"


def run_repl(model_name: str = 'all-MiniLM-L6-v2'):
    """
    Run the interactive REPL for semantic similarity comparison.

    Args:
        model_name: Name of the sentence transformer model to use
    """
    print("=" * 70)
    print("SEMANTIC SIMILARITY REPL")
    print("=" * 70)
    print("Compare two text strings using semantic embeddings")
    print(f"Model: {model_name}")
    print()
    print("Commands: 'quit', 'exit', 'q', or Ctrl+C to exit")
    print("=" * 70)
    print()

    # Initialize calculator (loads model)
    calc = SemanticSimilarityCalculator(model_name=model_name)

    try:
        while True:
            # Get first string
            print()
            text1 = input("Enter string 1 (or 'quit' to exit): ").strip()

            # Check for exit commands
            if text1.lower() in ['quit', 'exit', 'q', '']:
                print("\nGoodbye!")
                break

            # Get second string
            text2 = input("Enter string 2: ").strip()

            # Check for exit commands
            if text2.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            if not text2:
                print("Warning: Empty string entered for string 2")
                continue

            # Compute similarity
            score = calc.compute_similarity(text1, text2)
            interpretation = calc.interpret_score(score)

            # Display results
            print()
            print("-" * 70)
            print(f"String 1: {text1}")
            print(f"String 2: {text2}")
            print(f"Similarity: {score:.4f}")
            print(f"Interpretation: {interpretation}")
            print("-" * 70)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Interactive REPL for computing semantic similarity between text strings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ecsvt-similarity
  ecsvt-similarity --model paraphrase-MiniLM-L6-v2
  ecsvt-similarity --model all-mpnet-base-v2

Popular Models:
  all-MiniLM-L6-v2           Fast, lightweight (default, used by entity resolution)
  paraphrase-MiniLM-L6-v2    Good for paraphrase detection
  all-mpnet-base-v2          Higher quality, slower
  paraphrase-mpnet-base-v2   Best quality for paraphrases
        """
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model name (default: all-MiniLM-L6-v2)"
    )

    args = parser.parse_args()
    run_repl(model_name=args.model)


if __name__ == "__main__":
    main()
