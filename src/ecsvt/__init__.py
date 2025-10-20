"""
ecsvt - Entity Resolution and CSV Transformation Tool

A Python package for entity resolution, duplicate detection, and CSV data transformation
using hybrid matching techniques including fuzzy string matching, semantic embeddings,
and blocking strategies.
"""

__version__ = "0.1.0"

from ecsvt.ers import HybridEntityResolver, MatchResult, make_sample_data
from ecsvt.transformations import (
    split_lastname_firstname,
    extract_email_username,
    split_firstname_lastname,
    copy,
)

__all__ = [
    "HybridEntityResolver",
    "MatchResult",
    "make_sample_data",
    "split_lastname_firstname",
    "extract_email_username",
    "split_firstname_lastname",
    "copy",
    "__version__",
]
