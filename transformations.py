"""
Transformation functions for CSV data processing.

Each function should:
- Accept a pandas Series as input
- Return either a pandas Series (single output) or DataFrame (multiple outputs)
- Handle edge cases (None, NaN, empty strings)
- All string outputs should be converted to lowercase
"""

import pandas as pd
import re


def _to_lowercase(value):
    """Helper function to convert value to lowercase if it's a string."""
    if pd.isna(value) or value is None:
        return None
    if isinstance(value, str):
        return value.lower()
    return value


def split_lastname_firstname(series: pd.Series) -> pd.DataFrame:
    """
    Split 'LASTNAME, FIRSTNAME [MIDDLE]' format into LastName and FirstName.

    Rules:
    - Drops middle initials (single letter followed by optional period)
    - Keeps full middle names as part of FirstName

    Args:
        series: Column with format "LASTNAME, FIRSTNAME [MIDDLE]"

    Returns:
        DataFrame with two columns: LastName and FirstName

    Examples:
        "SMITH, JOHN" -> LastName: "smith", FirstName: "john"
        "SMITH, JOHN A" -> LastName: "smith", FirstName: "john" (drops initial)
        "SMITH, JOHN ALLEN" -> LastName: "smith", FirstName: "john allen"
        "SMITH, JOHN A." -> LastName: "smith", FirstName: "john" (drops initial with period)
    """
    def parse_name(name_str):
        if pd.isna(name_str) or not name_str or name_str.strip() == "":
            return pd.Series({"LastName": None, "FirstName": None})

        # Split by comma
        parts = name_str.split(",", 1)
        if len(parts) != 2:
            # No comma found, return as-is (lowercase)
            return pd.Series({"LastName": _to_lowercase(name_str.strip()), "FirstName": None})

        lastname = parts[0].strip()
        firstname_part = parts[1].strip()

        # Split firstname part by spaces
        name_tokens = firstname_part.split()

        if not name_tokens:
            return pd.Series({"LastName": _to_lowercase(lastname), "FirstName": None})

        # Filter out middle initials (single letter with optional period)
        # Keep full names (2+ characters)
        filtered_tokens = []
        for token in name_tokens:
            # Remove trailing period for checking
            clean_token = token.rstrip(".")
            # Keep if it's more than 1 character (full name)
            if len(clean_token) > 1:
                filtered_tokens.append(token)
            elif len(filtered_tokens) == 0:
                # Keep the first token even if it's a single char (actual first name)
                filtered_tokens.append(token)
            # Otherwise, skip (it's a middle initial)

        firstname = " ".join(filtered_tokens) if filtered_tokens else None

        return pd.Series({
            "LastName": _to_lowercase(lastname),
            "FirstName": _to_lowercase(firstname)
        })

    return series.apply(parse_name)


def extract_email_username(series: pd.Series) -> pd.Series:
    """
    Extract username from email address (removes @domain).

    Args:
        series: Column with email addresses

    Returns:
        Series with usernames only

    Examples:
        "john.doe@example.com" -> "john.doe"
        "jane@company.co.uk" -> "jane"
        "" -> None
    """
    def extract_username(email):
        if pd.isna(email) or not email or email.strip() == "":
            return None

        # Split by @ and take the first part
        parts = email.split("@", 1)
        return _to_lowercase(parts[0].strip()) if parts else None

    return series.apply(extract_username)


def split_firstname_lastname(series: pd.Series) -> pd.DataFrame:
    """
    Split 'FirstName LastName' format into separate columns.

    Assumes the first token is FirstName and remaining tokens are LastName.

    Args:
        series: Column with format "FirstName LastName"

    Returns:
        DataFrame with two columns: FirstName and LastName

    Examples:
        "john smith" -> FirstName: "john", LastName: "smith"
        "mary jane watson" -> FirstName: "mary", LastName: "jane watson"
        "single" -> FirstName: "single", LastName: None
    """
    def parse_fullname(fullname):
        if pd.isna(fullname) or not fullname or fullname.strip() == "":
            return pd.Series({"FirstName": None, "LastName": None})

        # Split by spaces
        tokens = fullname.strip().split(maxsplit=1)

        if len(tokens) == 0:
            return pd.Series({"FirstName": None, "LastName": None})
        elif len(tokens) == 1:
            return pd.Series({"FirstName": _to_lowercase(tokens[0]), "LastName": None})
        else:
            return pd.Series({
                "FirstName": _to_lowercase(tokens[0]),
                "LastName": _to_lowercase(tokens[1])
            })

    return series.apply(parse_fullname)


def copy(series: pd.Series) -> pd.Series:
    """
    Copy values as-is and convert to lowercase.

    This is useful for including columns in the output without transformation,
    while still applying the lowercase normalization.

    Args:
        series: Column to copy

    Returns:
        Series with values converted to lowercase

    Examples:
        "Yes" -> "yes"
        "NO" -> "no"
        "Y" -> "y"
        123 -> 123 (non-strings passed through)
    """
    return series.apply(_to_lowercase)


# Template for adding more transformations:
# def your_transformation_name(series: pd.Series) -> pd.Series | pd.DataFrame:
#     """
#     Description of transformation.
#
#     Args:
#         series: Input column
#
#     Returns:
#         Transformed output
#     """
#     def transform_value(value):
#         # Your transformation logic here
#         return transformed_value
#
#     return series.apply(transform_value)
