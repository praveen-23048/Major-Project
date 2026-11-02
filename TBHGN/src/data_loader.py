"""Reusable CSV loading helpers."""

from pathlib import Path
from typing import Union

import pandas as pd

PathLike = Union[str, Path]


def load_csv(file_path: PathLike) -> pd.DataFrame:
    """Load a CSV file and raise a helpful error when it cannot be read."""
    path = Path(file_path)
    if not path.exists():
        available = sorted(
            candidate.name for candidate in path.parent.glob("*.csv")
        )
        available_text = ", ".join(available) if available else "none"
        raise FileNotFoundError(
            f"CSV file not found: {path}. Available CSV files in "
            f"{path.parent}: {available_text}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"CSV path is not a file: {path}")

    try:
        return pd.read_csv(path)
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
        OSError,
    ) as error:
        raise ValueError(f"Could not parse CSV file '{path}': {error}") from error
