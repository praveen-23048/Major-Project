"""Tests for CSV loading utilities."""

from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import load_csv


def test_load_csv_success(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("id,value\n1,alpha\n2,beta\n", encoding="utf-8")

    dataframe = load_csv(csv_path)

    expected = pd.DataFrame({"id": [1, 2], "value": ["alpha", "beta"]})
    pd.testing.assert_frame_equal(dataframe, expected)


def test_load_csv_missing_file_lists_available_csvs(tmp_path: Path) -> None:
    (tmp_path / "available.csv").write_text("value\n1\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="available.csv"):
        load_csv(tmp_path / "missing.csv")


def test_load_csv_reports_parsing_error(tmp_path: Path) -> None:
    csv_path = tmp_path / "malformed.csv"
    csv_path.write_text('a,b\n"unterminated,2\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Could not parse CSV"):
        load_csv(csv_path)
