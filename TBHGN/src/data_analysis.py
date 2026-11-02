"""Basic dataset analysis and plotting utilities."""

from pathlib import Path
from typing import Any, Dict

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def analyze_dataset(
    dataframe: pd.DataFrame, target_column: str = "isFraud"
) -> Dict[str, Any]:
    """Return descriptive statistics and optional fraud-class information."""
    missing_counts = dataframe.isna().sum()
    missing_percentages = (missing_counts / len(dataframe) * 100).round(2)
    analysis: Dict[str, Any] = {
        "row_count": len(dataframe),
        "column_count": len(dataframe.columns),
        "columns": list(dataframe.columns),
        "dtypes": dataframe.dtypes.astype(str).to_dict(),
        "missing_counts": missing_counts.to_dict(),
        "missing_percentages": missing_percentages.to_dict(),
        "duplicate_count": int(dataframe.duplicated().sum()),
        "numeric_summary": dataframe.describe(
            include="number"
        ).transpose(),
        "target_column": target_column,
        "class_counts": None,
        "class_percentages": None,
    }

    if target_column in dataframe.columns:
        class_counts = dataframe[target_column].value_counts(dropna=False)
        analysis["class_counts"] = class_counts
        analysis["class_percentages"] = (
            dataframe[target_column].value_counts(
                normalize=True, dropna=False
            ).mul(100).round(2)
        )

    return analysis


def save_class_distribution_plot(
    dataframe: pd.DataFrame,
    output_directory: Path,
    target_column: str = "isFraud",
) -> Path | None:
    """Save a class-distribution plot when the target column is present."""
    if target_column not in dataframe.columns:
        return None

    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / "class_distribution.png"
    counts = dataframe[target_column].value_counts(dropna=False).sort_index()
    labels = [str(label) for label in counts.index]

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.bar(labels, counts.values, color=["#4C78A8", "#F58518"][: len(labels)])
    axis.set_title(f"Distribution of {target_column}")
    axis.set_xlabel(target_column)
    axis.set_ylabel("Transaction count")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def format_analysis_summary(analysis: Dict[str, Any]) -> str:
    """Format the key analysis results for command-line output."""
    lines = [
        f"Rows: {analysis['row_count']:,}",
        f"Columns: {analysis['column_count']:,}",
        f"Duplicate rows: {analysis['duplicate_count']:,}",
        "",
        "Columns and data types:",
    ]
    lines.extend(
        f"  - {name}: {dtype}"
        for name, dtype in analysis["dtypes"].items()
    )
    lines.append("")
    lines.append("Missing values:")
    for name in analysis["columns"]:
        lines.append(
            f"  - {name}: {analysis['missing_counts'][name]:,} "
            f"({analysis['missing_percentages'][name]:.2f}%)"
        )

    if analysis["class_counts"] is None:
        lines.append(
            f"\nTarget column '{analysis['target_column']}' is absent; "
            "class-distribution analysis cannot be performed."
        )
    else:
        lines.append(f"\n{analysis['target_column']} class distribution:")
        for label, count in analysis["class_counts"].items():
            percentage = analysis["class_percentages"][label]
            lines.append(f"  - {label}: {count:,} ({percentage:.2f}%)")

    return "\n".join(lines)
