#!/usr/bin/env python3
"""Import the simulated epilepsy data and make a descriptive-statistics table."""

import os
import tempfile
from pathlib import Path

# Use a writable temporary folder for Matplotlib's font cache.
cache_dir = Path(tempfile.gettempdir()) / "epilepsy-descriptive-matplotlib"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

import matplotlib.pyplot as plt
import pandas as pd


# File locations are relative to this script, so the project folder can be moved.
project_dir = Path(__file__).resolve().parent.parent
data_file = project_dir / "data" / "simulated_epilepsy_surgery_raw.csv"
output_dir = project_dir / "outputs"
output_dir.mkdir(exist_ok=True)


# 1. Import the raw data.
data = pd.read_csv(data_file)

# 2. Exclude the patient identifier and calculate descriptive statistics.
# Numeric variables receive means, quartiles, and ranges. Categorical variables
# receive the number of levels and their most common value.
analysis_data = data.drop(columns="patient_id")
summary = analysis_data.describe(include="all").transpose()

# 3. Add missing-data information and a simple variable-type label.
summary.insert(
    0,
    "type",
    [
        "numeric"
        if pd.api.types.is_numeric_dtype(analysis_data[column])
        else "categorical"
        for column in summary.index
    ],
)
summary.insert(2, "missing", analysis_data.isna().sum())
summary.insert(3, "missing_percent", 100 * analysis_data.isna().mean())
summary["most_common_percent"] = 100 * summary["freq"] / summary["count"]

summary = summary.rename(
    columns={
        "count": "n",
        "unique": "levels",
        "top": "most_common",
        "freq": "most_common_count",
        "std": "standard_deviation",
        "25%": "first_quartile",
        "50%": "median",
        "75%": "third_quartile",
        "min": "minimum",
        "max": "maximum",
    }
)
summary.index.name = "variable"

# 4. Save the statistics as a CSV file.
summary_file = output_dir / "descriptive_statistics.csv"
summary.to_csv(summary_file)

# 5. Make a table-style chart with variables as rows and statistics as columns.
plot_table = summary.reset_index().copy()
plot_table["variable"] = plot_table["variable"].str.replace("_", " ")

for column in plot_table.columns:
    if column not in ["variable", "type", "most_common"]:
        plot_table[column] = plot_table[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.2f}"
        )
plot_table = plot_table.fillna("")

plot_table = plot_table.rename(
    columns={
        "variable": "Variable",
        "type": "Type",
        "n": "N",
        "missing": "Missing",
        "missing_percent": "Missing %",
        "mean": "Mean",
        "standard_deviation": "SD",
        "minimum": "Min",
        "first_quartile": "Q1",
        "median": "Median",
        "third_quartile": "Q3",
        "maximum": "Max",
        "levels": "Levels",
        "most_common": "Most common",
        "most_common_count": "Count",
        "most_common_percent": "%",
    }
)

figure, axis = plt.subplots(figsize=(24, 14))
axis.axis("off")
axis.set_title(
    "Descriptive statistics for the simulated epilepsy-surgery cohort",
    fontsize=16,
    fontweight="bold",
    loc="left",
    pad=18,
)

table = axis.table(
    cellText=plot_table.values,
    colLabels=plot_table.columns,
    cellLoc="center",
    colLoc="center",
    loc="upper left",
    bbox=[0, 0, 1, 0.96],
)
table.auto_set_font_size(False)
table.set_fontsize(7.5)

column_widths = {column: 0.05 for column in range(len(plot_table.columns))}
column_widths[0] = 0.18  # Variable names
column_widths[1] = 0.08  # Variable type
column_widths[6] = 0.14  # Most common category

for (row, column), cell in table.get_celld().items():
    cell.set_edgecolor("#D1D5DB")
    cell.set_linewidth(0.5)
    cell.set_width(column_widths[column])
    if row == 0:
        cell.set_facecolor("#274C77")
        cell.set_text_props(color="white", weight="bold")
    else:
        cell.set_facecolor("#F7FAFC" if row % 2 == 0 else "white")
        if column in [0, 1, 6]:
            cell.get_text().set_ha("left")

figure_file = output_dir / "descriptive_statistics_table.png"
figure.savefig(figure_file, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(figure)

print(f"Imported {len(data)} rows and {len(data.columns)} columns.")
print(f"Saved descriptive statistics to: {summary_file}")
print(f"Saved summary chart to: {figure_file}")