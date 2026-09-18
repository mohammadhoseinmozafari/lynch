# lynch

> **Turn messy datasets into clear next steps.**

`lynch` is a lightweight toolkit for investigating data quality before it
becomes a modelling problem. Start with missingness patterns and density-based
coverage, inspect the result as a pandas object, and plot it when a picture
tells the story faster.

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/status-pre--release-6C5CE7)
![Focus](https://img.shields.io/badge/focus-data%20investigation-00B894)

## What it helps you see

| Lens | Questions it answers |
| --- | --- |
| **Missingness by column** | Which fields are incomplete, and by how much? |
| **Missingness by row** | Which records are partially—or entirely—missing? |
| **Missingness relationships** | Which fields go missing together, and what observed values are associated with that missingness? |
| **Data coverage** | Where is the data dense, sparse, or outside the common patterns? |

## Install

Once published to PyPI:

```bash
python -m pip install lynch
```

To work from a checkout:

```bash
git clone <your-repository-url>
cd lynch
python -m pip install -e .
```

## Quick start

```python
import pandas as pd

from lynch.missingness import ColumnsMissingness, RowsMissingness

df = pd.read_csv("customers.csv")

# A sortable table: one row per field.
column_report = ColumnsMissingness(df).summary()
print(column_report.raw)

# Count records with at least 40% of their fields missing.
row_report = RowsMissingness(df).missing_above(0.40)
print(row_report.raw)

# Render the corresponding chart in a notebook.
column_report.plot
```

Every analysis result exposes `.raw` for the underlying pandas object or
dictionary, and `.plot` when that analysis has a visualizer.

## Investigate missingness

```python
from lynch.missingness import VectorizedMissingnessCorrelation

investigation = VectorizedMissingnessCorrelation(df)

# Do two fields tend to be missing at the same time?
missingness_matrix = investigation.missing_missing_corr()
print(missingness_matrix.raw)

# Is a field's missingness related to other observed values?
drivers = investigation.missing_value_correlation()
print(drivers.raw.head())
```

`VectorizedMissingnessCorrelation` uses point-biserial correlation for numeric
comparisons and Cramér's V for categorical comparisons.

## Explore data coverage

```python
from lynch.coverage import DensityEstimate

# Numeric features only; impute or remove missing values first.
coverage = DensityEstimate(
    df.dropna(subset=["age", "income"]),
    columns=["age", "income"],
    method="kde",
)

print(coverage.density_summary())
print(coverage.low_density_regions())

# Optional visual exploration.
coverage.plot.density_2d(("age", "income"))
```

`DensityEstimate` supports kernel density estimation (`"kde"`) and Gaussian
mixture models (`"gmm"`). It scales numeric features internally so one
large-range column does not dominate the result.

## Notes

- Plotting uses matplotlib. In a notebook, use `%matplotlib inline` if your
  environment does not display figures automatically.
- Density estimation requires numeric columns with no missing values in the
  selected features.
- This project is pre-release software. Please validate findings against your
  domain knowledge before making production decisions.

## Development

Build the distributable archives and validate their metadata:

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

The distribution is named `lynch`, and the public Python imports also begin
with `lynch`.
