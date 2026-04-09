# APLS2

APLS2 is a modernized, user-friendly, **100% AI-written** version of the original APLS repository:
https://github.com/CosmiQ/apls

It keeps the core APLS metric behavior while updating the implementation and dependencies for modern Python environments.

## What This Project Is

- A modernization of the original APLS codebase for current Python and geospatial libraries
- A drop-in package for running APLS-style graph metric evaluation workflows
- A cleaned, test-validated codebase with bundled sample data

## What Was Modernized

- Python packaging with `pyproject.toml`
- Updated dependency stack (NetworkX 3.x, pandas 2.x, Shapely 2.x, rasterio, pyproj)
- Legacy API compatibility fixes across graph parsing, scoring, and utility layers
- Modern runtime compatibility across current Python environments

## Installation

From this repository root:

```bash
pip install -e .
```

## Quick Start

Run the CLI on bundled sample pickle data:

```bash
python -m apls2.apls \
  --test_method gt_pkl_prop_pkl \
  --truth_dir apls2/data/gt_pkl_prop_pkl/ground_truth_randomized \
  --prop_dir apls2/data/gt_pkl_prop_pkl/proposal \
  --im_dir apls2/data/images \
  --output_dir /tmp/apls2_cli \
  --output_name smoke \
  --max_files 1 \
  --n_plots 0
```

Or use the installed console script:

```bash
apls2 --help
```

## Running Tests

If you add tests to this repository, run:

```bash
python -m pytest -v --tb=short
```

## Repository Layout

- `apls2/apls2/` - package source code
- `apls2/apls2/data/` - bundled sample data and sample outputs

## Attribution

- Original APLS project: https://github.com/CosmiQ/apls
- This repository is an AI-authored modernization intended to preserve behavior while improving maintainability and installability.
