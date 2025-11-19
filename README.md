
# ITER — Magnetic Diagnostics Analysis (Transformers & Time Series)

This repository explores the use of Transformer-based architectures (PatchTST, TimesFM), together with transfer learning approaches in SciNet architecture, for the analysis of time series from magnetic diagnostics (MAST / tokamak) or synthetic toy model data. The work was carried out during a 6-month internship at the ITER Organization.

## Overview

This project provides tools and experiments to detect faulty signals and anomalies in magnetic diagnostics using modern machine learning methods:

- Transformer-based models tailored for time series (PatchTST / TimesFM). Principally implemented for time feature extraction.
- SCINet (with Transformers as the encoder) for extracting physical features
- Classical analysis tools and metrics for anomaly evaluation

## Table of Contents

1. [Overview](#-overview)
2. [Package Structure](#-package-structure)
3. [File Tree & Description](#-file-tree--description)
4. [Getting Started](#-getting-started)
	 - [Prerequisites](#prerequisites)
	 - [Installation](#installation)
5. [Development](#-development)
	 - [Running Tests](#running-tests)
	 - [Formatting & Linting](#formatting--linting)
	 - [Type Checking](#type-checking)
6. [Data Source](#-data-source)
7. [Contributing](#-contributing)
8. [License](#-license)
9. [Author](#-author)
10. [Acknowledgments](#-acknowledgments)

## Package Structure

Main code is located under `src/transformers_for_timeseries/`. Modules are organized by responsibility:

- `config_and_scripts/`: experiment settings and utility scripts
- `data_loading/`: dataset creation and loaders (synthetic pendulum, pipelines)
- `ml_tools/`: metrics, device selection, training callbacks, seed management
- `models/`: implementations (TimesFM, SCINet-Transformer hybrids)
- `utils/`: helper functions, synthetic dataset generators, evaluation utilities

Exploration and experiment notebooks are in `notebooks/`.
Trained models and outputs are stored in `results/`.

## File Tree

```
LICENSE
pyproject.toml
README.md
notebooks/                # EDA, experiments, validation notebooks
results/                  # Figures, model parameters, checkpoints
src/transformers_for_timeseries/
	├── config_and_scripts/
	├── data_loading/
	├── ml_tools/
	├── models/
	└── utils/
```

## Getting Started

### Prerequisites

- Python 3.9 — 3.11 (3.11 recommended)
- `pip` and a virtual environment manager (`venv` or `uv` - uv recommended)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/LouisBrusset/ITER-transformers-for-tokamak-timeseries.git
cd ITER-transformers-for-tokamak-timeseries
```

2. Create and activate a virtual environment (examples):

```powershell
# Using venv
python -m venv .venv
.venv\Scripts\Activate.ps1   # PowerShell

# Or, if you prefer uv (pipx/uv):
python -m pip install --user pipx; python -m pipx ensurepath
pipx install uv
uv venv --python 3.11
```

3. Install the package in development mode:

```bash
uv pip install -e .
```

Note: the project uses `pyproject.toml` for dependency configuration. Adjust installation according to your package manager of choice.

## 🔧 Development

### Running Tests

```bash
pytest tests/
pytest --cov=src tests/
```

To run a subset of tests:

```bash
pytest tests/test_scinet/
```

### Formatting and Linting

```bash
# Formatting (if black is used)
black src/

# Lint (if flake8 is configured)
flake8 src/
```

### Type Checking

```bash
mypy src/
```

## 📊 Data Source

The datasets used come primarily from MAST (Mega Amp Spherical Tokamak). data access and extraction utilities are grouped in `src/transformers_for_timeseries/data_loading/`. Synthetic validation datasets (pendulum, etc.) are available under `notebooks/` and `src/.../synthetic`.

Regarding Transformer models, the implementations leverage the Hugging Face Transformers library for ease of use and extensibility.

Useful resources:
- MAST Data Portal: https://mastapp.site/
- Hugging Face Transformers Library: https://huggingface.co/docs/transformers/index

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add feature"`
4. Push and open a Pull Request

Please add tests for new features and follow the project's style guidelines.

## 📄 License

This project is covered by the `LICENSE` file at the repository root.

## 👨‍💻 Author

**Louis Brusset** — louis.brusset@etu.minesparis.psl.eu

Work performed in collaboration with the ITER Organization and Mines ParisTech (École des Mines de Paris).

## 🙏 Acknowledgments

- ITER Organization for supporting the internship
- The MAST team for providing data access

## 📚 References

- [PatchTST - Transformers for time series](https://arxiv.org/pdf/2211.14730)
- [TimesFM - Decoder only Transformer for time series forecasting](https://arxiv.org/pdf/2310.10688)
- [SciNet - Physical concept learning from timeseries](https://arxiv.org/pdf/1807.10300)

