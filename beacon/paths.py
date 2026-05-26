"""Repo-relative paths used by experiment and plotting scripts."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
MATERIALS_DIR = DATA_DIR / "materials"
MNIST_DATA_DIR = DATA_DIR / "mnist"
RESULTS_DIR = REPO_ROOT / "results"
CONTINUOUS_SINGLE_OUTCOME_RESULTS_DIR = RESULTS_DIR / "continuous-single-outcome"
PLOT_DATA_DIR = RESULTS_DIR / "plot-data"
GENERATED_RESULTS_DIR = RESULTS_DIR / "generated"
FIGURES_DIR = REPO_ROOT / "figures"
FIGURE_OUTPUTS_DIR = FIGURES_DIR / "output"
MODELS_DIR = REPO_ROOT / "models"
MNIST_MODELS_DIR = MODELS_DIR / "mnist"
