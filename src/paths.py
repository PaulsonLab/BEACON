"""Repo-relative paths used by experiment and plotting scripts."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
MATERIALS_DIR = DATA_DIR / "materials"
MNIST_DATA_DIR = DATA_DIR / "mnist"
RESULTS_DIR = REPO_ROOT / "results"
CONTINUOUS_SINGLE_OUTCOME_RESULTS_DIR = RESULTS_DIR / "continuous_single_outcome"
FIGURE_INPUTS_DIR = RESULTS_DIR / "figure_inputs"
MODELS_DIR = REPO_ROOT / "models"
MNIST_MODELS_DIR = MODELS_DIR / "mnist"
