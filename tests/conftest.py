"""Shared fixtures: device parametrization and small fitted GP models."""

import sys
import warnings
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from beacon.models import build_independent_gps  # noqa: E402

warnings.filterwarnings("ignore", message=".*torch.jit.script.*")


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def device(request):
    """Run a test on the CPU and, when available, on the first CUDA device."""
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this machine")
    return torch.device(request.param)


def toy_data(n=12, d=2, m=1, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(n, d, generator=g, dtype=torch.double)
    cols = [torch.sin(6 * X[:, :1]) + X[:, 1:2] ** 2 + 0.5 * j * X[:, :1] for j in range(m)]
    Y = 3 * torch.cat(cols, dim=-1) + 5
    return X.to(device), Y.to(device)


@pytest.fixture
def fitted_model(device):
    torch.manual_seed(0)
    X, Y = toy_data(device=device)
    return build_independent_gps(X, Y, kernel="rbf"), X, Y
