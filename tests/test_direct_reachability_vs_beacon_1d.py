"""Checks for the illustrative direct-reachability vs BEACON SI figure."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "figures" / "scripts" / "plot-direct-reachability-vs-beacon-1d.py"


def load_si_figure_module():
    spec = importlib.util.spec_from_file_location("direct_reachability_vs_beacon_1d", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_direct_reachability_probabilities_are_valid():
    module = load_si_figure_module()
    data = module.build_illustration()

    assert module.N_BEHAVIOR_BINS == 13
    assert torch.isfinite(data.direct_reachability).all()
    assert (data.direct_reachability >= 0).all()
    assert (data.direct_reachability <= 1).all()
    assert torch.allclose(
        data.bin_probabilities.sum(dim=1),
        torch.ones(data.bin_probabilities.shape[0], dtype=data.bin_probabilities.dtype),
        atol=1e-6,
    )


def test_beacon_novelty_is_valid_and_reasonably_aligned():
    module = load_si_figure_module()
    data = module.build_illustration()

    assert torch.isfinite(data.beacon_average).all()
    assert torch.isfinite(data.beacon_single).all()
    assert (data.beacon_average >= 0).all()
    assert (data.beacon_single >= 0).all()
    assert 0.7 <= data.pearson_correlation <= 0.9
    assert 0.7 <= data.spearman_correlation <= 0.85
    assert data.two_sigma_coverage >= 0.9


def test_si_figure_script_runs_headless_and_writes_outputs(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(SCRIPT_PATH),
            "--check",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "pearson_correlation=" in result.stdout
    assert "spearman_correlation=" in result.stdout
    assert "two_sigma_coverage=" in result.stdout
    assert (tmp_path / "direct_reachability_vs_beacon_1d.pdf").stat().st_size > 0
    assert (tmp_path / "direct_reachability_vs_beacon_1d.png").stat().st_size > 0
    caption = tmp_path / "direct_reachability_vs_beacon_1d_caption.tex"
    assert caption.stat().st_size > 0
    caption_text = caption.read_text(encoding="utf-8")
    assert "not a benchmark" in caption_text
    assert "Spearman rank correlation" in caption_text
    assert "finite behavior-bin occupancy" in caption_text
    assert "continuous archive-distance novelty" in caption_text
    assert "Pearson" not in caption_text
