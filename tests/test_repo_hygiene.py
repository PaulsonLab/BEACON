"""Lightweight checks for the BEACON companion repository layout."""

import ast
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def iter_python_files():
    for path in REPO_ROOT.rglob("*.py"):
        if ".git" not in path.parts:
            yield path


def test_python_files_parse():
    for path in iter_python_files():
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(path))


def test_required_artifacts_exist():
    required_paths = [
        REPO_ROOT / "data" / "materials" / "Nitrogen.csv",
        REPO_ROOT / "data" / "materials" / "PMOF20K_traindata_7000_train.csv",
        REPO_ROOT / "data" / "materials" / "hydrogen_input_output.pkl",
        REPO_ROOT / "models" / "mnist" / "CNN2.pth",
        REPO_ROOT / "models" / "mnist" / "VAE1.pth",
        REPO_ROOT / "results" / "figure_inputs" / "MOF.mat",
        REPO_ROOT / "results" / "figure_inputs" / "MNIST.mat",
        REPO_ROOT / "results" / "figure_inputs" / "OilSorbent_distribution.png",
        REPO_ROOT / "results" / "continuous_single_outcome" / "4DAckley" / "4DAckley_cost_list_BEACON.pt",
        REPO_ROOT / "results" / "continuous_single_outcome" / "Noise" / "4DAckley" / "4DNoisyAckley_cost_list_BEACON_considernoise_1.0.pt",
    ]

    missing = [path.relative_to(REPO_ROOT) for path in required_paths if not path.exists()]
    assert missing == []


def test_no_stale_absolute_paths_in_python_files():
    stale_fragments = ["/" + "fs/ess/", "/" + "home/"]
    offenders = []

    for path in iter_python_files():
        source = path.read_text(encoding="utf-8")
        for fragment in stale_fragments:
            if fragment in source:
                offenders.append(f"{path.relative_to(REPO_ROOT)} contains {fragment}")

    assert offenders == []


def test_no_tracked_python_bytecode():
    result = subprocess.run(
        ["git", "ls-files", "*.pyc"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked_pyc = [line for line in result.stdout.splitlines() if line]
    assert tracked_pyc == []
