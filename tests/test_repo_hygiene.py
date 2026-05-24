"""Lightweight checks for the BEACON companion repository layout."""

import ast
import importlib
import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_NAME = re.compile(r"^[a-z0-9][a-z0-9-]*(\.py)?$")


def git_ls_files(*patterns):
    result = subprocess.run(
        ["git", "ls-files", *patterns],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [REPO_ROOT / line for line in result.stdout.splitlines() if line]


def tracked_python_files():
    return git_ls_files("*.py")


def test_python_files_parse():
    for path in tracked_python_files():
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(path))


def test_beacon_package_imports():
    paths = importlib.import_module("beacon.paths")
    importlib.import_module("beacon.thompson_sampling")

    assert paths.REPO_ROOT == REPO_ROOT
    assert paths.PLOT_DATA_DIR == REPO_ROOT / "results" / "plot-data"
    assert paths.FIGURE_OUTPUTS_DIR == REPO_ROOT / "figures" / "output"


def test_expected_layout_exists():
    required_dirs = [
        "beacon",
        "experiments/continuous/single-outcome",
        "experiments/continuous/multi-outcome",
        "experiments/discrete/single-outcome-material",
        "experiments/discrete/multi-outcome/mnist",
        "figures/scripts",
        "data/materials",
        "models/mnist",
        "results/continuous-single-outcome",
        "results/plot-data",
    ]
    retired_dirs = [
        "src",
        "Continuous_SingleOutcome",
        "Continuous_MultiOutcome",
        "Discrete_SingleOutcome_Material",
        "Discrete_MultiOutcome",
        "Plotting",
        "results/figure_inputs",
        "results/continuous_single_outcome",
    ]

    missing = [path for path in required_dirs if not (REPO_ROOT / path).is_dir()]
    present = [path for path in retired_dirs if (REPO_ROOT / path).exists()]
    assert missing == []
    assert present == []


def test_required_scripts_and_artifacts_exist():
    required_paths = [
        "experiments/continuous/single-outcome/ackley-beacon.py",
        "experiments/continuous/multi-outcome/synthetic-beacon.py",
        "experiments/discrete/single-outcome-material/logd-beacon.py",
        "experiments/discrete/multi-outcome/oil-beacon.py",
        "experiments/discrete/multi-outcome/mnist/mnist-beacon.py",
        "figures/scripts/plot-synthetic.py",
        "figures/scripts/plot-mof.py",
        "data/materials/Nitrogen.csv",
        "data/materials/PMOF20K_traindata_7000_train.csv",
        "data/materials/hydrogen_input_output.pkl",
        "models/mnist/CNN2.pth",
        "models/mnist/VAE1.pth",
        "results/plot-data/MOF.mat",
        "results/plot-data/MNIST.mat",
        "results/plot-data/OilSorbent_distribution.png",
        "results/continuous-single-outcome/4DAckley/4DAckley_cost_list_BEACON.pt",
        "results/continuous-single-outcome/Noise/4DAckley/4DNoisyAckley_cost_list_BEACON_considernoise_1.0.pt",
    ]

    missing = [path for path in required_paths if not (REPO_ROOT / path).exists()]
    assert missing == []


def test_no_stale_paths_or_imports_in_python_files():
    stale_fragments = [
        "/" + "fs/ess/",
        "/" + "home/",
        "src" + ".paths",
        "src" + ".ThompsonSampling",
        "figure" + "_inputs",
        "FIGURE" + "_INPUTS_DIR",
        "continuous" + "_single_outcome",
    ]
    offenders = []

    for path in tracked_python_files():
        if path.relative_to(REPO_ROOT).parts[0] == "tests":
            continue
        source = path.read_text(encoding="utf-8")
        for fragment in stale_fragments:
            if fragment in source:
                offenders.append(f"{path.relative_to(REPO_ROOT)} contains {fragment}")

    assert offenders == []


def test_public_script_names_are_lowercase_dash():
    helper_modules = {
        Path("experiments/discrete/multi-outcome/oil_sorbent.py"),
    }
    public_roots = (
        Path("experiments"),
        Path("figures") / "scripts",
    )
    offenders = []

    for path in tracked_python_files():
        rel_path = path.relative_to(REPO_ROOT)
        if rel_path in helper_modules:
            continue
        if not any(rel_path == root or root in rel_path.parents for root in public_roots):
            continue
        for part in rel_path.parts:
            if part.endswith(".py") or part not in {"experiments", "figures", "scripts"}:
                if not PUBLIC_NAME.fullmatch(part):
                    offenders.append(str(rel_path))
                    break

    assert offenders == []


def test_plotting_scripts_save_to_figure_outputs_dir():
    offenders = []

    for path in sorted((REPO_ROOT / "figures" / "scripts").glob("*.py")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if ".savefig(" in line and "FIGURE_OUTPUTS_DIR" not in line:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{line_number}")

    assert offenders == []


def test_no_tracked_python_bytecode():
    tracked_pyc = git_ls_files("*.pyc")
    assert tracked_pyc == []
