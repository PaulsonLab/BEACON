"""Lightweight checks for the BEACON companion repository layout."""

import ast
import importlib
import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_NAME = re.compile(r"^[a-z0-9][a-z0-9-]*(\.py)?$")
SI_FIGURE_SCRIPT = Path("figures/scripts/plot-direct-reachability-vs-beacon-1d.py")


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
    optimization = importlib.import_module("beacon.optimization")
    importlib.import_module("beacon.thompson_sampling")

    assert paths.REPO_ROOT == REPO_ROOT
    assert paths.PLOT_DATA_DIR == REPO_ROOT / "results" / "plot-data"
    assert paths.GENERATED_RESULTS_DIR == REPO_ROOT / "results" / "generated"
    assert paths.FIGURE_OUTPUTS_DIR == REPO_ROOT / "figures" / "output"
    assert optimization.DTYPE.__str__() == "torch.float64"


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
        rel_path = path.relative_to(REPO_ROOT)
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        tree = ast.parse(source, filename=str(path))
        savefig_calls = []
        show_calls = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr == "savefig":
                savefig_calls.append(node)
            elif node.func.attr == "show":
                show_calls.append(node)

        if not savefig_calls:
            offenders.append(f"{rel_path} has no active savefig call")

        for call in savefig_calls:
            if rel_path == SI_FIGURE_SCRIPT:
                if "FIGURES_DIR" not in source:
                    offenders.append(f"{rel_path} does not use FIGURES_DIR for default outputs")
                continue
            line = lines[call.lineno - 1]
            if "FIGURE_OUTPUTS_DIR" not in line:
                offenders.append(f"{rel_path}:{call.lineno}")

        for call in show_calls:
            offenders.append(f"{rel_path}:{call.lineno} has active show call")

    assert offenders == []


def test_experiment_scripts_save_to_generated_results_dir():
    helper_modules = {
        Path("experiments/discrete/multi-outcome/oil_sorbent.py"),
    }
    offenders = []

    for path in sorted((REPO_ROOT / "experiments").rglob("*.py")):
        rel_path = path.relative_to(REPO_ROOT)
        if rel_path in helper_modules:
            continue

        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        save_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "save"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "torch"
        ]

        if not save_calls:
            offenders.append(f"{rel_path} has no active torch.save call")
            continue
        if "GENERATED_RESULTS_DIR" not in source:
            offenders.append(f"{rel_path} does not import GENERATED_RESULTS_DIR")
        if "output_dir = GENERATED_RESULTS_DIR" not in source:
            offenders.append(f"{rel_path} does not build output_dir from GENERATED_RESULTS_DIR")

        for call in save_calls:
            if len(call.args) < 2:
                offenders.append(f"{rel_path}:{call.lineno} torch.save has no destination")
                continue
            destination = call.args[1]
            if isinstance(destination, ast.Constant) and isinstance(destination.value, str):
                offenders.append(f"{rel_path}:{call.lineno} saves to a bare string path")

    assert offenders == []


def test_continuous_scripts_use_quiet_acquisition_optimizer():
    offenders = []

    for path in sorted((REPO_ROOT / "experiments" / "continuous").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name) and func.id == "optimize_acqf":
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
            elif isinstance(func, ast.Attribute) and func.attr == "optimize_acqf":
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")

    assert offenders == []


def test_no_tracked_python_bytecode():
    tracked_pyc = git_ls_files("*.pyc")
    assert tracked_pyc == []
