"""The custom-problem examples in examples/ (shown in the README) must keep working."""

import subprocess
import sys

import pytest

from conftest import REPO_ROOT


@pytest.mark.parametrize("script", ["continuous_oil_sorbent.py", "discrete_nitrogen_mofs.py"])
def test_example_runs(script, tmp_path):
    result = subprocess.run(
        [sys.executable, "-W", "ignore", str(REPO_ROOT / "examples" / script), "--quick", "--device", "cpu"],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    assert "Part A" in result.stdout
    assert "beacon on" in result.stdout and "ug-beacon on" in result.stdout


def test_discrete_example_matches_builtin_nitrogen_benchmark():
    sys.path.insert(0, str(REPO_ROOT / "examples"))
    try:
        from discrete_nitrogen_mofs import make_problem
    finally:
        sys.path.pop(0)
    from beacon.problems import get_problem

    example, builtin = make_problem(), get_problem("nitrogen")
    assert (example.X == builtin.X).all() and (example.Y == builtin.Y).all()
    assert example.metric.normalizer == builtin.metric.normalizer
