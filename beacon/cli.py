"""Command-line interface: ``python -m beacon`` (or the ``beacon`` script).

Examples::

    beacon list
    beacon run --problem ackley --dim 4 --algorithm beacon --device cpu
    beacon run --problem oil-sorbent --algorithm ug-beacon --device cuda --replicates 10
    beacon run --problem logd --algorithm beacon-saas --set warmup_steps=64 num_samples=32
    beacon run --config my-experiment.yaml
    beacon reproduce synthetic --device auto
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from beacon.algorithms import ALGORITHMS, canonical_name
from beacon.device import resolve_device

RUN_KEYS = {
    "problem", "algorithm", "device", "output_dir", "problem_kwargs", "overrides", "verbose",
}


def _parse_value(text: str):
    """Parse ``KEY=VALUE`` values as Python literals when possible (``3``, ``0.5``, ``True``, ``[1,2]``)."""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def _parse_pairs(pairs: Optional[List[str]], flag: str) -> Dict:
    out = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"{flag} expects KEY=VALUE, got '{pair}'")
        key, value = pair.split("=", 1)
        out[key.strip().replace("-", "_")] = _parse_value(value.strip())
    return out


def load_config_file(path: str) -> Dict:
    """Load a JSON or YAML run configuration (see README for the schema)."""
    text = Path(path).read_text(encoding="utf-8")
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError as exc:
            raise SystemExit("YAML configs need `pip install pyyaml` (or use JSON).") from exc
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    unknown = set(data) - RUN_KEYS
    if unknown:
        raise SystemExit(f"Unknown key(s) in {path}: {sorted(unknown)}; allowed: {sorted(RUN_KEYS)}")
    return data


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", default=None, help="auto (default), cpu, cuda, or cuda:<index>")
    p.add_argument("--replicates", type=int, help="number of independent runs (seeds seed..seed+R-1)")
    p.add_argument("--iterations", type=int, help="evaluations after the initial design")
    p.add_argument("--n-init", type=int, dest="n_init", help="size of the initial design")
    p.add_argument("--seed", type=int, help="seed of the first replicate (default 0)")
    p.add_argument("--no-early-stop", action="store_true", help="keep evaluating after reachability hits 1")
    p.add_argument("--set", nargs="+", metavar="KEY=VALUE", help="override BEACON options, e.g. k=5 kernel=matern")
    p.add_argument("--output-dir", help="where to write results (default: results/generated/...)")
    p.add_argument("--no-save", action="store_true", help="do not write result files")
    p.add_argument("-v", "--verbose", type=int, default=1, help="0 silent, 1 per replicate, 2 per iteration")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="beacon", description="BEACON: Bayesian-optimization-inspired novelty search.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list problems, algorithms and paper experiment groups")

    run = sub.add_parser("run", help="run an algorithm on a problem")
    run.add_argument("--config", help="JSON/YAML file with run settings (CLI flags take precedence)")
    run.add_argument("--problem", help="problem name (see `beacon list`)")
    run.add_argument(
        "--algorithm", "-a", default=None,
        help=f"one of {sorted(ALGORITHMS)} (default: beacon)",
    )
    run.add_argument("--dim", type=int, help="input dimension (synthetic problems)")
    run.add_argument("--noise-std", type=float, help="observation noise std (noisy-ackley)")
    run.add_argument("--problem-arg", nargs="+", metavar="KEY=VALUE", help="extra problem factory arguments")
    _add_common(run)

    rep = sub.add_parser("reproduce", help="run a paper experiment group (see `beacon list`)")
    rep.add_argument("group", help="experiment group or run tag, e.g. synthetic, ackley-4d, oil-sorbent")
    _add_common(rep)
    return parser


def _overrides_from_args(args) -> Dict:
    overrides = _parse_pairs(args.set, "--set")
    for key in ("replicates", "iterations", "n_init", "seed"):
        if getattr(args, key) is not None:
            overrides[key] = getattr(args, key)
    if args.no_early_stop:
        overrides["early_stop"] = False
    return overrides


def cmd_list() -> int:
    from beacon.paper import PAPER_EXPERIMENTS, PAPER_GROUPS
    from beacon.problems import list_problems

    print("Algorithms:")
    for name, cls in ALGORITHMS.items():
        print(f"  {name:<14} {cls.__doc__.strip().splitlines()[0]}")
    print("\nProblems:")
    for name, desc in list_problems().items():
        print(f"  {name:<18} {desc}")
    print("\nPaper experiment groups (`beacon reproduce <group>`; single runs by tag also work):")
    for name in PAPER_GROUPS:
        print(f"  {name:<18} {', '.join(r.tag for r in PAPER_EXPERIMENTS[name])}")
    return 0


def cmd_run(args) -> int:
    from beacon.runner import run_experiment

    spec = load_config_file(args.config) if args.config else {}
    problem = args.problem or spec.get("problem")
    if not problem:
        raise SystemExit("Specify --problem (or `problem` in --config).")
    problem_kwargs = dict(spec.get("problem_kwargs", {}))
    if args.dim is not None:
        problem_kwargs["dim"] = args.dim
    if args.noise_std is not None:
        problem_kwargs["noise_std"] = args.noise_std
    problem_kwargs.update(_parse_pairs(args.problem_arg, "--problem-arg"))

    overrides = {**spec.get("overrides", {}), **_overrides_from_args(args)}
    output_dir = None if args.no_save else (args.output_dir or spec.get("output_dir") or "default")
    run_experiment(
        problem,
        canonical_name(args.algorithm or spec.get("algorithm", "beacon")),
        problem_kwargs=problem_kwargs,
        overrides=overrides,
        device=resolve_device(args.device or spec.get("device", "auto")),
        output_dir=output_dir,
        verbose=args.verbose if args.verbose is not None else spec.get("verbose", 1),
    )
    return 0


def cmd_reproduce(args) -> int:
    from beacon.paper import PAPER_EXPERIMENTS
    from beacon.paths import GENERATED_RESULTS_DIR
    from beacon.runner import run_experiment

    if args.group not in PAPER_EXPERIMENTS:
        raise SystemExit(f"Unknown group '{args.group}'. Run `beacon list` to see the options.")
    device = resolve_device(args.device)
    base = Path(args.output_dir) if args.output_dir else GENERATED_RESULTS_DIR / "paper"
    for run in PAPER_EXPERIMENTS[args.group]:
        overrides = {**run.overrides, **_overrides_from_args(args)}
        run_experiment(
            run.problem,
            run.algorithm,
            problem_kwargs=run.problem_kwargs,
            overrides=overrides,
            device=device,
            output_dir=None if args.no_save else base / run.tag,
            verbose=args.verbose,
        )
    return 0


def script_main(*default_args: str) -> int:
    """Entry point for the thin scripts in ``experiments/``.

    Runs ``beacon run <default_args> <command-line args>``; flags given on the
    command line override the defaults (argparse keeps the last occurrence).
    """
    return main(["run", *default_args, *sys.argv[1:]])


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "list":
            return cmd_list()
        if args.command == "run":
            return cmd_run(args)
        return cmd_reproduce(args)
    except (ValueError, FileNotFoundError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
