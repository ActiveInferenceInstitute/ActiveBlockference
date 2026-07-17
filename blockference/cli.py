"""Typed command-line interface for ActiveBlockference."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from blockference.config import OutputConfig, load_experiment_config
from blockference.io import RunPaths, validate_run_outputs
from blockference.pipeline import run_pipeline
from blockference.simulations.grid_sim import run_experiment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="blockference")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run one configured simulation")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path)

    pipeline = commands.add_parser("pipeline", help="run simulation, artefacts, rendering, and validation")
    pipeline.add_argument("--config", type=Path, required=True)
    pipeline.add_argument("--output-root", type=Path, default=Path("output"))
    pipeline.add_argument("--run-name")
    pipeline.add_argument("--timestamped", action="store_true")

    validation = commands.add_parser("validation", help="validate a persisted run tree")
    validation.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute a typed subcommand and return a process status code."""

    args = _parser().parse_args(argv)
    if args.command == "run":
        config = load_experiment_config(args.config)
        if args.output is not None:
            config = replace(config, output=OutputConfig(path=str(args.output)))
        frame = run_experiment(config)
        print(json.dumps({"ok": True, "rows": len(frame), "output": config.output.path}))
        return 0
    if args.command == "pipeline":
        result = run_pipeline(
            args.config,
            output_root=args.output_root,
            run_name=args.run_name,
            timestamped=args.timestamped,
        )
        print(json.dumps({"ok": result.ok, "run_dir": str(result.paths.run_dir)}))
        return 0 if result.ok else 1
    run_dir = args.run_dir.resolve()
    paths = RunPaths(root=run_dir.parent, run_name=run_dir.name)
    report = validate_run_outputs(paths)
    if paths.run_dir.is_dir() and paths.data_dir.is_dir() and paths.viz_dir.is_dir() and paths.animations_dir.is_dir():
        report.write(paths)
    print(json.dumps(report.to_dict(), sort_keys=True))
    return 0 if report.ok else 1


__all__ = ["main"]
