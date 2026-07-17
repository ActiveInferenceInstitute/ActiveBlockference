"""Local command-line interface for the research workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from GRTs.agents import ResearchOrchestrator, ResearchWorkspace, build_provider, process_csv


def main(argv: list[str] | None = None) -> int:
    """Run the selected offline-safe GRT command."""

    parser = argparse.ArgumentParser(prog="grts")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--workspace", type=Path, required=True)
    run.add_argument("--provider", choices=("offline", "openai"), default="offline")
    run.add_argument("--corpus", type=Path)
    csv_command = commands.add_parser("csv")
    csv_command.add_argument("path", type=Path)
    csv_command.add_argument("--instructions", default="")
    csv_command.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "csv":
        print(process_csv(args.path, args.instructions, args.output))
        return 0
    workspace = ResearchWorkspace(args.workspace)
    provider = build_provider(args.provider, corpus=args.corpus)
    print(ResearchOrchestrator(provider, workspace).run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
