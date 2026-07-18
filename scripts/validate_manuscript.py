"""Fail-closed manuscript validation for the ActiveBlockference release."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

from blockference.config import load_experiment_config

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "docs" / "manuscript"


def _bibliography_keys() -> set[str]:
    content = (MANUSCRIPT / "references.bib").read_text(encoding="utf-8")
    return set(re.findall(r"@\w+\{([^,]+),", content))


def validate() -> list[str]:
    errors: list[str] = []
    sources = sorted(MANUSCRIPT.glob("[0-9][0-9]_*.md"))
    if len(sources) < 9:
        errors.append(f"expected at least 9 ordered manuscript sections, found {len(sources)}")
    all_text = "\n".join(path.read_text(encoding="utf-8") for path in sources)
    labels = re.findall(r"#(eq|def|prop|alg|fig|tbl|sec):([A-Za-z0-9_-]+)", all_text)
    if len(labels) != len(set(labels)):
        errors.append("manuscript labels are not unique")
    citations = set(
        re.findall(
            r"\[@(?!(?:eq|def|prop|alg|fig|tbl|sec):)([A-Za-z][A-Za-z0-9_-]*)",
            all_text,
        )
    )
    missing_citations = sorted(citations - _bibliography_keys())
    if missing_citations:
        errors.append("missing bibliography keys: " + ", ".join(missing_citations))
    for figure in re.findall(r"\]\((figures/[^)]+)\)", all_text):
        if not (MANUSCRIPT / figure).is_file():
            errors.append(f"missing referenced figure: {figure}")
    if re.search(r"\{\{[A-Za-z_][A-Za-z0-9_]*\}\}", all_text):
        errors.append("unresolved manuscript variable token")
    if "\\eqref" in all_text or "\\ref{" in all_text:
        errors.append("raw LaTeX cross-reference found; use labelled Pandoc references")
    metadata_path = MANUSCRIPT / "config.yaml"
    try:
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
        experiment = metadata.get("experiment", {}) if isinstance(metadata, dict) else {}
        relative_config = experiment.get("config_path")
        if not isinstance(relative_config, str):
            errors.append("manuscript metadata must name experiment.config_path")
        else:
            config = load_experiment_config(ROOT / relative_config)
            expected = {
                "engine": config.engine,
                "dimension": config.grid.dimension,
                "planning_length": config.grid.planning_length,
                "max_policies": config.grid.max_policies,
                "timesteps": config.simulation.timesteps,
                "runs": config.simulation.runs,
                "agents": config.simulation.n_agents,
                "seed": config.seed,
                "target": list(config.simulation.target)
                if config.simulation.target != "random"
                else "random",
                "initial_states": [list(value) for value in config.simulation.resolved_initial_states],
                "affordances": list(config.grid.affordances),
            }
            mismatches = [
                f"{key} metadata={experiment.get(key)!r} config={value!r}"
                for key, value in expected.items()
                if experiment.get(key) != value
            ]
            if mismatches:
                errors.append(
                    "manuscript metadata disagrees with executable config: "
                    + "; ".join(mismatches)
                )
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        errors.append(f"manuscript executable config invalid: {exc}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    errors = validate()
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("manuscript source validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
