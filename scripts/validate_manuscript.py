"""Fail-closed manuscript validation for the ActiveBlockference release."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

from blockference.config import load_experiment_config

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "docs" / "manuscript"
VARIABLES_PATH = ROOT / "docs" / "_generated" / "manuscript-variables.json"


def _declared_variables() -> set[str]:
    """Return the variable keys declared by the generated variables file."""

    payload = json.loads(VARIABLES_PATH.read_text(encoding="utf-8"))
    return {
        key
        for key, value in payload.items()
        if key not in {"schema", "sources"} and not isinstance(value, dict)
    }


def _bibliography_keys() -> set[str]:
    content = (MANUSCRIPT / "references.bib").read_text(encoding="utf-8")
    return set(re.findall(r"@\w+\{([^,]+),", content))


def figure_registry_errors(
    manuscript: Path, figure_dir: Path | None = None, *, check_files: bool = True
) -> list[str]:
    """Compare the figure declaration with source references and asset names."""

    try:
        registry = json.loads((manuscript / "figure-registry.json").read_text(encoding="utf-8"))
        if not isinstance(registry, dict) or registry.get("schema") != "manuscript-figures/v1":
            raise ValueError("expected manuscript-figures/v1 schema")
        entries = registry.get("figures")
        if not isinstance(entries, list) or not entries:
            raise ValueError("figures must be a non-empty list")
        declared: dict[str, dict[str, str]] = {}
        filenames: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict) or any(
                not isinstance(entry.get(key), str) for key in ("label", "source", "svg", "png")
            ):
                raise ValueError("each figure must declare label, source, svg, and png strings")
            label = entry["label"]
            if not re.fullmatch(r"fig:[A-Za-z0-9_-]+", label) or label in declared:
                raise ValueError(f"invalid or duplicate figure label: {label}")
            if not re.fullmatch(r"[0-9][0-9]_[^/\\]+\.md", entry["source"]):
                raise ValueError(f"invalid figure source: {entry['source']}")
            for suffix in ("svg", "png"):
                name = entry[suffix]
                if Path(name).name != name or not name.endswith(f".{suffix}") or name in filenames:
                    raise ValueError(f"invalid or duplicate figure filename: {name}")
                filenames.add(name)
            if Path(entry["svg"]).stem != Path(entry["png"]).stem:
                raise ValueError(f"figure companions have different stems: {label}")
            declared[label] = entry
    except (OSError, ValueError) as exc:
        return [f"figure registry invalid: {exc}"]

    errors: list[str] = []
    referenced: set[str] = set()
    for source in sorted(manuscript.glob("[0-9][0-9]_*.md")):
        text = source.read_text(encoding="utf-8")
        for label in re.findall(r"@(fig:[A-Za-z0-9_-]+)", text):
            if label not in declared:
                errors.append(f"figure registry has no entry for reference {label} in {source.name}")
        for image in re.finditer(r"!\[[^\]]*\]\(figures/([^)]+)\)(?:\{([^}]*)\})?", text):
            name, attributes = image.groups()
            label_match = re.search(r"#(fig:[A-Za-z0-9_-]+)", attributes or "")
            label = label_match.group(1) if label_match else ""
            entry = declared.get(label)
            if entry is None:
                errors.append(f"figure registry has no entry for {name} ({label or 'unlabelled'})")
                continue
            referenced.add(label)
            if source.name != entry["source"] or name not in (entry["svg"], entry["png"]):
                errors.append(f"figure registry reference mismatch: {label} in {source.name}: {name}")
    for label in sorted(declared.keys() - referenced):
        errors.append(f"figure registry orphan declaration: {label}")
    if check_files:
        directory = figure_dir if figure_dir is not None else manuscript / "figures"
        actual = (
            {
                path.name
                for path in directory.iterdir()
                if path.is_file() and path.suffix in {".svg", ".png"}
            }
            if directory.is_dir()
            else set()
        )
        for name in sorted(filenames - actual):
            errors.append(f"figure registry missing asset: {name}")
        for name in sorted(actual - filenames):
            errors.append(f"figure registry orphan asset: {name}")
    return errors


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
    errors.extend(figure_registry_errors(MANUSCRIPT))
    tokens = set(re.findall(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}", all_text))
    try:
        undeclared = sorted(tokens - _declared_variables())
    except (OSError, ValueError) as exc:
        errors.append(f"manuscript variables unavailable: {exc}")
    else:
        if undeclared:
            errors.append(
                "undeclared manuscript variable tokens: " + ", ".join(undeclared)
            )
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
