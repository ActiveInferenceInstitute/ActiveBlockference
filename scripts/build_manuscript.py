"""Build the ActiveBlockference manuscript and its publication figures.

The source manuscript follows the ordered-file convention used by the
repository template.  Labels are resolved locally so the checked-in source
does not depend on a particular Pandoc filter for numbering.  The optional
PDF/HTML renderers can consume ``docs/_build/manuscript.md`` after this script
has composed the source files.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(os.getenv("TMPDIR", "/tmp")) / "activeblockference-mpl")
)

MANUSCRIPT_DIR = REPO_ROOT / "docs" / "manuscript"
BUILD_DIR = REPO_ROOT / "docs" / "_build" / "manuscript"
FIGURE_DIR = MANUSCRIPT_DIR / "figures"

LABEL_RE = re.compile(r"#(?P<kind>eq|def|prop|alg|fig|tbl|sec):(?P<label>[A-Za-z0-9_-]+)")
REF_RE = re.compile(r"\[@(?P<kind>eq|def|prop|alg|fig|tbl|sec):(?P<label>[A-Za-z0-9_-]+)\]")
NARRATIVE_REF_RE = re.compile(r"@(?P<kind>eq|def|prop|alg|fig|tbl|sec):(?P<label>[A-Za-z0-9_-]+)")
DIV_START_RE = re.compile(
    r"^::: (?P<kind>definition|proposition|algorithm|remark) \{#(?P<label>"
    r"(?:def|prop|alg|remark):[A-Za-z0-9_-]+)\}\s*$"
)
EQUATION_END_RE = re.compile(r"^\$\$ \{#(?P<label>eq:[A-Za-z0-9_-]+)\}\s*$")

KIND_NAMES = {
    "eq": "Equation",
    "def": "Definition",
    "prop": "Proposition",
    "alg": "Algorithm",
    "fig": "Figure",
    "tbl": "Table",
    "sec": "Section",
}


def _source_files() -> list[Path]:
    files = sorted(MANUSCRIPT_DIR.glob("[0-9][0-9]_*.md"))
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing manuscript source: " + ", ".join(map(str, missing)))
    return files


def _read_sources() -> list[tuple[Path, str]]:
    return [(path, path.read_text(encoding="utf-8")) for path in _source_files()]


def _collect_labels(sources: list[tuple[Path, str]]) -> dict[str, dict[str, int]]:
    counters: defaultdict[str, int] = defaultdict(int)
    labels: dict[str, dict[str, int]] = {}
    for path, text in sources:
        for match in LABEL_RE.finditer(text):
            kind = match.group("kind")
            label = f"{kind}:{match.group('label')}"
            if label in labels:
                raise ValueError(f"duplicate manuscript label {label} in {path}")
            counters[kind] += 1
            labels[label] = {"kind": kind, "number": counters[kind]}
    return labels


def _reference_text(kind: str, number: int, parenthetical: bool) -> str:
    name = KIND_NAMES[kind]
    return f"({name} {number})" if parenthetical else f"{name} {number}"


def _resolve_references(text: str, labels: dict[str, dict[str, int]]) -> str:
    def replace_parenthetical(match: re.Match[str]) -> str:
        key = f"{match.group('kind')}:{match.group('label')}"
        if key not in labels:
            raise ValueError(f"unresolved manuscript reference {key}")
        entry = labels[key]
        return _reference_text(entry["kind"], entry["number"], parenthetical=True)

    def replace_narrative(match: re.Match[str]) -> str:
        key = f"{match.group('kind')}:{match.group('label')}"
        if key not in labels:
            raise ValueError(f"unresolved manuscript reference {key}")
        entry = labels[key]
        return _reference_text(entry["kind"], entry["number"], parenthetical=False)

    text = REF_RE.sub(replace_parenthetical, text)
    return NARRATIVE_REF_RE.sub(replace_narrative, text)


def _resolve_formalism_divs(text: str, labels: dict[str, dict[str, int]]) -> str:
    lines = text.splitlines()
    output: list[str] = []
    index = 0
    while index < len(lines):
        match = DIV_START_RE.match(lines[index])
        if not match:
            output.append(lines[index])
            index += 1
            continue
        kind = {"definition": "def", "proposition": "prop", "algorithm": "alg", "remark": "remark"}[
            match.group("kind")
        ]
        label = match.group("label")
        if kind not in {"def", "prop", "alg"}:
            output.append(lines[index])
            index += 1
            continue
        if label not in labels:
            raise ValueError(f"unresolved formalism label {label}")
        body: list[str] = []
        index += 1
        while index < len(lines) and lines[index].strip() != ":::":
            body.append(lines[index])
            index += 1
        if index == len(lines):
            raise ValueError(f"unclosed formalism block {label}")
        number = labels[label]["number"]
        title = KIND_NAMES[kind]
        output.append(f"**{title} {number}.**")
        output.extend(body)
        output.append("")
        index += 1
    return "\n".join(output)


def _resolve_equations(text: str, labels: dict[str, dict[str, int]]) -> str:
    lines = text.splitlines()
    output: list[str] = []
    index = 0
    while index < len(lines):
        match = EQUATION_END_RE.match(lines[index])
        if not match:
            output.append(lines[index])
            index += 1
            continue
        label = match.group("label")
        if label not in labels:
            raise ValueError(f"unresolved equation label {label}")
        if not output or "$$" not in output:
            raise ValueError(f"equation label {label} is not attached to a display equation")
        number = labels[label]["number"]
        output.append(f"\\tag{{{number}}}\\label{{{label}}}\n$$")
        index += 1
    return "\n".join(output)


def _resolve_figure_and_table_captions(text: str, labels: dict[str, dict[str, int]]) -> str:
    lines = text.splitlines()
    output: list[str] = []
    image_re = re.compile(
        r"^(?P<prefix>!\[[^\]]*\]\([^)]*\))\{#(?P<label>fig:[A-Za-z0-9_-]+)(?P<attrs>[^}]*)\}$"
    )
    caption_re = re.compile(r"^: (?P<caption>.+) \{#(?P<label>tbl:[A-Za-z0-9_-]+\}\s*)$")
    for line in lines:
        image = image_re.match(line)
        if image:
            label = image.group("label")
            if label not in labels:
                raise ValueError(f"unresolved figure label {label}")
            prefix = image.group("prefix")
            output.append(prefix)
            continue
        caption = caption_re.match(line)
        if caption:
            label = caption.group("label").rstrip("}")
            if label not in labels:
                raise ValueError(f"unresolved table label {label}")
            output.append(f": {caption.group('caption')} {{#{label}}}")
            continue
        output.append(line)
    return "\n".join(output)


def compose_manuscript() -> tuple[str, dict[str, dict[str, int]]]:
    sources = _read_sources()
    labels = _collect_labels(sources)
    sections: list[str] = [
        "<!-- Generated by scripts/build_manuscript.py; edit docs/manuscript/[0-9][0-9]_*.md. -->",
        "<!-- Formalism labels and cross-references are resolved deterministically. -->",
        "",
    ]
    for path, source in sources:
        if path.name == "99_references.md":
            sections.append(source.strip())
            continue
        resolved = _resolve_formalism_divs(source, labels)
        resolved = _resolve_equations(resolved, labels)
        resolved = _resolve_references(resolved, labels)
        resolved = _resolve_figure_and_table_captions(resolved, labels)
        sections.append(f"<!-- source: {path.name} -->\n\n{resolved.strip()}")
    return "\n\n".join(sections) + "\n", labels


def _save_figure(fig: object, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, format="svg", bbox_inches="tight")
        _normalise_svg(path)
        fig.savefig(path.with_suffix(".png"), format="png", dpi=180, bbox_inches="tight")
    finally:
        plt.close(fig)


def _normalise_svg(path: Path) -> None:
    """Make a generated SVG stable across processes and build dates."""

    text = path.read_text(encoding="utf-8")
    text = re.sub(
        r"<dc:date>[^<]+</dc:date>",
        "<dc:date>1970-01-01T00:00:00</dc:date>",
        text,
    )
    dynamic_id = re.compile(r"(?:p|m)[0-9a-f]{10}|C\d+_\d+_[0-9a-f]{10}")
    replacements: dict[str, str] = {}

    def replace_id(match: re.Match[str]) -> str:
        value = match.group(0)
        if value not in replacements:
            replacements[value] = f"generated_{len(replacements) + 1}"
        return replacements[value]

    text = dynamic_id.sub(replace_id, text)
    lines = text.splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def _box(ax, x: float, y: float, width: float, height: float, label: str, color: str) -> None:
    from matplotlib.patches import FancyBboxPatch

    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor=color,
        edgecolor="#243447",
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=9, wrap=True)


def build_figures(output_dir: Path = FIGURE_DIR) -> list[Path]:
    """Generate the manuscript figures from the current implementation."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.axis("off")
    nodes = [
        (0.02, "observe", "#dbeafe"),
        (0.21, "infer q(s)", "#dcfce7"),
        (0.40, "evaluate G(π)", "#fef3c7"),
        (0.59, "sample u", "#fce7f3"),
        (0.78, "propagate B", "#ede9fe"),
    ]
    for x, label, color in nodes:
        _box(ax, x, 0.38, 0.15, 0.25, label, color)
    for (x, _, _), (next_x, _, _) in zip(nodes, nodes[1:], strict=False):
        ax.add_patch(
            FancyArrowPatch(
                (x + 0.15, 0.505),
                (next_x, 0.505),
                arrowstyle="->",
                mutation_scale=15,
                linewidth=1.4,
            )
        )
    ax.add_patch(
        FancyArrowPatch(
            (0.855, 0.38),
            (0.095, 0.38),
            connectionstyle="arc3,rad=0.35",
            arrowstyle="->",
            mutation_scale=15,
            linewidth=1.2,
            color="#475569",
        )
    )
    ax.text(
        0.50,
        0.03,
        "the persisted trajectory records every state, posterior, EFE vector, and sampled action",
        ha="center",
        fontsize=9,
        color="#334155",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    path = output_dir / "fig1_active_inference_loop.svg"
    _save_figure(fig, path)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(2.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.grid(True, color="#94a3b8", linewidth=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.add_patch(Rectangle((0.5, 0.5), 1, 1, facecolor="#e2e8f0", edgecolor="none", alpha=0.8))
    ax.annotate(
        "A → (1, 1)",
        xy=(1, 1),
        xytext=(0, 0),
        arrowprops={"arrowstyle": "->", "color": "#2563eb", "lw": 2},
        color="#1d4ed8",
        fontsize=10,
        ha="center",
    )
    ax.annotate(
        "B → (1, 1)",
        xy=(1, 1),
        xytext=(2, 2),
        arrowprops={"arrowstyle": "->", "color": "#dc2626", "lw": 2},
        color="#b91c1c",
        fontsize=10,
        ha="center",
    )
    ax.scatter(
        [0, 2], [0, 2], s=160, c=["#2563eb", "#dc2626"], edgecolor="white", linewidth=1.5, zorder=5
    )
    ax.text(0, 0, "A", color="white", ha="center", va="center", weight="bold", zorder=6)
    ax.text(2, 2, "B", color="white", ha="center", va="center", weight="bold", zorder=6)
    ax.set_title("Simultaneous collision resolution")
    path = output_dir / "fig2_collision_resolution.svg"
    _save_figure(fig, path)
    paths.append(path)

    from blockference.config import load_experiment_config
    from blockference.simulations.grid_sim import run_experiment

    cfg = load_experiment_config(REPO_ROOT / "configs" / "manuscript.yml")
    frame = run_experiment(cfg)

    def as_mapping(value):
        return value if isinstance(value, dict) else ast.literal_eval(str(value))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    try:
        positions = [as_mapping(value) for value in frame["env_states"]]
        for agent_id in sorted(positions[0], key=repr):
            ys = [state[agent_id][0] for state in positions]
            xs = [state[agent_id][1] for state in positions]
            axes[0].plot(xs, ys, "-o", label=f"agent {agent_id}")
        axes[0].set_xlim(-0.25, 2.25)
        axes[0].set_ylim(2.25, -0.25)
        axes[0].set_xticks(range(3))
        axes[0].set_yticks(range(3))
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title("Seeded paths")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].legend()
        efe_values: dict[object, list[float]] = defaultdict(list)
        for value in frame["efe"]:
            parsed = as_mapping(value)
            for agent_id, vector in parsed.items():
                if vector is not None:
                    efe_values[agent_id].append(min(map(float, vector)))
        for agent_id, values in efe_values.items():
            axes[1].plot(range(len(values)), values, "-o", label=f"agent {agent_id}")
        axes[1].set_title("Persisted policy EFE")
        axes[1].set_xlabel("step")
        axes[1].set_ylabel("min G(π) (nats)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        fig.tight_layout()
        path = output_dir / "fig3_seeded_run.svg"
        fig.savefig(path, format="svg", bbox_inches="tight")
        _normalise_svg(path)
        fig.savefig(path.with_suffix(".png"), format="png", dpi=180, bbox_inches="tight")
        paths.append(path)
    finally:
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.axis("off")
    labels = [
        (0.02, "config", "#dbeafe"),
        (0.21, "simulation", "#dcfce7"),
        (0.40, "CSV + JSON + NPZ", "#fef3c7"),
        (0.59, "PNG + GIF", "#fce7f3"),
        (0.78, "aggregate\nvalidation", "#ede9fe"),
    ]
    for x, label, color in labels:
        _box(ax, x, 0.40, 0.15, 0.22, label, color)
    for (x, _, _), (next_x, _, _) in zip(labels, labels[1:], strict=False):
        ax.add_patch(
            FancyArrowPatch(
                (x + 0.15, 0.51), (next_x, 0.51), arrowstyle="->", mutation_scale=15, linewidth=1.4
            )
        )
    ax.text(
        0.50,
        0.12,
        "A release is valid only when the trajectory, model, diagnostics, artefacts, and renders agree",
        ha="center",
        fontsize=9,
        color="#334155",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    path = output_dir / "fig4_release_contract.svg"
    _save_figure(fig, path)
    paths.append(path)
    return paths


def _write_outputs(compiled: str) -> Path:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    output = BUILD_DIR / "manuscript.md"
    output.write_text(compiled, encoding="utf-8")
    build_figures_dir = BUILD_DIR / "figures"
    build_figures_dir.mkdir(parents=True, exist_ok=True)
    for figure in FIGURE_DIR.glob("*.svg"):
        shutil.copy2(figure, build_figures_dir / figure.name)
    for figure in FIGURE_DIR.glob("*.png"):
        shutil.copy2(figure, build_figures_dir / figure.name)
    return output


def _check_figures_against_committed() -> list[str]:
    """Return committed figure filenames that drift from a fresh, content-identical build.

    Regenerates the figures into a temporary directory and compares every
    committed SVG/PNG byte-for-byte against the fresh output. This is a
    checkout-independent drift baseline: it does not depend on ``git diff``
    state, so a stale or hand-edited committed figure is always caught.
    """
    import tempfile
    from filecmp import cmp
    from pathlib import Path

    drift: list[str] = []
    with tempfile.TemporaryDirectory(prefix="activeblockference-figcheck-") as directory:
        build_figures(Path(directory))
        produced = {path.name: path for path in Path(directory).iterdir() if path.is_file()}
        for committed in sorted(FIGURE_DIR.iterdir()):
            if not committed.is_file():
                continue
            fresh = produced.get(committed.name)
            if fresh is None or not cmp(fresh, committed, shallow=False):
                drift.append(committed.name)
    return drift


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate source and existing publication assets without writing",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="compose the manuscript without regenerating figures",
    )
    args = parser.parse_args(argv)
    try:
        compiled, labels = compose_manuscript()
        if not args.check:
            if not args.no_figures:
                build_figures()
            output = _write_outputs(compiled)
            print(f"wrote {output}")
        else:
            missing = [
                str(path)
                for path in (
                    FIGURE_DIR / f"fig{i}_{name}.svg"
                    for i, name in (
                        (1, "active_inference_loop"),
                        (2, "collision_resolution"),
                        (3, "seeded_run"),
                        (4, "release_contract"),
                    )
                )
                if not path.is_file()
            ]
            if missing:
                raise FileNotFoundError("missing manuscript figures: " + ", ".join(missing))
            if not (BUILD_DIR / "manuscript.md").is_file():
                raise FileNotFoundError(
                    f"missing composed manuscript: {BUILD_DIR / 'manuscript.md'}"
                )
            if (BUILD_DIR / "manuscript.md").read_text(encoding="utf-8") != compiled:
                raise ValueError(
                    "composed manuscript is out of date; run scripts/build_manuscript.py"
                )
            figure_drift = _check_figures_against_committed()
            if figure_drift:
                raise ValueError(
                    "committed figures drift from a fresh build: " + ", ".join(figure_drift)
                )
            print(f"manuscript check passed ({len(labels)} numbered labels)")
        return 0
    except (FileNotFoundError, ValueError, TypeError) as exc:
        print(f"manuscript build failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
