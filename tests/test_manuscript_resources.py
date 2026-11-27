"""Source-relative resource relocation and metadata-selected publication figures."""

from __future__ import annotations

import json
import re
import shutil
from urllib.parse import unquote, urlsplit

import numpy as np
import pandas as pd
import pytest
import yaml

import scripts.build_manuscript as builder
import scripts.validate_manuscript as validator
from blockference.config import ExperimentConfig
from blockference.manuscript_links import rebase_local_links


def _publication(tmp_path, monkeypatch):
    original_root = builder.REPO_ROOT
    original_manuscript = builder.MANUSCRIPT_DIR
    root = tmp_path / "publication"
    manuscript = root / "docs" / "manuscript"
    manuscript.mkdir(parents=True)
    for source in [
        *original_manuscript.glob("[0-9][0-9]_*.md"),
        original_manuscript / "references.bib", original_manuscript / "config.yaml",
        original_manuscript / "figure-registry.json",
    ]:
        shutil.copy2(source, manuscript / source.name)
    figures = manuscript / "figures"
    figures.mkdir()
    for source in builder.FIGURE_DIR.iterdir():
        if source.suffix in {".svg", ".png"}:
            shutil.copy2(source, figures / source.name)
    reference = root / "docs" / "reference"
    reference.mkdir()
    shutil.copy2(original_root / "docs" / "reference" / "gnn.md", reference / "gnn.md")
    (root / ".github").mkdir()
    shutil.copy2(original_root / ".github" / "GNN.md", root / ".github" / "GNN.md")
    (root / "configs").mkdir()
    shutil.copy2(original_root / "configs" / "manuscript.yml", root / "configs" / "manuscript.yml")
    output = root / "relocated" / "deep" / "output"
    monkeypatch.setattr(builder, "REPO_ROOT", root)
    monkeypatch.setattr(builder, "MANUSCRIPT_DIR", manuscript)
    monkeypatch.setattr(builder, "FIGURE_DIR", figures)
    monkeypatch.setattr(builder, "BUILD_DIR", output)
    monkeypatch.setattr(validator, "ROOT", root)
    monkeypatch.setattr(validator, "MANUSCRIPT", manuscript)
    return root, manuscript, output


def test_relocated_real_composition_resolves_gnn_bibliography_and_copied_figures(tmp_path, monkeypatch):
    root, manuscript, output = _publication(tmp_path, monkeypatch)
    compiled, _labels = builder.compose_manuscript()
    path = builder._write_outputs(compiled)
    targets = set()
    for destination in re.findall(r"!?\[[^\]]*\]\(([^\s)]+)", compiled):
        parsed = urlsplit(destination)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        target = (path.parent / unquote(parsed.path)).resolve()
        assert target.is_file(), destination
        targets.add(target)
    assert root / "docs" / "reference" / "gnn.md" in targets
    assert root / ".github" / "GNN.md" in targets
    assert output / "references.bib" in targets
    assert (output / "references.bib").read_bytes() == (manuscript / "references.bib").read_bytes()
    assert (output / "config.yaml").read_bytes() == (manuscript / "config.yaml").read_bytes()
    registry = json.loads((manuscript / "figure-registry.json").read_text(encoding="utf-8"))
    for entry in registry["figures"]:
        assert output / "figures" / entry["png"] in targets


def test_inline_reference_and_escaped_parenthesis_destinations_keep_suffixes_and_titles(tmp_path):
    source = tmp_path / "sources" / "manuscript" / "section.md"
    source.parent.mkdir(parents=True)
    spaced = source.parent.parent / "a file(v1).md"
    spaced.write_text("# Target\n", encoding="utf-8")
    parenthesized = source.parent / "manual(v2).md"
    parenthesized.write_text("# Part\n", encoding="utf-8")
    figure = source.parent / "figures" / "chart.png"
    figure.parent.mkdir()
    figure.write_bytes(b"resource fixture")
    output = tmp_path / "compiled" / "nested"
    output.mkdir(parents=True)
    text = (
        '[inline](<../a%20file(v1).md?version=1#target> "A title")\n'
        '[manual]: <../a%20file(v1).md?version=2#target> "Reference title"\n'
        '[plain](manual(v2).md#part)\n'
        '[escaped](manual\\(v2\\).md#part)\n'
        '![chart](figures/chart.png)\n'
    )
    rendered = rebase_local_links(text, source=source, output_dir=output, copied={figure: output / "figures" / "chart.png"})
    assert '?version=1#target> "A title"' in rendered
    assert '?version=2#target> "Reference title"' in rendered
    destinations = re.findall(r"<([^>]+)>|\]\(([^<\s)]+)", rendered)
    resolved = [
        (output / unquote(urlsplit(first or second).path)).resolve()
        for first, second in destinations
    ]
    assert resolved == [spaced, spaced, parenthesized, parenthesized, output / "figures" / "chart.png"]
    assert "#part" in rendered
    assert "# Target" in spaced.read_text(encoding="utf-8")
    assert "# Part" in parenthesized.read_text(encoding="utf-8")


def test_external_links_fragments_citations_and_code_are_not_rewritten(tmp_path):
    original = (
        '[web](https://example.invalid/a(b)?v=1#part)\n'
        '[mail](mailto:someone@example.invalid) [doi](doi:10.1000/example)\n'
        '[cdn](//example.invalid/a) [local](#sec:overview) [@Scholar2020] [@eq:belief]\n'
        '`[inline code](missing.md)`\n'
        '````markdown\n[fenced](also-missing.md)\n````\n'
    )
    assert rebase_local_links(
        original, source=tmp_path / "section.md", output_dir=tmp_path / "elsewhere",
    ) == original


def test_missing_local_resource_refuses_composition_without_a_fake_target(tmp_path):
    with pytest.raises(FileNotFoundError):
        rebase_local_links(
            "[missing](no-such-file.md#anchor)",
            source=tmp_path / "section.md", output_dir=tmp_path / "compiled",
        )


@pytest.mark.slow
def test_alternate_metadata_config_reaches_experiment_and_dimension_four_run_plots(tmp_path, monkeypatch):
    root, manuscript, _output = _publication(tmp_path, monkeypatch)
    config = ExperimentConfig.from_dict({
        "name": "selected", "engine": "cadcad", "seed": 19,
        "grid": {"dimension": 4, "planning_length": 1, "max_policies": 20,
                 "affordances": ["RIGHT", "STAY"]},
        "simulation": {"timesteps": 2, "runs": 2, "n_agents": 1, "target": [3, 3]},
        "output": {"path": None},
    })
    selected = root / "configs" / "selected.yml"
    selected.write_text(yaml.safe_dump(config.to_dict()), encoding="utf-8")
    metadata_path = manuscript / "config.yaml"
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    metadata["experiment"].update(
        config_path="configs/selected.yml", engine="cadcad", dimension=4,
        planning_length=1, max_policies=20, timesteps=2, runs=2, agents=1,
        seed=19, target=[3, 3], initial_states=[[0, 0]], affordances=["RIGHT", "STAY"],
    )
    metadata_path.write_text(yaml.safe_dump(metadata), encoding="utf-8")
    admitted = []

    def synthetic(selected_config):
        admitted.append(selected_config)
        return pd.DataFrame([
            {
                "run": run, "timestep": time, "substep": int(time > 0),
                "env_states": {0: (0, time) if run == 1 else (time, 0)},
                "inferences": {0: np.eye(16)[0].tolist() if time else None},
                "efe": {0: [float(run + time)] * 2 if time else None},
            }
            for run in (1, 2) for time in range(3)
        ])

    monkeypatch.setattr("blockference.simulations.grid_sim.run_experiment", synthetic)
    observations = {}
    save = builder._save_figure

    def observe(fig, path):
        if path.name == "fig3_seeded_run.svg":
            axis = fig.axes[0]
            observations.update(
                xlim=axis.get_xlim(), ylim=axis.get_ylim(), ticks=axis.get_xticks().tolist(),
                lines={line.get_label(): line.get_xydata().tolist() for line in axis.lines},
            )
        return save(fig, path)

    monkeypatch.setattr(builder, "_save_figure", observe)
    paths = builder.build_figures(tmp_path / "fresh-figures")
    assert admitted[0].to_dict() == config.to_dict()
    assert observations["xlim"] == (-0.25, 3.25)
    assert observations["ylim"] == (3.25, -0.25)
    assert observations["ticks"] == [0, 1, 2, 3]
    assert observations["lines"] == {
        "run 1 / agent 0": [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        "run 2 / agent 0": [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]],
    }
    assert all(path.with_suffix(".png").is_file() for path in paths)


def test_metadata_mirror_disagreement_remains_an_explicit_failure(tmp_path, monkeypatch):
    _root, manuscript, _output = _publication(tmp_path, monkeypatch)
    metadata_path = manuscript / "config.yaml"
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    metadata["experiment"]["dimension"] = 999
    metadata_path.write_text(yaml.safe_dump(metadata), encoding="utf-8")
    errors = validator.validate()
    assert any("metadata disagrees" in error and "dimension" in error for error in errors)
