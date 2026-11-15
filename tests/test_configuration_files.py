"""Mapping-only file admission and one metadata-selected typed experiment."""

from __future__ import annotations

import pytest
import yaml

from blockference.config import (
    ExperimentConfig,
    load_experiment_config,
    load_manuscript_experiment,
)
from blockference.pipeline import run_pipeline


@pytest.mark.parametrize("extension", ["yml", "yaml"])
@pytest.mark.parametrize("content", ["[]", "false", "0", "0.0", "''", "[1]", "plain text"])
def test_nonmapping_yaml_roots_fail_before_any_output_tree(tmp_path, extension, content):
    path = tmp_path / f"invalid.{extension}"
    path.write_text(content, encoding="utf-8")
    output = tmp_path / "uncreated-output"
    with pytest.raises(TypeError, match="config root must be a mapping"):
        run_pipeline(path, output_root=output)
    assert not output.exists()


@pytest.mark.parametrize("content", ["", "# comment only\n", "null", "~", "{}"])
def test_empty_and_null_yaml_keep_explicit_default_compatibility(tmp_path, content):
    path = tmp_path / "empty.yml"
    path.write_text(content, encoding="utf-8")
    assert load_experiment_config(path).to_dict() == ExperimentConfig().to_dict()


def test_yaml_toml_and_metadata_selection_share_typed_defaults(tmp_path):
    configs = tmp_path / "configs"
    configs.mkdir()
    toml = configs / "alternate.toml"
    toml.write_text(
        'engine = "CaDcAd"\n[grid]\ndimension = 4\nplanning_length = 1\n'
        'affordances = ["RIGHT", "STAY"]\n[simulation]\ntarget = [3, 3]\n',
        encoding="utf-8",
    )
    selected, config = load_manuscript_experiment(
        tmp_path, {"experiment": {"config_path": "configs/alternate.toml"}},
    )
    assert selected == toml
    assert config.engine == "cadcad"
    assert config.grid.dimension == 4
    assert config.grid.affordances == ("RIGHT", "STAY")
    assert config.simulation.timesteps == ExperimentConfig().simulation.timesteps
    equivalent = configs / "same.yaml"
    equivalent.write_text(yaml.safe_dump(config.to_dict()), encoding="utf-8")
    assert load_experiment_config(equivalent).to_dict() == config.to_dict()


@pytest.mark.parametrize("metadata", [
    None, [], {}, {"experiment": []}, {"experiment": {"config_path": ""}},
    {"experiment": {"config_path": "../outside.yml"}},
])
def test_missing_or_escaping_metadata_never_selects_a_default_config(tmp_path, metadata):
    root = tmp_path / "repo"
    root.mkdir()
    (tmp_path / "outside.yml").write_text("{}", encoding="utf-8")
    with pytest.raises((TypeError, ValueError)):
        load_manuscript_experiment(root, metadata)


def test_absolute_missing_and_symlinked_outside_selections_refuse(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside.yml"
    outside.write_text("{}", encoding="utf-8")
    linked = root / "linked.yml"
    linked.symlink_to(outside)
    for selection in (str(outside), "linked.yml"):
        with pytest.raises(ValueError):
            load_manuscript_experiment(root, {"experiment": {"config_path": selection}})
    with pytest.raises(FileNotFoundError):
        load_manuscript_experiment(root, {"experiment": {"config_path": "missing.yml"}})


def test_nested_null_compatibility_does_not_weaken_unknown_key_rejection(tmp_path):
    path = tmp_path / "nested.yml"
    path.write_text("grid: null\nsimulation: null\noutput: null\n", encoding="utf-8")
    assert load_experiment_config(path).to_dict() == ExperimentConfig().to_dict()
    path.write_text("unexpected: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown configuration"):
        load_experiment_config(path)
