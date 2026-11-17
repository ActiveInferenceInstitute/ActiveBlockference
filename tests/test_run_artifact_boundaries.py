"""Synthetic multi-run producer/consumer round trips and refreshed-manifest attacks."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from blockference.cli import main
from blockference.config import ExperimentConfig
from blockference.gridference import ActiveGridference, make_grid
from blockference.io import build_run_paths, persist_manifest, validate_run_outputs
from blockference.io.validate import validate_manifest_boundary
from blockference.pipeline import _extract_policies, persist, summarise_trajectory
from blockference.viz import (
    animate_trajectory,
    plot_action_distribution,
    plot_belief_heatmap,
    plot_efe,
    plot_trajectory,
)
from blockference.viz._common import extract_agent_positions, extract_belief_history


def _fixture_frame():
    config = ExperimentConfig.from_dict({
        "name": "synthetic", "seed": None,
        "grid": {"dimension": 2, "planning_length": 2, "max_policies": 16,
                 "affordances": ["DOWN", "RIGHT", "STAY"]},
        "simulation": {"timesteps": 2, "runs": 2, "n_agents": 1, "target": "random"},
        "output": {"path": None},
    })
    rows = []
    for run, destination in ((1, (0, 1)), (2, (1, 0))):
        agent = ActiveGridference(
            make_grid(2), planning_length=2, affordances=config.grid.affordances, max_policies=16,
        )
        agent.get_D((0, 0))
        agent.get_C((0, 1) if run == 1 else (1, 1))
        previous = (0, 0)
        for time in range(3):
            position = (0, 0) if time == 0 else destination
            posterior = np.eye(4)[previous[0] * 2 + previous[1]].tolist()
            prior = np.eye(4)[position[0] * 2 + position[1]].tolist()
            energy = [run + time / 10] * 9
            action = (1 if time == 1 else 2) if run == 1 else 0
            diagnostics = {
                "actions": action, "inferences": posterior, "efe": energy,
                "efe_epistemic": [0.0] * 9, "efe_pragmatic": energy,
                "q_pi": [1 / 9] * 9, "p_u": [1 / 3] * 3,
                "obs_idx": previous[0] * 2 + previous[1],
            }
            rows.append({
                "agents": {0: agent}, "priors": {0: prior}, "env_states": {0: position},
                **{key: {0: value if time else None} for key, value in diagnostics.items()},
                "run": run, "timestep": time, "substep": int(time > 0),
            })
            previous = position
    return config, pd.DataFrame(rows)


def _fresh(paths):
    persist_manifest(paths)
    assert validate_manifest_boundary(paths).ok


@pytest.fixture
def run_tree(tmp_path):
    config, frame = _fixture_frame()
    paths = build_run_paths(tmp_path, "synthetic").ensure()
    persist(config, frame, paths)
    image = Image.new("RGB", (8, 8), "white")
    for name in ("trajectory.png", "action_distribution.png", "efe.png"):
        image.save(paths.viz_dir / name)
    for run in (1, 2):
        for label in ("first", "last"):
            image.save(paths.viz_dir / f"belief_heatmap_run_{run}_agent_0_{label}.png")
    image.save(paths.animations_dir / "trajectory.gif")
    paths.run_log.write_text("synthetic artifact fixture\n", encoding="utf-8")
    _fresh(paths)
    assert validate_run_outputs(paths).ok
    return config, frame, paths


def _model_json(paths):
    return json.loads(paths.matrices_json.read_text(encoding="utf-8"))


def _write_models(paths, payload):
    paths.matrices_json.write_text(json.dumps(payload), encoding="utf-8")
    arrays = {
        f"run_{run}/agent_{agent}/{name}": np.asarray(value[name])
        for run, agents in payload["runs"].items() for agent, value in agents.items()
        for name in ("A", "B", "C", "D", "E")
    }
    np.savez_compressed(paths.matrices_npz, **arrays)


def test_distinct_run_models_policies_and_summaries_round_trip(run_tree):
    _config, frame, paths = run_tree
    models = _model_json(paths)["runs"]
    assert set(models) == {"1", "2"}
    assert models["1"]["0"]["C"] != models["2"]["0"]["C"]
    with np.load(paths.matrices_npz, allow_pickle=False) as archive:
        assert np.array_equal(archive["run_1/agent_0/C"], models["1"]["0"]["C"])
        assert np.array_equal(archive["run_2/agent_0/C"], models["2"]["0"]["C"])
    policies = json.loads(paths.policies_json.read_text(encoding="utf-8"))
    for run in ("1", "2"):
        assert policies["runs"][run]["0"]["n_policies"] == 9
    summary = json.loads(paths.summary_json.read_text(encoding="utf-8"))
    assert summary["n_agents"] == 1 and summary["n_runs"] == 2
    assert summary["runs"]["1"]["final_positions"] == {"0": [0, 1]}
    assert summary["runs"]["2"]["final_positions"] == {"0": [1, 0]}
    assert summary["runs"]["1"]["trajectory_lengths"] == {"0": 3}
    assert summary["runs"]["2"]["belief_history_lengths"] == {"0": 2}
    shuffled = frame.sample(frac=1, random_state=7)
    assert extract_agent_positions(shuffled) == extract_agent_positions(frame)
    assert len(extract_belief_history(shuffled)[(1, 0)]) == 2


def test_swapped_json_and_npz_models_fail_even_after_manifest_refresh(run_tree):
    _config, _frame, paths = run_tree
    payload = _model_json(paths)
    payload["runs"]["1"], payload["runs"]["2"] = payload["runs"]["2"], payload["runs"]["1"]
    _write_models(paths, payload)
    _fresh(paths)
    report = validate_run_outputs(paths)
    assert not report.ok
    assert report.checks["artefacts.generative_model_npz_values"]
    assert not report.checks["artefacts.model_run_association"]


@pytest.mark.parametrize("change", ["truncate-step", "truncate-run", "relabel-run"])
def test_configured_schedule_rejects_coherent_history_rewrites(run_tree, change):
    config, frame, paths = run_tree
    if change == "truncate-step":
        changed = frame.loc[frame["timestep"] < 2].copy()
    elif change == "truncate-run":
        changed = frame.loc[frame["run"] == 1].copy()
    else:
        changed = frame.copy()
        changed.loc[changed["run"] == 2, "run"] = 3
    persist(config, changed, paths)
    _fresh(paths)
    report = validate_run_outputs(paths)
    assert not report.ok
    assert not report.checks["artefacts.configured_schedule"]


def test_duplicate_trajectory_identity_is_not_hidden_by_row_count(run_tree):
    _config, _frame, paths = run_tree
    frame = pd.read_csv(paths.trajectory_csv)
    frame.loc[2] = frame.loc[1]
    frame.to_csv(paths.trajectory_csv, index=False)
    _fresh(paths)
    report = validate_run_outputs(paths)
    assert not report.ok
    assert not report.checks["artefacts.trajectory.unique_row_identities"]


def test_every_run_initial_position_and_prior_are_checked(run_tree):
    _config, _frame, paths = run_tree
    frame = pd.read_csv(paths.trajectory_csv)
    index = frame.index[(frame["run"] == 2) & (frame["timestep"] == 0)][0]
    frame.at[index, "env_states"] = repr({"0": [1, 1]})
    frame.at[index, "priors"] = repr({"0": [0.5, 0.5]})
    frame.to_csv(paths.trajectory_csv, index=False)
    _fresh(paths)
    report = validate_run_outputs(paths)
    assert not report.ok
    assert not report.checks["artefacts.configured_initial_states"]
    assert not report.checks["artefacts.model_initial_priors"]


@pytest.mark.parametrize("change", ["duplicate", "missing", "wrong-agent"])
def test_diagnostics_are_one_to_one_by_configured_identity(run_tree, change):
    _config, _frame, paths = run_tree
    records = pd.read_csv(paths.per_step_csv)
    if change == "duplicate":
        records.loc[1] = records.loc[0]
    elif change == "missing":
        records = records.iloc[:-1]
    else:
        records.at[0, "agent_id"] = 1
    records.to_csv(paths.per_step_csv, index=False)
    _fresh(paths)
    report = validate_run_outputs(paths)
    assert not report.ok
    assert not report.checks["artefacts.per_step_coverage"]


def test_same_shape_cross_run_diagnostic_swap_cannot_qualify(run_tree):
    _config, _frame, paths = run_tree
    records = pd.read_csv(paths.per_step_csv)
    columns = ["efe", "efe_epistemic", "efe_pragmatic"]
    first = records.loc[0, columns].copy()
    records.loc[0, columns] = records.loc[2, columns].to_numpy()
    records.loc[2, columns] = first.to_numpy()
    records.to_csv(paths.per_step_csv, index=False)
    _fresh(paths)
    report = validate_run_outputs(paths)
    assert not report.ok
    assert report.checks["artefacts.per_step_coverage"]
    assert not report.checks["artefacts.per_step_trajectory_values"]


@pytest.mark.parametrize("column", ["actions", "obs_idx"])
@pytest.mark.parametrize("convert", [str, float, bool], ids=["string", "float", "boolean"])
def test_categorical_trajectory_types_cannot_be_hidden_by_numeric_agreement(
    run_tree, column, convert,
):
    _config, original, paths = run_tree
    frame = pd.read_csv(paths.trajectory_csv)
    index = frame.index[(frame["run"] == 1) & (frame["timestep"] == 1)][0]
    frame.at[index, column] = repr({"0": convert(original.at[index, column][0])})
    frame.to_csv(paths.trajectory_csv, index=False)
    _fresh(paths)
    report = validate_run_outputs(paths)
    assert not report.ok
    assert not report.checks["artefacts.per_step_trajectory_values"]


def test_policy_vectors_are_policy_sized_not_merely_action_sized(run_tree):
    _config, _frame, paths = run_tree
    records = pd.read_csv(paths.per_step_csv)
    trajectory = pd.read_csv(paths.trajectory_csv)
    for field in ("efe", "efe_epistemic", "efe_pragmatic", "q_pi"):
        value = [1 / 3] * 3 if field == "q_pi" else [0.0] * 3
        records.at[0, field] = repr(value)
        trajectory.at[1, field] = repr({"0": value})
    records.to_csv(paths.per_step_csv, index=False)
    trajectory.to_csv(paths.trajectory_csv, index=False)
    _fresh(paths)
    report = validate_run_outputs(paths)
    assert not report.ok
    assert not report.checks["artefacts.per_step.0.model_dimensions"]


@pytest.mark.parametrize("field", ["E", "policy_len", "max_policies"])
def test_model_affordance_order_and_planning_metadata_match_configuration(run_tree, field):
    _config, _frame, paths = run_tree
    payload = _model_json(paths)
    model = payload["runs"]["1"]["0"]
    model[field] = list(reversed(model[field])) if field == "E" else model[field] + 1
    _write_models(paths, payload)
    _fresh(paths)
    report = validate_run_outputs(paths)
    assert not report.ok
    assert not report.checks["artefacts.configured_model_dimensions"]


def test_missing_run_model_and_policy_reordering_fail_with_fresh_digests(run_tree):
    _config, _frame, paths = run_tree
    payload = _model_json(paths)
    del payload["runs"]["2"]
    _write_models(paths, payload)
    policies = json.loads(paths.policies_json.read_text(encoding="utf-8"))
    policies["runs"]["1"]["0"]["policies"].reverse()
    paths.policies_json.write_text(json.dumps(policies), encoding="utf-8")
    _fresh(paths)
    report = validate_run_outputs(paths)
    assert not report.ok
    assert not report.checks["artefacts.configured_model_agents"]
    assert not report.checks["artefacts.policies_complete_ordered"]


@pytest.mark.parametrize("fault", ["scalar-policies", "missing-timestep", "nonmapping-state", "scalar-matrix", "invalid-utf8"])
def test_malformed_artifacts_return_failed_cli_json_not_tracebacks(run_tree, capsys, fault):
    _config, _frame, paths = run_tree
    if fault == "scalar-policies":
        payload = json.loads(paths.policies_json.read_text(encoding="utf-8"))
        payload["runs"]["1"]["0"]["policies"] = 9
        paths.policies_json.write_text(json.dumps(payload), encoding="utf-8")
    elif fault in {"missing-timestep", "nonmapping-state"}:
        frame = pd.read_csv(paths.trajectory_csv)
        if fault == "missing-timestep":
            frame = frame.drop(columns="timestep")
        else:
            frame.at[0, "env_states"] = "[]"
        frame.to_csv(paths.trajectory_csv, index=False)
    elif fault == "scalar-matrix":
        payload = _model_json(paths)
        payload["runs"]["1"]["0"]["A"] = 0.5
        _write_models(paths, payload)
    else:
        paths.summary_json.write_bytes(b"\xff")
    _fresh(paths)
    assert main(["validation", "--run-dir", str(paths.run_dir)]) == 1
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) == 1
    report = json.loads(lines[0])
    assert report["ok"] is False and report["issues"]
    assert report["checks"]["artefacts.manifest.digests"]


def test_required_heatmap_coverage_is_run_scoped(run_tree):
    _config, _frame, paths = run_tree
    (paths.viz_dir / "belief_heatmap_run_2_agent_0_last.png").unlink()
    _fresh(paths)
    report = validate_run_outputs(paths)
    assert not report.ok
    assert not report.checks["artefacts.render.belief_heatmap_run_2_agent_0_last.png"]


def test_policy_extraction_enforces_actual_cap_and_preserves_allowed_order():
    agent = ActiveGridference(make_grid(2), affordances=["RIGHT", "STAY"], planning_length=3, max_policies=8)
    extracted = _extract_policies({(2, "0"): agent})[(2, "0")]
    assert [tuple(policy[:, 0]) for policy in extracted] == list(itertools.product(range(2), repeat=3))
    agent.max_policies = 4
    with pytest.raises(ValueError, match="max_policies=4"):
        _extract_policies({(2, "0"): agent})


def test_run_scoped_plots_use_actual_times_and_never_connect_runs(tmp_path, monkeypatch):
    import blockference.viz.plots as plots

    _config, frame = _fixture_frame()
    captures = {}
    save = plots.atomic_savefig

    def capture(fig, path, **kwargs):
        axis = fig.axes[0]
        captures[Path(path).name] = {
            "lines": {line.get_label(): line.get_xydata().tolist() for line in axis.lines},
            "bars": {container.get_label(): [bar.get_height() for bar in container] for container in axis.containers},
            "images": [np.asarray(image.get_array()).tolist() for image in axis.images],
            "title": axis.get_title(),
        }
        return save(fig, path, **kwargs)

    monkeypatch.setattr(plots, "atomic_savefig", capture)
    shuffled = frame.sample(frac=1, random_state=5)
    plot_trajectory(shuffled, tmp_path / "paths.png")
    plot_efe(shuffled, tmp_path / "energy.png")
    plot_action_distribution(shuffled, tmp_path / "actions.png", affordances=["DOWN", "RIGHT", "STAY"])
    plot_belief_heatmap(shuffled, tmp_path / "posterior.png", run=2, agent_id=0)
    assert captures["paths.png"]["lines"] == {
        "run 1 / agent 0": [[0.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
        "run 2 / agent 0": [[0.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
    }
    assert [point[0] for point in captures["energy.png"]["lines"]["run 2 / agent 0"]] == [1.0, 2.0]
    assert captures["actions.png"]["bars"] == {"run 1 / agent 0": [0, 1, 1], "run 2 / agent 0": [2, 0, 0]}
    assert captures["posterior.png"]["images"] == [[[0.0, 0.0], [1.0, 0.0]]]
    assert "run 2 / agent 0 @ t=2, substep=1" in captures["posterior.png"]["title"]
    summary = summarise_trajectory(shuffled)
    assert summary["runs"]["1"]["trajectory_lengths"] == {"0": 3}
    with pytest.raises(ValueError, match="ambiguous"):
        plot_belief_heatmap(shuffled, tmp_path / "ambiguous.png", agent_id=0)


def test_animation_keeps_separate_paths_and_actual_run_time_labels(tmp_path, monkeypatch):
    import blockference.viz.animation as animation

    _config, frame = _fixture_frame()
    snapshots = {}
    implementation = animation.FuncAnimation

    def observe(fig, update, **kwargs):
        def wrapped(index):
            artists = update(index)
            axis = fig.axes[0]
            snapshots[index] = {
                "lines": {line.get_label(): line.get_xydata().tolist() for line in axis.lines if line.get_label().startswith("run ")},
                "labels": [text.get_text() for text in axis.texts],
            }
            return artists
        return implementation(fig, wrapped, **kwargs)

    monkeypatch.setattr(animation, "FuncAnimation", observe)
    output = animate_trajectory(frame.sample(frac=1, random_state=3), tmp_path / "runs.gif")
    assert len(snapshots[2]["lines"]["run 1 / agent 0"]) == 3
    assert len(snapshots[2]["lines"]["run 2 / agent 0"]) == 3
    assert any("run 1: t=2, substep=1" in text and "run 2: t=2, substep=1" in text for text in snapshots[2]["labels"])
    with Image.open(output) as image:
        assert image.format == "GIF" and image.n_frames > 1
