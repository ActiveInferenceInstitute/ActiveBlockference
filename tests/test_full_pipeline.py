"""Comprehensive tests for the extended pipeline:

* EFE / Q_pi / P_u per-step capture and persistence
* generative_model.npz round-trip
* per-step posterior + EFE-decomposition validation
* composable stage API (simulate / persist / validate)
* engine selection (radcad + cadcad)
"""

import json

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Per-step diagnostic capture
# ---------------------------------------------------------------------------


def _smoke_cfg():
    from blockference.config import ExperimentConfig

    return ExperimentConfig.from_dict({
        "name": "fullpipe_smoke",
        "seed": 0,
        "engine": "radcad",
        "grid": {"dimension": 3, "planning_length": 1},
        "simulation": {
            "timesteps": 2,
            "n_agents": 1,
            "target": [2, 2],
            "initial_state": [0, 0],
        },
        "output": {"path": "ignored.csv"},
    })


def test_engine_field_round_trips_and_validates():
    from blockference.config import ExperimentConfig

    cfg = ExperimentConfig.from_dict({"engine": "RADCAD"})
    assert cfg.engine == "radcad"

    with pytest.raises(ValueError, match="unknown engine"):
        ExperimentConfig.from_dict({"engine": "ray"})


def test_simulate_emits_all_diagnostic_columns():
    pytest.importorskip("radcad")
    from blockference.pipeline import simulate

    df = simulate(_smoke_cfg())
    for col in ("efe", "efe_epistemic", "efe_pragmatic",
                "q_pi", "p_u", "obs_idx"):
        assert col in df.columns, f"missing diagnostic column {col}"
    # At least one post-init row carries a non-empty diagnostic dict.
    later = df[df["timestep"] > 0]
    assert len(later) > 0
    sample = later.iloc[0]["efe"]
    assert isinstance(sample, dict) and sample, "efe column should be {agent_id: G(π)}"


def test_build_per_step_records_carries_full_diagnostic_set():
    pytest.importorskip("radcad")
    from blockference.pipeline import build_per_step_records, simulate

    df = simulate(_smoke_cfg())
    records = build_per_step_records(df)
    later = [r for r in records if r["timestep"] > 0]
    assert later, "expected per-step records past t=0"
    r = later[0]
    for key in ("posterior", "prior", "efe", "efe_epistemic", "efe_pragmatic",
                "q_pi", "p_u", "obs_idx",
                "posterior_entropy", "prior_entropy",
                "efe_min", "efe_argmin", "q_pi_entropy", "p_u_entropy"):
        assert key in r, f"per-step record missing {key}"

    # Decomposition is self-consistent.
    efe = np.asarray(r["efe"])
    eps = np.asarray(r["efe_epistemic"])
    prag = np.asarray(r["efe_pragmatic"])
    assert np.allclose(efe, eps + prag)
    # Policy posterior & action marginal are stochastic.
    assert np.isclose(np.asarray(r["q_pi"]).sum(), 1.0)
    assert np.isclose(np.asarray(r["p_u"]).sum(), 1.0)


# ---------------------------------------------------------------------------
# generative_model.npz round-trip
# ---------------------------------------------------------------------------


def test_persist_generative_model_npz_round_trip(tmp_path):
    from blockference.gridference import ActiveGridference, make_grid
    from blockference.io import build_run_paths, persist_generative_model_npz

    grid = make_grid(3, 2)
    a = ActiveGridference(grid)
    a.get_C((2, 2))
    a.get_D((0, 0))

    paths = build_run_paths(root=tmp_path, run_name="r")
    out = persist_generative_model_npz({0: a}, paths)
    assert out.exists()

    loaded = np.load(out, allow_pickle=False)
    assert "agent_0/A" in loaded.files
    assert "agent_0/B" in loaded.files
    assert "agent_0/C" in loaded.files
    assert "agent_0/D" in loaded.files
    # Lossless round-trip
    assert np.array_equal(loaded["agent_0/A"], a.A)
    assert np.array_equal(loaded["agent_0/B"], a.B)
    assert np.array_equal(loaded["agent_0/C"], a.C)
    assert np.array_equal(loaded["agent_0/D"], a.D)


# ---------------------------------------------------------------------------
# Per-step validator
# ---------------------------------------------------------------------------


def test_validate_per_step_records_passes_for_valid_records():
    from blockference.io import validate_per_step_records

    n_states = 4
    qs = np.full(n_states, 1.0 / n_states)
    prior = qs.copy()
    eps = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    prag = np.array([1.0, 0.9, 0.8, 0.7, 0.6])
    efe = eps + prag
    q_pi = np.exp(-efe) / np.exp(-efe).sum()
    p_u = q_pi.copy()

    records = [
        {
            "posterior": qs.tolist(),
            "prior": prior.tolist(),
            "efe": efe.tolist(),
            "efe_epistemic": eps.tolist(),
            "efe_pragmatic": prag.tolist(),
            "q_pi": q_pi.tolist(),
            "p_u": p_u.tolist(),
        }
    ]
    rep = validate_per_step_records(records)
    assert rep.ok, f"unexpected validation issues: {rep.issues}"


def test_validate_per_step_records_flags_bad_decomposition():
    from blockference.io import validate_per_step_records

    records = [
        {
            "posterior": [0.5, 0.5],
            "prior": [0.5, 0.5],
            "efe": [1.0, 1.0],
            "efe_epistemic": [0.0, 0.0],
            "efe_pragmatic": [0.0, 0.0],   # 0 + 0 ≠ 1 → flag
            "q_pi": [0.5, 0.5],
            "p_u": [0.5, 0.5],
        }
    ]
    rep = validate_per_step_records(records)
    assert not rep.ok
    assert any("efe_decomposition" in i for i in rep.issues)


def test_validate_per_step_records_flags_bad_posterior():
    from blockference.io import validate_per_step_records

    records = [
        {
            "posterior": [0.7, 0.7],   # sums to 1.4
            "prior": [0.5, 0.5],
        }
    ]
    rep = validate_per_step_records(records)
    assert not rep.ok
    assert any("posterior_stochastic" in i for i in rep.issues)


# ---------------------------------------------------------------------------
# Composable stage API
# ---------------------------------------------------------------------------


def test_composable_stages_simulate_persist_validate(tmp_path):
    pytest.importorskip("radcad")
    from blockference.io import build_run_paths
    from blockference.pipeline import persist, simulate, validate

    cfg = _smoke_cfg()
    paths = build_run_paths(root=tmp_path, run_name="stages")
    cfg.output.path = str(paths.trajectory_csv)

    df = simulate(cfg)
    summary, per_step, agents = persist(cfg, df, paths)
    schema_rep, model_reports, per_step_rep, artefacts_rep = validate(
        df, agents, per_step, paths
    )

    assert paths.matrices_npz.exists()
    assert paths.matrices_json.exists()
    assert paths.per_step_csv.exists()
    assert paths.policies_json.exists()
    assert paths.validation_report.exists()

    assert summary["n_agents"] == 1
    assert per_step, "per-step records should not be empty"
    assert schema_rep.ok
    assert artefacts_rep.ok
    assert per_step_rep.ok
    for rep in model_reports.values():
        assert rep.ok


# ---------------------------------------------------------------------------
# Full pipeline end-to-end
# ---------------------------------------------------------------------------


def test_run_pipeline_full_artefact_tree(tmp_path):
    pytest.importorskip("radcad")
    pytest.importorskip("PIL")
    from blockference.pipeline import run_pipeline

    result = run_pipeline(_smoke_cfg(), output_root=tmp_path, run_name="full")
    p = result.paths
    # Required artefacts.
    for artefact in (
        p.config_path,
        p.trajectory_csv,
        p.summary_json,
        p.matrices_json,
        p.matrices_npz,
        p.per_step_csv,
        p.policies_json,
        p.run_log,
        p.validation_report,
        p.viz_dir / "trajectory.png",
        p.viz_dir / "action_distribution.png",
        p.viz_dir / "efe_proxy.png",
        p.animations_dir / "trajectory.gif",
    ):
        assert artefact.exists(), f"missing artefact: {artefact}"

    # Every report passes.
    report = json.loads(p.validation_report.read_text())
    assert report["ok"] is True
    assert result.schema_report.ok
    assert result.artefacts_report.ok
    assert result.per_step_report.ok

    # Per-step CSV carries the full diagnostic surface.
    df = pd.read_csv(p.per_step_csv)
    for col in ("posterior", "prior", "efe", "efe_epistemic", "efe_pragmatic",
                "q_pi", "p_u", "posterior_entropy", "efe_argmin"):
        assert col in df.columns


# ---------------------------------------------------------------------------
# Engine selection: cadCAD vs radCAD produce equivalent shapes
# ---------------------------------------------------------------------------


def test_cadcad_engine_runs_end_to_end(tmp_path, capsys):
    pytest.importorskip("cadCAD")
    from blockference.config import ExperimentConfig
    from blockference.simulations import run_experiment

    cfg = ExperimentConfig.from_dict({
        "name": "cadcad_smoke",
        "seed": 0,
        "engine": "cadcad",
        "grid": {"dimension": 3, "planning_length": 1},
        "simulation": {
            "timesteps": 2,
            "n_agents": 1,
            "target": [2, 2],
            "initial_state": [0, 0],
        },
        "output": {"path": None},
    })
    df = run_experiment(cfg)
    assert len(df) > 0
    for col in ("agents", "env_states", "efe", "q_pi", "p_u"):
        assert col in df.columns


def test_radcad_and_cadcad_produce_same_diagnostic_columns():
    pytest.importorskip("radcad")
    pytest.importorskip("cadCAD")
    from blockference.config import ExperimentConfig
    from blockference.simulations import run_experiment

    common = {
        "seed": 0,
        "grid": {"dimension": 3, "planning_length": 1},
        "simulation": {
            "timesteps": 2,
            "n_agents": 1,
            "target": [2, 2],
            "initial_state": [0, 0],
        },
        "output": {"path": None},
    }
    df_rad = run_experiment(ExperimentConfig.from_dict({**common, "engine": "radcad", "name": "rad"}))
    df_cad = run_experiment(ExperimentConfig.from_dict({**common, "engine": "cadcad", "name": "cad"}))
    diagnostic = {"agents", "priors", "env_states", "actions", "inferences",
                  "efe", "efe_epistemic", "efe_pragmatic",
                  "q_pi", "p_u", "obs_idx"}
    assert diagnostic <= set(df_rad.columns)
    assert diagnostic <= set(df_cad.columns)
