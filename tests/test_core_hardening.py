"""Unit and negative-control tests for the strict core contracts."""

from __future__ import annotations

import json
import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from blockference import GridWorld
from blockference.config import ExperimentConfig
from blockference.gridference import ActiveGridference, _move, make_grid
from blockference.io import (
    build_run_paths,
    validate_generative_model,
    validate_per_step_records,
    validate_run_outputs,
    validate_trajectory_dataframe,
)
from blockference.maths import norm_dist, onehot, sample, softmax
from blockference.simulations.grid_sim import run_experiment
from blockference.utils import utils as bu
from blockference.utils.policy import p_actinf_dict


def valid_step() -> dict:
    efe = [1.0, 2.0]
    return {
        "agent_id": 0,
        "env_state": (0, 0),
        "action": 0,
        "obs_idx": 0,
        "posterior": [0.5, 0.5],
        "prior": [0.5, 0.5],
        "efe": efe,
        "efe_epistemic": [0.4, 0.4],
        "efe_pragmatic": [0.6, 1.6],
        "q_pi": [0.7, 0.3],
        "p_u": [0.7, 0.3],
    }


@pytest.mark.parametrize("value", [[-1.0, 2.0], [], [float("nan"), 1.0]])
def test_probability_primitives_reject_invalid_inputs(value):
    with pytest.raises((ValueError, TypeError)):
        norm_dist(value)
    with pytest.raises((ValueError, TypeError)):
        sample(value)


def test_boolean_indices_are_not_accepted_as_integers():
    with pytest.raises(TypeError):
        _move(True, (0, 0), 2)
    with pytest.raises(TypeError):
        onehot(True, 3)
    with pytest.raises(ValueError):
        make_grid(True)


def test_softmax_rejects_nonfinite_values():
    with pytest.raises(ValueError, match="non-finite"):
        softmax([-np.inf, -np.inf])


def test_config_rejects_unknown_keys_and_invalid_coordinates():
    with pytest.raises(ValueError, match="unknown configuration"):
        ExperimentConfig.from_dict({"unexpected": 1})
    with pytest.raises(ValueError, match="unknown grid"):
        ExperimentConfig.from_dict({"grid": {"unexpected": 1}})
    with pytest.raises(ValueError, match="outside"):
        ExperimentConfig.from_dict({"grid": {"dimension": 3}, "simulation": {"target": [3, 0]}})
    with pytest.raises(ValueError, match="unsupported affordances"):
        ExperimentConfig.from_dict({"grid": {"affordances": ["TELEPORT"]}})


def test_config_supports_explicit_multi_agent_initial_states():
    config = ExperimentConfig.from_dict(
        {
            "simulation": {
                "n_agents": 2,
                "initial_state": [0, 0],
                "initial_states": [[0, 0], [2, 2]],
            }
        }
    )
    assert config.simulation.resolved_initial_states == ((0, 0), (2, 2))
    assert ExperimentConfig.from_dict(config.to_dict()).to_dict() == config.to_dict()
    with pytest.raises(ValueError, match="unique"):
        ExperimentConfig.from_dict(
            {"simulation": {"n_agents": 2, "initial_states": [[0, 0], [0, 0]]}}
        )


def test_simulation_does_not_mutate_process_global_rng_state():
    config = ExperimentConfig.from_dict(
        {
            "seed": 11,
            "grid": {"dimension": 3, "planning_length": 1},
            "simulation": {"timesteps": 1, "n_agents": 1, "target": [2, 2]},
            "output": {"path": None},
        }
    )
    np.random.seed(123)
    random.seed(123)
    before_numpy = np.random.get_state()
    before_python = random.getstate()
    run_experiment(config)
    after_numpy = np.random.get_state()
    assert before_numpy[0] == after_numpy[0]
    assert np.array_equal(before_numpy[1], after_numpy[1])
    assert before_numpy[2:] == after_numpy[2:]
    assert before_python == random.getstate()


def test_multi_agent_collision_prior_tracks_realized_position(monkeypatch):
    grid = make_grid(3)
    agents = {}
    for agent_id, position in ((0, (1, 0)), (1, (1, 2))):
        agent = ActiveGridference(grid, planning_length=1, env_state=position)
        agent.get_C((2, 2))
        agent.get_D(position)
        agents[agent_id] = agent
    actions = iter((3, 2))
    monkeypatch.setattr("blockference.gridference.bu.sample", lambda _, rng=None: next(actions))

    result = p_actinf_dict({}, 0, [], {"agents": agents}, grid)
    updates = {update["source"]: update for update in result["agent_updates"]}
    assert updates[0]["update_env"] == (1, 1)
    assert updates[1]["update_env"] == (1, 2)
    assert np.array_equal(updates[0]["update_prior"], np.eye(9)[4])
    assert np.array_equal(updates[1]["update_prior"], np.eye(9)[5])


def test_gridworld_resolves_collisions_swaps_and_boundaries():
    same_target = GridWorld(3, {"a": (1, 0), "b": (1, 2)})
    assert same_target.step({"a": 3, "b": 2}) == {"a": (1, 1), "b": (1, 2)}
    occupied = GridWorld(3, {"a": (1, 0), "b": (1, 1)})
    assert occupied.step({"a": 3, "b": 4}) == {"a": (1, 0), "b": (1, 1)}
    swapped = GridWorld(3, {"a": (1, 0), "b": (1, 1)})
    assert swapped.step({"a": 3, "b": 2}) == {"a": (1, 0), "b": (1, 1)}
    boundary = GridWorld(3, {0: (0, 0)})
    assert boundary.step({0: 0}) == {0: (0, 0)}
    with pytest.raises(ValueError, match="unique"):
        GridWorld(3, {"a": (0, 0), "b": (0, 0)})


def test_math_indices_reject_booleans_and_fractional_values():
    identity = np.eye(2)
    B = np.stack([identity], axis=-1)
    with pytest.raises(TypeError):
        bu.infer_states(True, identity, [0.5, 0.5])
    with pytest.raises(TypeError):
        bu.get_expected_states(B, [1.0, 0.0], True)
    with pytest.raises(TypeError):
        bu.compute_prob_actions(["STAY"], [np.array([[0.5]])], [1.0])


def test_affordances_drive_transition_tensor():
    agent = ActiveGridference(make_grid(3), affordances=["RIGHT", "STAY"])
    assert agent.B.shape == (9, 9, 2)
    assert agent.get_E(["LEFT", "STAY"]) == ["LEFT", "STAY"]
    assert agent.B.shape[-1] == 2


def test_validators_fail_closed_for_missing_and_negative_values():
    assert not validate_per_step_records([{"posterior": [0.5, 0.5]}]).ok
    frame = pd.DataFrame({"env_states": []})
    assert not validate_trajectory_dataframe(frame).ok
    model = SimpleNamespace(
        n_states=2,
        n_observations=2,
        E=["STAY"],
        A=np.eye(2),
        B=np.array([[[1.0], [-1.0]], [[0.0], [2.0]]]),
        C=np.array([0.5, 0.5]),
        D=np.array([0.5, 0.5]),
    )
    assert not validate_generative_model(model).ok


def test_valid_per_step_record_passes():
    assert validate_per_step_records([valid_step()]).ok


def test_per_step_validation_can_enforce_model_dimensions():
    report = validate_per_step_records([valid_step()], n_states=3, n_actions=2)
    assert not report.ok
    assert not report.checks["model_dimensions"]


def test_trajectory_validation_rejects_inconsistent_agent_registries():
    row = {
        column: {}
        for column in (
            "agents",
            "priors",
            "env_states",
            "actions",
            "inferences",
            "efe",
            "efe_epistemic",
            "efe_pragmatic",
            "q_pi",
            "p_u",
            "obs_idx",
        )
    }
    row["env_states"] = {"0": (0, 0)}
    row["actions"] = {"1": 0}
    row.update({"timestep": 0, "substep": 0, "run": 1})
    report = validate_trajectory_dataframe(pd.DataFrame([row]))
    assert not report.ok
    assert not report.checks["agent_keys_consistent"]


def test_persisted_model_validation_fails_closed_on_malformed_payload(tmp_path):
    paths = build_run_paths(tmp_path, "malformed").ensure()
    paths.matrices_json.write_text(json.dumps({"0": []}), encoding="utf-8")
    report = validate_run_outputs(paths)
    assert not report.ok
    assert any("model.0.payload" in issue for issue in report.issues)


def test_run_name_stays_inside_output_root(tmp_path):
    with pytest.raises(ValueError):
        build_run_paths(tmp_path, "../escape")
