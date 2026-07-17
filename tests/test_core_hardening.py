"""Unit and negative-control tests for the strict core contracts."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from blockference import GridWorld
from blockference.config import ExperimentConfig
from blockference.gridference import ActiveGridference, make_grid
from blockference.io import (
    build_run_paths,
    validate_generative_model,
    validate_per_step_records,
    validate_trajectory_dataframe,
)
from blockference.maths import norm_dist, sample, softmax


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


def test_gridworld_resolves_collisions_swaps_and_boundaries():
    same_target = GridWorld(3, {"a": (1, 0), "b": (1, 2)})
    assert same_target.step({"a": 3, "b": 2}) == {"a": (1, 1), "b": (1, 2)}
    occupied = GridWorld(3, {"a": (1, 0), "b": (1, 1)})
    assert occupied.step({"a": 3, "b": 4}) == {"a": (1, 0), "b": (1, 1)}
    swapped = GridWorld(3, {"a": (1, 0), "b": (1, 1)})
    assert swapped.step({"a": 3, "b": 2}) == {"a": (1, 0), "b": (1, 1)}
    boundary = GridWorld(3, {0: (0, 0)})
    assert boundary.step({0: 0}) == {0: (0, 0)}


def test_affordances_drive_transition_tensor():
    agent = ActiveGridference(make_grid(3), affordances=["RIGHT", "STAY"])
    assert agent.B.shape == (9, 9, 2)
    assert agent.get_E(["LEFT", "STAY"]) == ["LEFT", "STAY"]
    assert agent.B.shape[-1] == 2


def test_validators_fail_closed_for_missing_and_negative_values():
    assert not validate_per_step_records([{ "posterior": [0.5, 0.5] }]).ok
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


def test_run_name_stays_inside_output_root(tmp_path):
    with pytest.raises(ValueError):
        build_run_paths(tmp_path, "../escape")
