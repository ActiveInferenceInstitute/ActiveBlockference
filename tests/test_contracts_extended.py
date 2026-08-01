"""Expanded local tests for numerical, API, IO, and rendering contracts."""

from __future__ import annotations

import logging
import runpy
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import yaml

from blockference.actions import validate_affordances
from blockference.config import ExperimentConfig, GridConfig, load_experiment_config
from blockference.envs import GridWorld, resolve_moves
from blockference.gridference import ActiveGridference, _move, actinf_graph, make_grid
from blockference.io import (
    build_run_paths,
    persist_config,
    persist_dataframe,
    persist_generative_model,
    persist_generative_model_npz,
    persist_manifest,
    persist_per_step_records,
    persist_policies,
    persist_summary,
    validate_run_outputs,
)
from blockference.io.validate import (
    validate_generative_model,
    validate_per_step_records,
    validate_trajectory_dataframe,
)
from blockference.logging_setup import configure_run_logging, get_logger, remove_handler
from blockference.maths import construct_policies, norm_dist, sample, softmax
from blockference.pipeline import build_per_step_records, summarise_trajectory
from blockference.utils import utils as bu
from blockference.utils.mutual_info import _cli, calculate_mi
from blockference.viz import (
    animate_trajectory,
    plot_action_distribution,
    plot_belief_heatmap,
    plot_efe,
    plot_trajectory,
)
from blockference.viz._common import (
    coerce_cell,
    extract_action_history,
    extract_agent_positions,
    extract_belief_history,
    extract_efe_history,
    infer_grid_dimension,
)


def _frame() -> pd.DataFrame:
    vector = {0: [1.0, 0.0, 0.0, 0.0]}
    return pd.DataFrame(
        {
            "agents": [{0: object()}],
            "priors": [vector],
            "env_states": [{0: (0, 0)}],
            "actions": [{0: 3}],
            "inferences": [vector],
            "efe": [{0: [1.0, 2.0]}],
            "efe_epistemic": [{0: [0.4, 0.5]}],
            "efe_pragmatic": [{0: [0.6, 1.5]}],
            "q_pi": [{0: [0.7, 0.3]}],
            "p_u": [{0: [0.1, 0.2, 0.3, 0.4]}],
            "obs_idx": [{0: 0}],
            "timestep": [0],
            "substep": [0],
            "run": [1],
        }
    )


def test_action_and_math_edges_are_explicit():
    assert validate_affordances(None)
    with pytest.raises(ValueError):
        validate_affordances([])
    with pytest.raises(ValueError):
        validate_affordances([""])
    with pytest.raises(ValueError):
        validate_affordances(["UP", "UP"])
    with pytest.raises(ValueError):
        validate_affordances(["NOPE"])

    assert np.allclose(softmax(np.array(3.0)), 1.0)
    assert np.allclose(norm_dist(np.zeros((2, 2))), 0.5)
    rng = np.random.default_rng(2)
    assert sample([1.0, 0.0], rng=rng) == 0
    with pytest.raises(ValueError):
        norm_dist(np.zeros((0, 2)))
    with pytest.raises(ValueError):
        softmax([1.0, 2.0], axis=2)

    policies = construct_policies([2], [2], policy_len=2, control_fac_idx=[0])
    assert len(policies) == 4
    assert construct_policies([2, 2], [2, 2], policy_len=2, control_fac_idx=[0])
    with pytest.raises(ValueError, match="max_policies"):
        construct_policies([2], [2], policy_len=3, max_policies=4)
    with pytest.raises(ValueError):
        construct_policies([2], [2], policy_len=1, control_fac_idx=[True])


def test_hand_calculated_inference_transition_efe_and_marginalisation():
    A = np.eye(2)
    B = np.zeros((2, 2, 2))
    B[:, :, 0] = np.eye(2)
    B[:, :, 1] = np.array([[0.0, 1.0], [1.0, 0.0]])
    C = np.array([1.0, 0.0])
    qs = bu.infer_states(0, A, [0.5, 0.5])
    assert np.allclose(qs, [1.0, 0.0])
    assert np.allclose(bu.get_expected_states(B, qs, 1), [0.0, 1.0])
    assert np.allclose(bu.get_expected_observations(A, [0.0, 1.0]), [0.0, 1.0])
    G, parts = bu.calculate_G_policies_traced(
        A, B, C, qs, [np.array([[0]]), np.array([[1]])]
    )
    assert np.allclose(G, parts["epistemic"] + parts["pragmatic"])
    assert np.allclose(bu.compute_prob_actions(["stay", "switch"], [np.array([[0]]), np.array([[1]])], [0.25, 0.75]), [0.25, 0.75])
    with pytest.raises(ValueError):
        bu.calculate_G(A, B, C, qs, ["only-one"])
    with pytest.raises(ValueError):
        bu.calculate_G_policies(A, B, C, qs, [])
    with pytest.raises(ValueError):
        bu.get_expected_observations(np.ones((2, 2)), [0.2, 0.2])
    with pytest.raises(ValueError):
        bu.kl_divergence([0.5, 0.5], [1.0, 0.0, 0.0])
    for call in (
        lambda: bu.plot_likelihood([[1.0, 0.0]]),
        lambda: bu.plot_likelihood([[0.5, 0.5], [0.4, 0.5]]),
        lambda: bu.plot_grid([]),
        lambda: bu.plot_grid([(0, 0)], num_x=0),
        lambda: bu.plot_point_on_grid([1.0, 0.0], [(0, 0)]),
        lambda: bu.plot_beliefs([0.25, 0.25]),
    ):
        with pytest.raises(ValueError):
            call()


def test_config_loaders_and_round_trip(tmp_path):
    config = ExperimentConfig.from_dict(
        {
            "name": "toml",
            "seed": 4,
            "engine": "cadcad",
            "grid": {"dimension": 2, "planning_length": 1, "max_policies": 8},
            "simulation": {"target": "random", "initial_state": [0, 0]},
            "output": {"path": None},
        }
    )
    assert config.grid.max_policies == 8
    assert config.simulation.target == "random"
    path = tmp_path / "config.toml"
    path.write_text(
        '[grid]\ndimension = 2\nplanning_length = 1\n\n[simulation]\ntimesteps = 1\n',
        encoding="utf-8",
    )
    assert load_experiment_config(path).grid.dimension == 2
    with pytest.raises(FileNotFoundError):
        load_experiment_config(tmp_path / "missing.yml")
    bad = tmp_path / "config.txt"
    bad.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="extension"):
        load_experiment_config(bad)
    with pytest.raises(TypeError):
        GridConfig(dimension=True)
    for data in (
        {"grid": {"dimension": 0}},
        {"grid": {"dimension": 1}},
        {"grid": {"dimension": 3, "planning_length": 1, "max_policies": 1.5}},
        {"simulation": {"timesteps": True}},
        {"simulation": {"initial_states": "bad"}},
        {"simulation": {"n_agents": 2, "initial_states": [[0, 0]]}},
        {"output": {"path": ""}},
        {"engine": "unknown"},
        {"name": "../unsafe"},
    ):
        with pytest.raises((TypeError, ValueError)):
            ExperimentConfig.from_dict(data)
    malformed = tmp_path / "malformed.yml"
    malformed.write_text("[not yaml", encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        load_experiment_config(malformed)


def test_grid_agent_graph_and_environment_contracts():
    grid = make_grid(2)
    assert len(make_grid(2, 3)) == 8
    assert _move(0, (1, 1), 1) == (0, 1)
    assert _move(1, (0, 0), 1) == (1, 0)
    assert _move(2, (0, 1), 1) == (0, 0)
    assert _move(3, (0, 0), 1) == (0, 1)
    assert _move(4, (1, 1), 1) == (1, 1)
    with pytest.raises(ValueError):
        ActiveGridference([(0, 0), (1, 0)], planning_length=1)
    agent = ActiveGridference(grid, planning_length=1, max_policies=20)
    agent.get_C((1, 1))
    agent.get_D((0, 0))
    assert agent.get_A().shape == (4, 4)
    assert agent.get_B().shape == (4, 4, 5)
    assert agent.get_E(["RIGHT", "STAY"]) == ["RIGHT", "STAY"]
    with pytest.raises(ValueError):
        agent.get_C((2, 2))

    network = nx.Graph()
    network.add_node("agent", agent=agent, prior=agent.D, prior_A=agent.A, prior_B=agent.B, prior_C=agent.C, env_state=agent.env_state)
    assert len(actinf_graph(network, rng=np.random.default_rng(0))["agent_updates"]) == 1
    world = GridWorld(2, {"a": (0, 0), "b": (1, 1)})
    assert world.step([3, 2]) == {"a": (0, 1), "b": (1, 0)}
    with pytest.raises(ValueError):
        resolve_moves({"a": (0, 0)}, {"b": 0}, 2)
    with pytest.raises(ValueError):
        GridWorld(1)
    with pytest.raises(ValueError):
        GridWorld(2, {0: (2, 0)})
    with pytest.raises(ValueError):
        world.step("bad")
    with pytest.raises((ValueError, TypeError)):
        _move(0, (0.5, 0), 1)
    with pytest.raises(ValueError):
        _move(0, (0, 0), -1)
    with pytest.raises(IndexError):
        _move(99, (0, 0), 1)


def test_persistence_manifest_and_summary_helpers(tmp_path):
    paths = build_run_paths(tmp_path, "artifacts").ensure()
    cfg = ExperimentConfig.from_dict({"output": {"path": None}})
    frame = _frame()
    persist_config(cfg, paths)
    persist_dataframe(frame, paths)
    persist_summary(summarise_trajectory(frame), paths)
    agent = ActiveGridference(make_grid(2), planning_length=1)
    agent.get_C((1, 1))
    agent.get_D((0, 0))
    persist_generative_model({0: agent}, paths)
    persist_generative_model_npz({0: agent}, paths)
    persist_policies([np.array([[0]])], paths)
    persist_per_step_records(build_per_step_records(frame), paths)
    assert persist_manifest(paths).is_file()
    assert paths.manifest_json.read_text(encoding="utf-8").find("sha256") >= 0


def test_validation_negative_controls_cover_shapes_registries_and_manifests(tmp_path):
    assert not validate_generative_model(SimpleNamespace(n_states=0)).ok
    assert not validate_generative_model(SimpleNamespace(n_states=2, n_observations=0)).ok
    assert not validate_generative_model(SimpleNamespace(n_states=2, n_observations=2, E=[])).ok
    bad = SimpleNamespace(
        n_states=2,
        n_observations=2,
        E=["same", "same"],
        A=[[1.0], [0.0]],
        B=np.zeros((2, 2, 2)),
        C=[0.5, 0.5],
        D=[0.5, 0.5],
    )
    assert not validate_generative_model(bad).ok
    malformed = {key: None for key in ("env_state", "posterior", "prior", "efe", "efe_epistemic", "efe_pragmatic", "q_pi", "p_u")}
    malformed.update({"agent_id": 0, "action": True, "obs_idx": True})
    assert not validate_per_step_records([malformed]).ok
    frame = _frame().copy()
    frame.at[0, "env_states"] = {0: (0, 0), 1: (1, 1)}
    assert not validate_trajectory_dataframe(frame).ok
    paths = build_run_paths(tmp_path, "manifest").ensure()
    paths.manifest_json.write_text("{}", encoding="utf-8")
    assert not validate_run_outputs(paths).ok


def test_validation_rejects_bad_vectors_bounds_and_manifest_entries(tmp_path):
    record = {
        "agent_id": 0,
        "env_state": "not-a-coordinate",
        "action": 9,
        "obs_idx": 9,
        "posterior": [2.0, -1.0],
        "prior": [0.5, 0.5],
        "efe": [float("nan"), 1.0],
        "efe_epistemic": [0.1],
        "efe_pragmatic": [0.2, 0.8],
        "q_pi": [0.5, 0.5],
        "p_u": [1.0, 0.0],
    }
    assert not validate_per_step_records([record], n_states=2, n_actions=2).ok
    paths = build_run_paths(tmp_path, "bad-manifest").ensure()
    paths.manifest_json.write_text(
        '{"version": 1, "complete": true, "files": [null, {"path": "../escape"}]}',
        encoding="utf-8",
    )
    assert not validate_run_outputs(paths).ok


def test_viz_common_and_renderers_reject_bad_inputs_and_write_outputs(tmp_path):
    frame = _frame()
    assert coerce_cell("[1, 2]") == [1, 2]
    assert extract_agent_positions(frame)[0] == [(0, 0)]
    assert extract_action_history(frame)[0] == [3]
    assert extract_belief_history(frame)[0][0].shape == (4,)
    assert extract_efe_history(frame)[0][0].shape == (2,)
    assert infer_grid_dimension(frame) == 2
    for renderer, name in (
        (plot_trajectory, "trajectory.png"),
        (plot_action_distribution, "actions.png"),
        (plot_belief_heatmap, "belief.png"),
        (plot_efe, "efe.png"),
    ):
        assert renderer(frame, tmp_path / name).is_file()
    assert animate_trajectory(frame, tmp_path / "trajectory.gif").is_file()
    with pytest.raises(ValueError):
        plot_trajectory(frame, tmp_path / "bad.svg")
    with pytest.raises(ValueError):
        animate_trajectory(frame, tmp_path / "bad.svg")
    with pytest.raises(ValueError):
        infer_grid_dimension(pd.DataFrame({"env_states": [{0: (0, 0)}], "inferences": [{0: [1, 0, 0]}]}))


def test_viz_common_rejects_nonfinite_and_singleton_grids():
    from blockference.io.persist import _to_jsonable

    frame = _frame().copy()
    frame.at[0, "inferences"] = {0: [1.0, float("nan"), 0.0, 0.0]}
    with pytest.raises(ValueError, match="finite vector"):
        extract_belief_history(frame)
    assert _to_jsonable({0: (0, 1)}) == {"0": [0, 1]}
    assert _to_jsonable(np.float64(3.5)) == 3.5
    assert _to_jsonable(Path("abc/def")) == "abc/def"


def test_atomic_replace_publishes_and_leaves_no_temp(tmp_path):
    from blockference.io.persist import atomic_replace

    target = tmp_path / "out.json"
    returned = atomic_replace(target, lambda p: p.write_text("{}", encoding="utf-8"))
    assert returned == target
    assert target.read_text(encoding="utf-8") == "{}"
    leftovers = [p.name for p in tmp_path.iterdir()]
    assert leftovers == ["out.json"]


def test_atomic_replace_cleans_temp_and_keeps_previous_on_failure(tmp_path):
    from blockference.io.persist import atomic_replace

    target = tmp_path / "out.json"
    target.write_text("original", encoding="utf-8")

    def failing_writer(path):
        path.write_text("partial", encoding="utf-8")
        raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        atomic_replace(target, failing_writer)
    # Previous complete file is untouched and the temp file is cleaned up.
    assert target.read_text(encoding="utf-8") == "original"
    assert [p.name for p in tmp_path.iterdir()] == ["out.json"]


def test_atomic_npz_round_trip_is_loadable(tmp_path):
    paths = build_run_paths(tmp_path, "npz").ensure()
    agent = ActiveGridference(make_grid(2), planning_length=1)
    agent.get_C((1, 1))
    agent.get_D((0, 0))
    persist_generative_model_npz({0: agent}, paths)
    with np.load(paths.matrices_npz, allow_pickle=False) as archive:
        assert set(archive.files) == {"agent_0/A", "agent_0/B", "agent_0/C", "agent_0/D", "agent_0/E"}
        assert archive["agent_0/A"].shape == (4, 4)


def test_trajectory_schema_typed_round_trip(tmp_path):
    from blockference.io import build_run_paths, parse_trajectory_records, persist_dataframe

    frame = _frame()
    records = parse_trajectory_records(frame)
    assert len(records) == 1
    rec = records[0]
    assert rec.run == 1 and rec.env_state == (0, 0) and rec.action == 3 and rec.obs_idx == 0
    assert rec.agent_id == "0"
    assert rec.posterior == (1.0, 0.0, 0.0, 0.0)
    assert rec.prior == (1.0, 0.0, 0.0, 0.0)
    # The CSV round-trip preserves the same typed values.
    paths = build_run_paths(tmp_path, "schema").ensure()
    persist_dataframe(frame, paths)
    reloaded = pd.read_csv(paths.trajectory_csv)
    assert parse_trajectory_records(reloaded) == records


def test_plot_helpers_and_mutual_information_edges(tmp_path):
    import matplotlib.pyplot as plt

    A = np.eye(2)
    figures = [
        bu.plot_likelihood(A),
        bu.plot_grid([(0, 0), (0, 1), (1, 0), (1, 1)]),
        bu.plot_point_on_grid([1.0, 0.0, 0.0, 0.0], [(0, 0), (0, 1), (1, 0), (1, 1)]),
        bu.plot_beliefs([0.5, 0.5]),
    ]
    assert len(figures) == 4
    plt.close("all")
    X = pd.DataFrame({"kind": ["a", "b"] * 10, "number": list(range(2)) * 10})
    target = [0, 1] * 10
    assert set(calculate_mi(X, target).index) == {"kind", "number"}
    with pytest.raises(ValueError):
        calculate_mi(X, [0, 1])
    with pytest.raises(ValueError):
        calculate_mi(pd.DataFrame(), [])
    source = tmp_path / "mi.csv"
    source.write_text(
        "feature,target\n" + "\n".join(f"{index % 2},{index % 2}" for index in range(20)) + "\n",
        encoding="utf-8",
    )
    assert _cli([str(source), "--target", "target"]) == 0


def test_logging_and_module_entrypoint(tmp_path):
    logger = get_logger("blockference.test")
    assert isinstance(logger, logging.Logger)
    handler = configure_run_logging(tmp_path / "run.log")
    logger.info("hello")
    remove_handler(handler)
    assert "hello" in (tmp_path / "run.log").read_text(encoding="utf-8")
    with pytest.raises(SystemExit):
        runpy.run_module("blockference.__main__", run_name="__main__")
