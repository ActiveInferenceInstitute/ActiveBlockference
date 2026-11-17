"""Lossless supported identities and deliberately distinct world occupancy rules."""

from __future__ import annotations

import copy
import json

import networkx as nx
import numpy as np
import pytest

from blockference.envs import GridWorld, resolve_moves
from blockference.gridference import ActiveGridference, actinf_graph, make_grid
from blockference.utils.policy import p_actinf_dict


def test_json_round_trip_preserves_mixed_identity_types_registry_and_sequence_actions():
    positions = {2: (0, 0), 10: (2, 2), 1: (0, 2), "1": (2, 0)}
    world = GridWorld(3, positions, stochastic=True)
    restored = GridWorld.load(json.loads(json.dumps(world.serialize())))
    assert list(restored.current_state.items()) == list(positions.items())
    assert [type(key) for key in restored.current_state] == [int, int, int, str]
    assert restored.affordances == world.affordances
    assert restored.stochastic is True
    expected = {2: (0, 1), 10: (2, 1), 1: (1, 2), "1": (1, 0)}
    assert world.step([3, 2, 1, 0]) == expected
    assert restored.step([3, 2, 1, 0]) == expected


@pytest.mark.parametrize(("left", "right", "winner"), [(1, "1", 1), (2, 10, 10)])
@pytest.mark.parametrize("reverse", [False, True])
def test_json_round_trip_retains_existing_type_repr_collision_priority(left, right, winner, reverse):
    entries = [(left, (1, 0)), (right, (1, 2))]
    positions = dict(reversed(entries) if reverse else entries)
    world = GridWorld(3, positions)
    restored = GridWorld.load(json.loads(json.dumps(world.serialize())))
    actions = {left: 3, right: 2}
    expected = dict(positions)
    expected[winner] = (1, 1)
    assert world.step(actions) == expected
    assert restored.step(actions) == expected
    assert list(restored.current_state) == list(positions)


@pytest.mark.parametrize("identity", [True, 1.0, np.int64(1), b"1", (1,), frozenset({1}), object()])
def test_unsupported_hashables_cannot_enter_construction_direct_moves_or_serialization(identity):
    with pytest.raises(TypeError):
        GridWorld(3, {identity: (0, 0)})
    with pytest.raises(TypeError):
        resolve_moves({identity: (0, 0)}, {identity: 4}, 3)
    with pytest.raises(TypeError):
        resolve_moves({1: (0, 0)}, {identity: 4}, 3)
    world = GridWorld(3)
    world.positions[identity] = (0, 0)
    with pytest.raises(TypeError):
        world.serialize()


@pytest.mark.parametrize("identity", [
    {"type": "int", "value": True}, {"type": "int", "value": 1.0},
    {"type": "int", "value": "1"}, {"type": "str", "value": 1},
    {"type": "repr", "value": "1"}, {"value": 1}, None,
])
def test_malformed_typed_identity_records_are_not_coerced(identity):
    payload = GridWorld(3, {1: (0, 0)}).serialize()
    payload["positions"][0]["agent_id"] = identity
    with pytest.raises((TypeError, ValueError)):
        GridWorld.load(payload)


def test_duplicate_records_and_lossy_legacy_payloads_are_rejected():
    payload = GridWorld(3, {1: (0, 0)}).serialize()
    duplicate = copy.deepcopy(payload["positions"][0])
    duplicate["position"] = [1, 1]
    payload["positions"].append(duplicate)
    with pytest.raises(ValueError, match="duplicate"):
        GridWorld.load(payload)
    with pytest.raises(ValueError):
        GridWorld.load({"dimension": 3, "positions": {"1": [0, 0]}})
    empty = GridWorld.load(json.loads(json.dumps(GridWorld(3).serialize())))
    assert empty.current_state == {}
    assert empty.step([]) == {}


def test_graph_worlds_and_shared_occupancy_keep_distinct_realized_priors(monkeypatch):
    grid = make_grid(3)
    agents = {}
    graph = nx.Graph()
    for agent_id, position in ((0, (1, 0)), (1, (1, 2))):
        agent = ActiveGridference(grid, planning_length=1)
        agent.get_C((2, 2))
        agent.get_D(position)
        agents[agent_id] = agent
        graph.add_node(
            agent_id, agent=agent, prior=agent.D, prior_A=agent.A,
            prior_B=agent.B, prior_C=agent.C, env_state=position,
        )
    actions = iter([3, 2, 3, 2])
    monkeypatch.setattr("blockference.gridference.bu.sample", lambda _p, rng=None: next(actions))
    independent = {
        update["source"]: update for update in actinf_graph(graph)["agent_updates"]
    }
    shared = {
        update["source"]: update
        for update in p_actinf_dict({}, 0, [], {"agents": agents}, grid)["agent_updates"]
    }
    assert [independent[agent]["update_env"] for agent in (0, 1)] == [(1, 1), (1, 1)]
    assert [shared[agent]["update_env"] for agent in (0, 1)] == [(1, 1), (1, 2)]
    assert np.argmax(independent[1]["update_prior"]) == grid.index((1, 1))
    assert np.argmax(shared[1]["update_prior"]) == grid.index((1, 2))
