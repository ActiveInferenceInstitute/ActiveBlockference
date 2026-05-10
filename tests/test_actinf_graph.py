"""Tests for blockference.gridference.actinf_graph (networkx-based step)."""

import pytest

nx = pytest.importorskip("networkx")

from blockference.gridference import ActiveGridference, actinf_graph, make_grid  # noqa: E402


def test_actinf_graph_emits_one_update_per_node():
    grid = make_grid(3, 2)
    a = ActiveGridference(grid, planning_length=1)
    a.get_C((2, 2))
    a.get_D((0, 0))

    g = nx.Graph()
    g.add_node(
        "a",
        agent=a,
        env_state=a.env_state,
        prior=a.prior,
        prior_A=a.A,
        prior_B=a.B,
        prior_C=a.C,
    )

    out = actinf_graph(g)
    assert "agent_updates" in out
    assert len(out["agent_updates"]) == 1
    upd = out["agent_updates"][0]
    assert upd["source"] == "a"
    assert set(upd) >= {
        "update_prior",
        "update_env",
        "update_action",
        "update_inference",
    }


def test_actinf_graph_handles_two_nodes():
    grid = make_grid(3, 2)
    g = nx.Graph()
    for name, start in (("a", (0, 0)), ("b", (1, 1))):
        ag = ActiveGridference(grid, planning_length=1)
        ag.get_C((2, 2))
        ag.get_D(start)
        g.add_node(
            name,
            agent=ag,
            env_state=ag.env_state,
            prior=ag.prior,
            prior_A=ag.A,
            prior_B=ag.B,
            prior_C=ag.C,
        )
    out = actinf_graph(g)
    sources = {u["source"] for u in out["agent_updates"]}
    assert sources == {"a", "b"}
