"""Tests for blockference.viz (plots + animation)."""

import numpy as np
import pandas as pd
import pytest


def _toy_trajectory_df(n_steps=4, n_agents=2):
    """Build a tiny trajectory frame compatible with the viz helpers."""
    env_states, actions, inferences, agents, priors = [], [], [], [], []
    for t in range(n_steps + 1):
        env_states.append({i: (t % 3, (t + i) % 3) for i in range(n_agents)})
        actions.append({i: (i + t) % 5 if t > 0 else "" for i in range(n_agents)})
        belief = np.zeros(9)
        belief[t % 9] = 1.0
        inferences.append({i: belief.copy() if t > 0 else "" for i in range(n_agents)})
        agents.append({i: object() for i in range(n_agents)})
        priors.append({i: belief for i in range(n_agents)})
    return pd.DataFrame(
        {
            "agents": agents,
            "priors": priors,
            "env_states": env_states,
            "actions": actions,
            "inferences": inferences,
            "timestep": list(range(n_steps + 1)),
            "substep": [0] + [1] * n_steps,
            "run": [1] * (n_steps + 1),
        }
    )


def test_plot_trajectory_writes_png(tmp_path):
    from blockference.viz import plot_trajectory

    out = tmp_path / "t.png"
    p = plot_trajectory(_toy_trajectory_df(), out)
    assert p.exists()
    assert p.stat().st_size > 100


def test_plot_action_distribution_writes_png(tmp_path):
    from blockference.viz import plot_action_distribution

    out = tmp_path / "a.png"
    p = plot_action_distribution(_toy_trajectory_df(), out)
    assert p.exists()


def test_plot_belief_heatmap_writes_png(tmp_path):
    from blockference.viz import plot_belief_heatmap

    out = tmp_path / "b.png"
    p = plot_belief_heatmap(_toy_trajectory_df(), out, agent_id=0, timestep=-1)
    assert p.exists()


def test_plot_efe_proxy_writes_png(tmp_path):
    from blockference.viz import plot_efe_proxy

    out = tmp_path / "e.png"
    p = plot_efe_proxy(_toy_trajectory_df(), out)
    assert p.exists()


def test_animate_trajectory_writes_gif(tmp_path):
    pytest.importorskip("PIL")
    from blockference.viz import animate_trajectory

    out = tmp_path / "anim.gif"
    p = animate_trajectory(_toy_trajectory_df(), out, fps=2)
    assert p.exists()
    # Pillow GIFs always exceed 100 bytes for non-trivial content
    assert p.stat().st_size > 500


def test_animate_trajectory_rejects_unknown_extension(tmp_path):
    from blockference.viz import animate_trajectory

    with pytest.raises(ValueError, match="extension"):
        animate_trajectory(_toy_trajectory_df(), tmp_path / "x.webp")
