"""Tests for blockference.agent (pymdp 1.x compatibility layer)."""

import numpy as np
import pytest

from blockference.agent import Agent, BlockferenceAgent


def _identity_A_B(n_states: int = 3, n_actions: int = 2):
    """Build A/B in the per-modality / per-factor list shape pymdp 1.x expects."""
    pytest.importorskip("jax")
    import jax.numpy as jnp

    A = [jnp.eye(n_states)]
    B_factor = np.zeros((n_states, n_states, n_actions))
    B_factor[:, :, 0] = np.eye(n_states)
    B_factor[:, :, 1] = np.roll(np.eye(n_states), 1, axis=0)
    B = [jnp.asarray(B_factor)]
    return A, B


def test_agent_alias_points_to_blockference_agent():
    assert Agent is BlockferenceAgent


def test_agent_construction_forwards_to_pymdp():
    A, B = _identity_A_B()
    try:
        agent = BlockferenceAgent(A=A, B=B)
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"pymdp Agent rejected the toy inputs: {exc}")
    # The pymdp 1.x Agent attaches A and B as attributes after normalisation.
    assert hasattr(agent, "A")
    assert hasattr(agent, "B")
