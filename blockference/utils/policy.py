"""cadCAD/radCAD policy functions for Active Inference simulations.

Each function takes the canonical cadCAD signature ``(params, substep,
state_history, previous_state, …)`` and returns a dict of state-update
instructions consumed by the corresponding state-update functions.

The dict returned per agent carries the *full* per-step diagnostics that
the pipeline can persist downstream:

``update_prior``      next prior over hidden states (``B[:,:,a] · qs``)
``update_env``        next ``(y, x)`` environment coordinate
``update_action``     sampled action index ``a``
``update_inference``  posterior over hidden states ``q(s_t)``
``update_efe``        per-policy expected free energy ``G(π)``
``update_efe_epistemic``  per-policy ambiguity term
``update_efe_pragmatic``  per-policy risk (KL-to-C) term
``update_q_pi``       policy posterior ``softmax(-G)``
``update_p_u``        marginal action posterior ``P(u)``
``update_obs_idx``    flattened observation index used by the inference step
"""

from blockference.envs import resolve_moves
from blockference.gridference import _step_agent

__all__ = ["p_actinf_single", "p_actinf_dict"]


def p_actinf_single(params, substep, state_history, previous_state, act, grid):
    """Single-agent Active Inference cadCAD policy.

    Expects ``previous_state`` to contain ``env_state``, ``prior``,
    ``prior_A``, ``prior_B``, ``prior_C``.
    """
    return _step_agent(
        agent=act,
        prior=previous_state["prior"],
        A=previous_state["prior_A"],
        B=previous_state["prior_B"],
        C=previous_state["prior_C"],
        env_state=previous_state["env_state"],
        grid=grid,
    )


def p_actinf_dict(params, substep, state_history, previous_state, grid):
    """Multi-agent Active Inference cadCAD policy (dict-based agent registry).

    ``previous_state['agents']`` must be a ``dict[source_id -> agent]``.
    """
    agents = previous_state["agents"]
    agent_updates = []
    for source, agent in agents.items():
        upd = _step_agent(
            agent=agent,
            prior=agent.prior,
            A=agent.A,
            B=agent.B,
            C=agent.C,
            env_state=agent.env_state,
            grid=grid,
        )
        upd["source"] = source
        agent_updates.append(upd)
    positions = {source: agent.env_state for source, agent in agents.items()}
    actions = {upd["source"]: upd["update_action"] for upd in agent_updates}
    resolved = resolve_moves(
        positions,
        actions,
        dimension=next(iter(agents.values())).border + 1,
        affordances=next(iter(agents.values())).E,
    )
    for upd in agent_updates:
        upd["update_env"] = resolved[upd["source"]]
    return {"agent_updates": agent_updates}
