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

from blockference.gridference import _move
from blockference.maths import construct_policies
from blockference.utils import utils as u

__all__ = ["p_actinf_single", "p_actinf_dict"]


def _step_agent(agent, prior, A, B, C, env_state, grid):
    """Run a single Active Inference inference + planning step.

    Returns a dict carrying the full diagnostic set described in the
    module docstring (sans ``source``).
    """
    policies = construct_policies(
        [agent.n_states], [len(agent.E)], policy_len=agent.policy_len
    )
    obs_idx = grid.index(env_state)

    qs_current = u.infer_states(obs_idx, A, prior)
    G, parts = u.calculate_G_policies_traced(A, B, C, qs_current, policies=policies)
    Q_pi = u.softmax(-G)
    P_u = u.compute_prob_actions(agent.E, policies, Q_pi)
    chosen_action = u.sample(P_u)

    next_prior = B[:, :, chosen_action].dot(qs_current)
    next_env = _move(chosen_action, env_state, agent.border)

    return {
        "update_prior": next_prior,
        "update_env": next_env,
        "update_action": int(chosen_action),
        "update_inference": qs_current,
        "update_efe": G,
        "update_efe_epistemic": parts["epistemic"],
        "update_efe_pragmatic": parts["pragmatic"],
        "update_q_pi": Q_pi,
        "update_p_u": P_u,
        "update_obs_idx": int(obs_idx),
    }


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
    return {"agent_updates": agent_updates}
