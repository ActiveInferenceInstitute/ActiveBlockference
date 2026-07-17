# Active Inference primer

This document explains the discrete-state Active Inference (AIF) loop
that ActiveBlockference implements. It is intentionally short — the
canonical references are listed at the end.

## The free-energy principle in one paragraph

A self-organising system that persists in a non-equilibrium steady state
must minimise the *surprise* of its sensory observations. Surprise is
intractable, so the system minimises a tractable upper bound: the
**variational free energy** `F`. **Active Inference** adds the imperative
that the system can also act, and so it additionally minimises the
**expected free energy** `G(π)` of future trajectories under each
candidate policy π.

## The five matrices

ActiveBlockference works with a discrete POMDP factored as:

| Matrix | Math                                                | Meaning                                                                            |
|--------|-----------------------------------------------------|------------------------------------------------------------------------------------|
| **A**  | `P(o ∣ s)`                                          | Likelihood — probability of an observation given a hidden state.                   |
| **B**  | `P(s_t ∣ s_{t-1}, u_{t-1})`                         | Controllable transitions — how actions move hidden state.                          |
| **C**  | `log P̃(o)` (preferences over observations)          | Prior preference — what the agent *wants* to observe.                              |
| **D**  | `P(s_0)`                                            | Prior over the initial hidden state.                                               |
| **E**  | action labels                                        | Affordances — the action set; in `ActiveGridference`: `UP, DOWN, LEFT, RIGHT, STAY`. |

In `ActiveGridference` for an `n×n` grid:
* `A.shape == (n_obs, n_states) == (n², n²)` (identity = fully observed).
* `B.shape == (n_states, n_states, len(E)) == (n², n², 5)`.
* `C.shape == (n_obs,) == (n²,)`.
* `D.shape == (n_states,) == (n²,)`.

## The inference + planning loop

For each timestep `t`:

1. **Observe** `o_t`, get its index in `grid`.
2. **Infer** posterior over hidden states:
   `q(s_t) = softmax(log A[o_t, :] + log prior_t)`.
3. **Score policies** by expected free energy
   `G(π) = Σ_t  q(s_t∣π) · H[A]  +  KL(q(o_t∣π) ∥ C)`,
   where the first term is *epistemic* (information gain about states)
   and the second is *pragmatic* (preferences over outcomes).
4. **Select policy** with `Q(π) = softmax(-G(π))`.
5. **Marginalise** the action: `P(u) = Σ_π π[0] · Q(π)`; **sample** an
   action `u_t`.
6. **Propagate** the prior: `prior_{t+1} = B[:, :, u_t] · q(s_t)`.

The corresponding code paths:

* Steps 2–6 single-agent: `actinf_planning_single` in
  `blockference/gridference.py`.
* Steps 2–6 multi-agent (graph): `actinf_graph` in the same file.
* Steps 2–6 multi-agent (dict): `p_actinf_dict` in
  `blockference/utils/policy.py`.

## Why expected free energy decomposes into epistemic + pragmatic

`G(π)` admits the rewriting:

```
G(π) = E_q[ log q(s_t∣π) − log p(o_t, s_t∣π) ]
     ≈  H[ A · q(s_t∣π) ]                       # ambiguity
       + KL( A · q(s_t∣π) ∥ C )                  # risk / preference
```

Minimising `G` therefore *simultaneously* explores (reduces ambiguity)
and exploits (matches preferred outcomes). That's why a single objective
can resolve the explore/exploit dilemma.

## Mapping to cadCAD

In cadCAD vocabulary, an Active Inference loop becomes a **partial state
update block**:

* `policies = { "p_actinf": <policy fn> }` — the AIF inference function.
* `variables = { "agents": ..., "priors": ..., "env_states": ..., "actions": ..., "inferences": ... }`
   — five state variables updated by the policy's emitted updates.

See `blockference/simulations/grid_sim.py` for a complete worked example.

## Implementation note: self-contained math layer

The numerical primitives implementing the loop above (``softmax``,
``log_stable``, ``infer_states``, ``calculate_G_policies``,
``compute_prob_actions``, ``sample``, ``onehot``, ``construct_policies``)
all live in :mod:`blockference.maths` and :mod:`blockference.utils.utils`.
They are pure NumPy and have **no pymdp dependency**, so the grid
pipeline keeps producing reproducible NumPy outputs even as upstream
pymdp remains an optional upstream adapter; the grid pipeline uses the package's
validated NumPy primitives directly.
for the complete pymdp 0.0.x → 1.0.x mapping.

## References

* Friston, K., Daunizeau, J., & Kiebel, S. (2009). *Reinforcement
  learning or active inference?*
* Friston, K., et al. (2017). *Active inference: a process theory*.
* Sajid, N., Ball, P. J., Parr, T., & Friston, K. J. (2021). *Active
  Inference: Demystified and Compared*.
* Heins, C., Tschantz, A., et al. (2022). *pymdp: A Python library for
  active inference in discrete state spaces*.
* Active Inference Institute → <https://activeinference.org/>
