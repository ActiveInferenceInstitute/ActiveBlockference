# `blockference/` — installable library

This is the importable Python package. Everything outside this directory
(notebooks, GRTs, docs, tests) consumes from here, never the other way
around.

## Module map

| Module                         | Responsibility                                       |
|--------------------------------|------------------------------------------------------|
| `blockference.agent`           | Thin extension hook over `pymdp.agent.Agent` (v1.x). |
| `blockference.maths`           | Self-contained AIF primitives (no pymdp dep).        |
| `blockference.gridference`     | `ActiveGridference` generative model + step funcs.   |
| `blockference.config`          | Typed YAML/TOML experiment config.                   |
| `blockference.envs`            | Environments (single-agent grid, two-agent grid).    |
| `blockference.simulations`     | End-to-end cadCAD/radCAD experiment scripts.         |
| `blockference.io`              | Output layout, persistence, validation.              |
| `blockference.viz`             | Static plots + animated GIFs.                        |
| `blockference.pipeline`        | Composable run / persist / viz / validate pipeline.  |
| `blockference.logging_setup`   | stdlib logging + per-run file handler.               |
| `blockference.utils`           | High-level AIF math + policy templates + mutual-info.|

## Public surface

The package re-exports the most-used symbols at the top level so callers
don't need to memorise sub-paths:

```python
from blockference import (
    ActiveGridference,
    BlockferenceAgent,
    actinf_graph,
    actinf_planning_single,
    make_grid,
)
```

## The canonical step

The single-agent step — implemented by `actinf_planning_single` — is:

```
qs_current = infer_states(obs_idx, A, prior)
G          = calculate_G_policies(A, B, C, qs_current, policies)
Q_pi       = softmax(-G)
P_u        = compute_prob_actions(E, policies, Q_pi)
action     = sample(P_u)
next_prior = B[:,:,action] @ qs_current
next_env   = move(action, env_state, border)
```

`actinf_graph` is the same loop but applied per node of a `networkx`
graph; the multi-agent dict variant lives in
`blockference.utils.policy.p_actinf_dict`.

## Guarantees

* Every public function has a NumPy-style docstring.
* Every public class is exported from a parent `__init__.py`.
* Every public function/class has at least one regression test in `tests/`.

## Testing this layer

```bash
pytest tests/ -k blockference
```

See [`AGENTS.md`](AGENTS.md) for editing rules.
