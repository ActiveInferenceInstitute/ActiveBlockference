# Single-agent model

This tutorial builds one validated generative model and inspects its transition
semantics. It assumes the package is installed from the repository root:

```bash
uv sync --locked --extra dev
```

```python
from blockference import ActiveGridference, make_grid

agent = ActiveGridference(
    make_grid(3),
    planning_length=2,
    env_state=(0, 0),
    affordances=["UP", "RIGHT", "STAY"],
)
agent.get_C((2, 2))
agent.get_D((0, 0))

print(agent.B.shape)  # (9, 9, 3)
print(agent.E)       # ['UP', 'RIGHT', 'STAY']
print(agent.D.sum()) # 1.0
```

`B` has one column-stochastic transition matrix per configured affordance.
`A` is the identity likelihood for the fully observed grid, while `C` and `D`
are one-hot vectors after `get_C` and `get_D`.

What to try next: compare the single-agent inference loop in
[`theory.md`](../theory.md) with the complete persisted run in
[`03_cadcad_pipeline.md`](03_cadcad_pipeline.md).
