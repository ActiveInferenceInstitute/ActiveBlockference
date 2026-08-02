# Multi-agent collision resolution

This tutorial demonstrates deterministic simultaneous movement in `GridWorld`.
It assumes the package is installed from the repository root with
`uv sync --locked --extra dev`.

```python
from blockference import GridWorld

world = GridWorld(3, {"a": (1, 0), "b": (1, 2)})
next_positions = world.step({"a": 3, "b": 2})
print(next_positions)
assert next_positions == {"a": (1, 1), "b": (1, 2)}
```

With the default affordance order, action `3` is `RIGHT` and action `2` is
`LEFT`; both agents propose the centre cell. Identifier order grants it to
`a`, so `b` stays in place. A target occupied at the beginning of a step is
unavailable, which makes swaps deterministic and safe.

The environment uses `(y, x)` coordinates and validates all positions and
actions. For configured multi-agent simulation runs, provide distinct starts
through `simulation.initial_states`; see [`pipeline.md`](../pipeline.md).

What to try next: change the two actions to test a boundary or a swap, then
inspect the environment contract in [`architecture.md`](../architecture.md).
