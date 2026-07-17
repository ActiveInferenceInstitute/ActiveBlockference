# Multi-agent collision resolution

```python
from blockference import GridWorld

world = GridWorld(3, {"a": (1, 0), "b": (1, 2)})
print(world.step({"a": 3, "b": 2}))
```

Both agents propose the centre cell. Identifier order grants it to `a`; `b`
stays in place. A target occupied at the beginning of a step is unavailable,
which makes swaps deterministic and safe.
