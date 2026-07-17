# Single-agent model

```python
from blockference import ActiveGridference, make_grid

agent = ActiveGridference(make_grid(3), planning_length=2,
                          affordances=["UP", "RIGHT", "STAY"])
agent.get_C((2, 2))
agent.get_D((0, 0))
print(agent.B.shape, agent.E)
```

`B` has one column-stochastic transition matrix per configured affordance.
