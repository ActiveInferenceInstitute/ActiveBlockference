# 01 — A single Active Inference agent on a 3×3 grid

**Prerequisites:** [`docs/theory.md`](../theory.md) (skim is fine).

By the end you will have driven one agent from `(0, 0)` to a preferred
cell using only the public Python surface.

## 1. Build the grid and the agent

```python
from blockference import ActiveGridference, make_grid

grid  = make_grid(grid_len=3, grid_dim=2)        # nine cells: (0,0) … (2,2)
agent = ActiveGridference(grid, planning_length=2, env_state=(0, 0))

agent.get_C(preferred_state=(2, 2))              # I want to be at (2, 2)
agent.get_D(initial_state=(0, 0))                # I start at (0, 0)
```

`agent.A`, `agent.B`, `agent.C`, `agent.D` are now populated. `agent.E`
holds the affordance set: `["UP", "DOWN", "LEFT", "RIGHT", "STAY"]`.

## 2. Take one step

```python
from blockference import actinf_planning_single

update = actinf_planning_single(
    agent,
    env_state=agent.env_state,
    A=agent.A, B=agent.B, C=agent.C,
    prior=agent.prior,
)

print(update["update_action"])   # int in 0..4
print(update["update_env"])      # new (y, x)
```

## 3. Loop until the agent reaches the goal

```python
target = (2, 2)
for t in range(20):
    update = actinf_planning_single(
        agent, agent.env_state, agent.A, agent.B, agent.C, agent.prior
    )
    agent.prior     = update["update_prior"]
    agent.env_state = update["update_env"]
    if agent.env_state == target:
        print(f"reached {target} at t={t}")
        break
```

## What just happened

Each iteration ran the full Active Inference loop:

1. Inferred posterior over hidden state given the current observation.
2. Scored every length-2 policy with expected free energy.
3. Sampled an action from the policy posterior.
4. Propagated the prior with the chosen action's transition matrix.

## What to try next

* Increase `planning_length` to 3 — the agent should reach the goal
  sooner.
* Move the goal to `(0, 2)` and check the action sequence.
* Move on to [`02_multi_agent_dict.md`](02_multi_agent_dict.md) for many
  agents in cadCAD.
