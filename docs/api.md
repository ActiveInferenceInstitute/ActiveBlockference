# API reference

The public Python surface of ActiveBlockference. Anything not listed
here is **internal** — it may change without notice.

## Top-level (`blockference`)

| Symbol                          | Kind     | Source                              |
|---------------------------------|----------|-------------------------------------|
| `ActiveGridference`             | class    | `blockference.gridference`          |
| `BlockferenceAgent`             | class    | `blockference.agent`                |
| `ExperimentConfig`              | class    | `blockference.config`               |
| `PipelineResult`                | class    | `blockference.pipeline`             |
| `RunPaths`                      | class    | `blockference.io`                   |
| `ValidationReport`              | class    | `blockference.io`                   |
| `actinf_planning_single`        | function | `blockference.gridference`          |
| `actinf_graph`                  | function | `blockference.gridference`          |
| `build_run_paths`               | function | `blockference.io`                   |
| `load_experiment_config`        | function | `blockference.config`               |
| `make_grid`                     | function | `blockference.gridference`          |
| `run_pipeline`                  | function | `blockference.pipeline`             |
| `validate_run_outputs`          | function | `blockference.io`                   |
| `validate_trajectory_dataframe` | function | `blockference.io`                   |
| `__version__`                   | str      | `blockference`                      |

## `blockference.maths` (self-contained AIF primitives)

Replaces all legacy `pymdp.maths` / `pymdp.utils` helpers; pure NumPy,
no JAX dependency.

| Symbol               | Returns                  | Notes                                       |
|----------------------|--------------------------|---------------------------------------------|
| `MIN_PROB`           | `float`                  | `1e-16` — log-floor constant.               |
| `log_stable(x)`      | `np.ndarray`             | Clipped natural log.                        |
| `softmax(x, axis)`   | `np.ndarray`             | Numerically stable.                         |
| `norm_dist(x, axis)` | `np.ndarray`             | Sum-to-1 along axis; uniform on zero sums.  |
| `sample(p, rng=None)`| `int`                    | Honours `np.random.seed` when `rng=None`.   |
| `onehot(i, n)`       | `np.ndarray`             | Length-`n` one-hot.                         |
| `construct_policies(num_states, num_controls, policy_len, control_fac_idx=None)` | `list[np.ndarray]` | Drop-in for `pymdp.control.construct_policies` returning NumPy. |

## `blockference.gridference`

### `class ActiveGridference(grid, planning_length=2, env_state=(0, 0))`

Generative model + state container for a single Active Inference grid
agent.

**Attributes** (set in `__init__` or via `get_*`):

| Attribute            | Type                  | Meaning                                       |
|----------------------|-----------------------|-----------------------------------------------|
| `A`                  | `np.ndarray`          | Likelihood `P(o|s)` (identity by default).    |
| `B`                  | `np.ndarray`          | Transition tensor `B[s', s, a]`.              |
| `C`                  | `np.ndarray`          | Prior preferences over observations.          |
| `D`                  | `np.ndarray`          | Prior over initial hidden state.              |
| `E`                  | `list[str]`           | Affordance labels.                            |
| `grid`               | `list[tuple[int,int]]`| Coordinate-to-index mapping.                  |
| `policy_len`         | `int`                 | Planning horizon.                             |
| `n_states`           | `int`                 | `len(grid)`.                                  |
| `n_observations`     | `int`                 | `len(grid)`.                                  |
| `border`             | `int`                 | `sqrt(n_states) - 1`.                         |
| `prior`              | `np.ndarray`          | Current prior over hidden states.             |
| `current_action`     | `int | str`           | Last sampled action (str initially).          |
| `current_inference`  | `np.ndarray | str`    | Last posterior (str initially).               |
| `env_state`          | `tuple[int,int]`      | Current `(y, x)` location.                    |

**Methods:**

* `get_A()` — build identity likelihood.
* `get_B()` — build transition tensor for the 5-action grid.
* `get_C(preferred_state: tuple)` — build one-hot preferences.
* `get_D(initial_state)` — build one-hot prior; sets `prior = D`.
* `get_E(actions: list)` — override the affordance set.

### `actinf_planning_single(agent, env_state, A, B, C, prior) -> dict`

Run one Active Inference inference + planning step for a single agent.

Returns a dict with keys `update_prior`, `update_env`, `update_action`,
`update_inference`.

### `actinf_graph(agent_network) -> dict`

Run one step per node of a `networkx` graph. Each node is expected to
expose `agent`, `env_state`, `prior`, `prior_A`, `prior_B`, `prior_C`.

Returns `{"agent_updates": [ ... ]}`.

### `make_grid(grid_len: int, grid_dim: int = 2) -> list[tuple]`

Cartesian product of `range(grid_len)` × `range(grid_len)` (or higher
dimensions).

## `blockference.agent`

### `class BlockferenceAgent(*args, **kwargs)`

Subclass of `pymdp.agent.Agent`. Forwards all constructor arguments
verbatim; exists as a hook for future extensions. Aliased as `Agent` for
backwards compatibility.

## `blockference.envs`

### `class GridAgent(grid_len, grid_dim=2, agents=None)`

Single-/multi-agent grid environment with simple `step(actions)` and
`get_rel_pos(loc1, loc2)` helpers.

### `class TwoMultiGridAgent(grid_len, grid_dim=2, agents=None, init_pos=None, init_obs=None)`

Two-agent grid with collision handling and observation generation
including the *other* agent's location.

## `blockference.simulations`

### `run_grid(dimension, no_agents, no_timesteps, output_path="result.csv") -> pandas.DataFrame`

Run the full multi-agent gridworld experiment, write CSV, return the
result frame.

## `blockference.utils.utils`

| Function                                                | Returns                                |
|----------------------------------------------------------|----------------------------------------|
| `infer_states(observation_index, A, prior)`              | posterior `q(s_t)`                     |
| `entropy(A)`                                             | per-column entropy of `A`              |
| `kl_divergence(qo_u, C)`                                 | KL between predicted obs and `C`       |
| `get_expected_states(B, qs_current, action)`             | `q(s_{t+1} | a)`                       |
| `get_expected_observations(A, qs_u)`                     | `q(o_{t+1} | a)`                       |
| `calculate_G(A, B, C, qs_current, actions)`              | EFE per action (1 step)                |
| `calculate_G_policies(A, B, C, qs_current, policies)`    | EFE per policy (`policy_len` steps)    |
| `compute_prob_actions(actions, policies, Q_pi)`          | marginal action probability vector     |

Plus plotting helpers (`plot_grid`, `plot_likelihood`, `plot_beliefs`,
`plot_point_on_grid`).

## `blockference.utils.policy`

* `p_actinf_single(params, substep, state_history, previous_state, act, grid)`
* `p_actinf_dict(params, substep, state_history, previous_state, grid)`

Both implement one cadCAD policy step of Active Inference (single-agent
and dict-multi-agent respectively).

## `blockference.utils.mutual_info`

* `calculate_mi(X: pd.DataFrame, y: pd.Series) -> pd.Series`
* CLI: `python -m blockference.utils.mutual_info <csv> --target <col> [--viz]`
