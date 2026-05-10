# `blockference/utils/` — math, plotting, policy templates

Three concerns share this folder, each in its own module:

| Module          | Purpose                                                |
|-----------------|--------------------------------------------------------|
| `utils.py`      | AIF math (entropy, KL, EFE) and plotting helpers.      |
| `policy.py`     | cadCAD policy functions (single, dict-multi).          |
| `mutual_info.py`| Mutual-information CLI for analysing simulation traces.|

## Math layer (`utils.py`)

Builds on `pymdp.maths`/`pymdp.utils`. Key functions:

* `infer_states(observation_index, A, prior)` — softmax(log A + log prior).
* `entropy(A)` — column-wise entropy of an observation model.
* `kl_divergence(qo_u, C)` — KL between expected obs and preferences.
* `calculate_G(A, B, C, qs_current, actions)` — single-step EFE per action.
* `calculate_G_policies(A, B, C, qs_current, policies)` — EFE per policy.
* `compute_prob_actions(actions, policies, Q_pi)` — marginalise to actions.

Plotting utilities require `matplotlib` + `seaborn` and are intended for
notebook usage.

## Policy layer (`policy.py`)

Each policy function follows the canonical cadCAD signature:

```python
def p_actinf_single(params, substep, state_history, previous_state, act, grid):
    ...
```

`act` is the agent (carries `n_states`, `E`, `policy_len`, `border`); `grid`
is the list of coordinate tuples used to translate `(y, x) ↔ index`.

## Mutual-information CLI (`mutual_info.py`)

```bash
python -m blockference.utils.mutual_info path/to/trace.csv --target action
python -m blockference.utils.mutual_info path/to/trace.csv --target action --viz
```

See [`AGENTS.md`](AGENTS.md) for editing rules.
