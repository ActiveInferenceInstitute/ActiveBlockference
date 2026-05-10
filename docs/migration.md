# Migrating from `inferactively-pymdp` 0.0.x → 1.0.x

ActiveBlockference v0.5+ targets `inferactively-pymdp >= 1.0.1`. The 1.0
release is a JAX/Equinox rewrite that removes or relocates many of the
helpers the legacy 0.0.x series exposed:

| Legacy symbol                                | Status in 1.0+              | Replacement in ActiveBlockference                |
|---------------------------------------------|-----------------------------|--------------------------------------------------|
| `pymdp.maths.spm_log_single`                | removed                     | `blockference.maths.log_stable`                  |
| `pymdp.maths.softmax`                       | removed                     | `blockference.maths.softmax`                     |
| `pymdp.utils.norm_dist`                     | removed                     | `blockference.maths.norm_dist`                   |
| `pymdp.utils.sample`                        | removed                     | `blockference.maths.sample`                      |
| `pymdp.utils.onehot`                        | removed                     | `blockference.maths.onehot`                      |
| `pymdp.control.construct_policies`          | present, returns JAX array  | `blockference.maths.construct_policies` (numpy)  |
| `pymdp.agent.Agent` (legacy ctor)           | new ctor: per-modality lists| `blockference.agent.BlockferenceAgent` (passthru)|

## Why we own the math layer

The legacy primitives are small, well-defined NumPy operations. Owning
them inside ``blockference.maths`` gives us:

* **Numerical stability** we control (e.g. ``MIN_PROB = 1e-16`` floor on
  ``log_stable``).
* **Determinism** under the legacy ``np.random.seed(...)`` idiom, which
  many of the project's notebooks and tests rely on.
* **Independence** from upstream's API churn — pymdp may rewrite again,
  and we don't need to follow every time.
* **No JAX in cadCAD state** — cadCAD/radCAD doesn't compose well with
  JAX tracers, so keeping the inference loop pure NumPy is a hard
  requirement.

We still install pymdp 1.0+ so callers who *want* the high-level Agent
class can opt in via `BlockferenceAgent`.

## What changed in `BlockferenceAgent`

The pymdp 1.0 ``Agent`` constructor expects:

* ``A`` — a list of JAX arrays, one per observation modality.
* ``B`` — a list of JAX arrays, one per hidden-state factor.
* Renamed kwargs (``inference_algo='VANILLA'`` → ``'fpi'``, etc.).

`BlockferenceAgent` is a thin extension hook over the 1.0 ``Agent`` —
all kwargs are forwarded verbatim. ActiveBlockference's grid pipeline
does **not** use it; the per-step inference loop runs the math directly.

## What changed for callers

If you only use `from blockference import ...`:

* **Nothing.** The public surface is unchanged.

If you imported pymdp helpers directly:

* Replace `from pymdp.maths import softmax, spm_log_single` with
  `from blockference.maths import softmax, log_stable`.
* Replace `from pymdp.utils import sample, onehot, norm_dist` with
  `from blockference.maths import sample, onehot, norm_dist`.
* Replace `from pymdp.control import construct_policies` with
  `from blockference.maths import construct_policies` if you want a
  NumPy array back.

## Verifying the migration

```bash
uv run pytest tests/test_maths.py -v       # 21 unit tests for primitives
uv run pytest tests/                       # full suite — currently 103/103
uv run python -m blockference.simulations.grid_sim --pipeline configs/example.yml
ls output/gridworld_example/                # full artefact tree present
cat output/gridworld_example/run.log        # structured log of the run
cat output/gridworld_example/validation_report.json   # all checks pass
```
