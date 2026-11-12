# Gate and CI

One entry point runs every local gate:
[`../scripts/release_check.py`](../scripts/release_check.py). Each arm is a
separately runnable selection, and CI invokes the same arms instead of keeping a
second copy of any command.

```bash
uv run python scripts/release_check.py              # fast arms (default)
uv run python scripts/release_check.py --arm all --gnn-runtime /path/to/gnn-runtime/bin/python
uv run python scripts/release_check.py --list       # arms and their commands
```

Run `uv sync --locked --extra dev` before the gate. Its child commands set
`UV_NO_SYNC=1` to reuse that prepared project environment instead of repeating
dependency synchronization.

Measured wall times and test counts live in the generated
[`_generated/gate-metrics.md`](_generated/gate-metrics.md), which shows the
recorded before/after arm times and labels baseline workloads that are not
equivalent to their arms. These are one-run, host-specific receipts, not a
controlled benchmark or a claimed speedup. The offline package baseline failed
because its required package was absent from the local cache; preserve that
failure rather than comparing it as a successful workload. Do not hand-edit
measurements.

## Arms

| Arm | Command | In `all` |
| --- | --- | --- |
| `lint` | `uv run ruff check blockference tests GRTs scripts` | yes |
| `types` | `uv run mypy blockference` (configured at the `requires-python` floor) | yes |
| `test` | `uv run pytest -m "not slow"` | no |
| `test-full` | `uv run pytest --cov=blockference --cov-report=term --cov-report=xml --cov-fail-under=90` | yes |
| `docs` | `scripts/check_docs_links.py`, `scripts/gate_metrics.py --check` | yes |
| `manuscript` | `scripts/manuscript_variables.py --check`, `scripts/validate_manuscript.py`, `scripts/build_manuscript.py`, `scripts/build_manuscript.py --check`; the `gnn_*` variables come from `gnn.importers` in the pinned runtime | yes |
| `gnn` | `scripts/export_gnn_spec.py --check`, then pinned `scripts/validate_gnn_spec.py --verify-tag-remote` conformance (parser digests verified before anything parses, the pin's tag resolved upstream to the pinned commit; needs network), then the shared `python -m gnn.importers check --repo .` pin and pointer check; missing runtime fails | yes |
| `smoke` | pipeline, validation, and standalone `run` JSON smoke checks in a temporary directory | yes |
| `package` | `uv build`, locked `uv export` constraints for clean-wheel dependencies, and isolated import outside the source tree | yes |
| `render` | `scripts/build_manuscript.py --render pdf` (Pandoc + XeLaTeX) | no |

`--arm` is repeatable and arms always run in the canonical order above. `fast`
is the default because it is the selection used while editing; `all` is the
release gate. The `slow` marker (declared in `pyproject.toml`) keeps notebook
execution, real-pipeline integration, CLI smoke, and figure-byte checks out of
the fast arm. They stay in `test-full`.

## GNN conformance

The canonical pin is [`../specs/gnn-pin.json`](../specs/gnn-pin.json) with schema
`gnn-importer-pin/v1`. It pins
`ActiveInferenceInstitute/Generalized_Notation_Notation` at `v5.1.0`
(`75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf`) and records SHA-256 digests for the
five parser files under `src/gnn/`. The pointer
[`../.github/GNN.md`](../.github/GNN.md) records exactly what the conformance
job parses.

The conformance job's key is `gnn-conformance` and its display name is
`gnn conformance (pinned)`.

The pin, the pointer and the specification are read by the estate's shared
reader, `gnn.importers`, so this repository carries no copy of the pin shape. The
`gnn` arm runs the offline drift check, then real pinned-parser conformance, then
`python -m gnn.importers check --repo . --require-pointer
--workflow .github/workflows/ci.yml --job gnn-conformance` from the pinned
runtime. Conformance precedes that reader check because `gnn.importers check`
parses the document with the installed parser, while the validator verifies the
pin's parser digests before it parses anything: no arm command reaches the
parser before its digests are verified.

Select an installed pinned runtime with `--gnn-runtime` or
`GNN_RUNTIME_PYTHON`, or supply a pinned source checkout via `GNN_ROOT` as
described in [`gnn.md`](gnn.md); with `GNN_ROOT` and no runtime named, the
checkout's own `.venv/bin/python` is used. Missing or invalid runtime
dependencies fail the arm and therefore `all`; conformance is never silently
skipped.

## GitHub Actions

[`../.github/workflows/ci.yml`](../.github/workflows/ci.yml) calls these arms:

| Job | Arms |
| --- | --- |
| `test` (matrix: Python 3.11 only) | `--arm test-full` |
| `gnn-conformance` (Python 3.12; display name `gnn conformance (pinned)`) | `--arm all`, with `GNN_RUNTIME_PYTHON` set to the provisioned runtime |

The full suite therefore runs on both supported CI Python versions. The
remaining release arms run in `gnn-conformance` on Python 3.12, alongside the
real pinned conformance check; there is no separate `gate` job. The pinned GNN
runtime remains separate from the locked library environment because their
NumPy requirements differ. PDF rendering is an explicit local `--arm render`
selection, not part of `all` or CI.

The workflow installs uv directly with
`astral-sh/setup-uv@68559e99cc807c1eb7e9238fd43a15bba460eec2` (`v10.3.3`,
pinned by commit because setup-uv publishes no floating `v10` tag). It does not
use the Institute composite action
`ActiveInferenceInstitute/.github/.github/actions/setup-python-uv@7159e67315330aac8e414aebae0ad1e999a41448`;
the workflow installs the locked environment directly.

## Recording measurements

`release_check.py --timings PATH` writes the actual commands, statuses, and
elapsed times for the selected run, including a failed arm. Use a fresh receipt
path for each run; selecting `render` separately records the explicit local PDF
build without adding it to CI. A passing render arm also records `artifact`:
the repository-relative PDF path, exact byte count, SHA-256, and source commit
(`git rev-parse HEAD`). Commit source changes before recording a release render
so that commit identifies the rendered source. The PDF remains ignored; the
receipt identifies the local bytes without publishing the binary.

Import receipts before regenerating the checked metrics page:

```bash
uv run python scripts/gate_metrics.py \
  --record-baseline /path/to/before-receipt.json \
  --record-baseline /path/to/before-render-receipt.json \
  --record-timings /path/to/after-receipt.json
uv run python scripts/gate_metrics.py --check
```

Replace these paths with receipts produced by real gate runs. `--record-baseline`
is repeatable for disjoint selections; `--record-timings` accepts one combined
`--arm all --arm test --arm render` receipt. Do not synthesize timings or turn
failed, missing, or non-equivalent measurements into passes.
