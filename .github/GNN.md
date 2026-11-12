# GNN conformance

`gnn conformance (pinned)` is the display name of job `gnn-conformance` in
`.github/workflows/ci.yml`. It reads `specs/gnn-pin.json` (schema
`gnn-importer-pin/v1`), which pins
`ActiveInferenceInstitute/Generalized_Notation_Notation` at `v5.1.0`
(`75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf`) and records SHA-256 digests for
the five parser files under `src/gnn/`.

The job installs that commit into a separate Python 3.12 runtime, then invokes
the canonical release gate with `--arm all` and `GNN_RUNTIME_PYTHON` set to that runtime. Its `gnn`
arm runs the estate's shared reader,
`python -m gnn.importers check --repo . --require-pointer --workflow
.github/workflows/ci.yml --job gnn-conformance`, from that runtime, then the
offline drift check and the deep pinned-parser conformance check. The reader
owns pin discovery and validation, and reports a pointer that fails to name the
pin file, the pinned revision, the workflow or the job. A missing runtime
fails instead of skipping conformance. See
[`docs/gate.md`](../docs/gate.md) for the complete job and arm map, and
[`docs/gnn.md`](../docs/gnn.md) for standalone validation commands.

The validator checks that the resolved `gnn.__version__` matches the pin tag,
verifies the installed parser file digests, parses `specs/gridworld_3x3.gnn.md`
with the pinned `MarkdownGNNParser`, and compares the declared parameters and the `A`,
`B`, `C`, `D`, and `E` matrices against the runtime model built by
`blockference.gnn`; it also requires the pinned semantic validator, the
B-orientation check, and the consistency checker to report no errors or
warnings. It does not scan the rest of the repository for `*.gnn.md`.
