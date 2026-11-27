# GNN conformance

`gnn conformance (pinned)` is the display name of job `gnn-conformance` in
`.github/workflows/ci.yml`. It reads `specs/gnn-pin.json` (schema
`gnn-importer-pin/v1`), which pins
`ActiveInferenceInstitute/Generalized_Notation_Notation` at `v5.1.0`
(`75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf`) and records SHA-256 digests for
the five parser files under `src/gnn/`.

The job selects the canonical release gate with `--arm all` and a separate
`GNN_RUNTIME_PYTHON`. It must also receive independently reviewed
`GNN_BOOTSTRAP` and `GNN_SOURCE_MANIFEST` resources before hosted acceptance.
Within the `gnn` arm, exporter drift is checked first; the project validator
then owns deep conformance and the producer's syntax/pointer/workflow/job
check in separate isolated qualified lifetimes. The shared producer remains
authoritative for pin discovery and validation. Missing resources, runtime,
installed provenance, or exit revalidation fail rather than skip. See
[`docs/development/gate.md`](../docs/development/gate.md) for the complete job and arm map, and
[`docs/reference/gnn.md`](../docs/reference/gnn.md) for standalone validation commands.

Independent full-package content and actual module ownership are admitted
before target GNN execution and revalidated at verdict exit. The validator
also checks version and recorded parser hashes, parses the committed document
with the pinned `MarkdownGNNParser`, and compares declared parameters and `A`,
`B`, `C`, `D`, and `E` matrices against the runtime model built by
`blockference.gnn`; it also requires the pinned semantic validator, the
B-orientation check, and the consistency checker to report no errors or
warnings. It does not scan the rest of the repository for `*.gnn.md`.
