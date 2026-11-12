# GNN conformance

`gnn conformance (pinned)` is the display name of job `gnn-conformance` in
`.github/workflows/ci.yml`. It reads `specs/gnn-pin.json` (schema
`gnn-importer-pin/v1`), which pins
`ActiveInferenceInstitute/Generalized_Notation_Notation` at `v5.0.3`
(`8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`) and records SHA-256 digests for
the five parser files under `src/gnn/`.

The job installs that commit into a separate runtime, then runs
`python scripts/validate_gnn_spec.py --gnn-runtime <runtime> --json` and checks
that the resolved `gnn.__version__` matches the pin tag. The validator verifies
the installed parser file digests, parses `specs/gridworld_3x3.gnn.md` with the
pinned `MarkdownGNNParser`, and compares the declared parameters and the `A`,
`B`, `C`, `D`, and `E` matrices against the runtime model built by
`blockference.gnn`; it also requires the pinned semantic validator, the
B-orientation check, and the consistency checker to report no errors or
warnings. It does not scan the rest of the repository for `*.gnn.md`.
