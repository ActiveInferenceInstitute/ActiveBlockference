# Generalized Notation Notation

[Generalized Notation Notation](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation)
(GNN) is a text format for declaring Active Inference generative models. The
grid-world model this package implements is also declared as a GNN document, so
the likelihood, transition, preference, and prior are readable outside the
Python code and are checked against it.

## Files

* `specs/gridworld_3x3.gnn.md` — the GNN document for the canonical 3x3
  fixture: identity likelihood `A`, deterministic boundary-clamped transition
  `B[next_state, previous_state, action]`, one-hot preference `C`, one-hot
  prior `D`, and the uniform policy prior `E` over the five affordances.
* `specs/gnn-pin.json` — the pin record (schema `gnn-importer-pin/v1`):
  upstream repository, tag `v5.0.3`, commit
  `8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`, recorded date, and SHA-256
  digests of the parser files that validate the document.

The document is generated from the implementation by
`blockference.gnn.render_spec`; the declared matrices come from
`ActiveGridference`, so the document cannot describe a model the package does
not build.

## Commands

```bash
# Regenerate the document, or verify it is current (release gate)
uv run python scripts/export_gnn_spec.py
uv run python scripts/export_gnn_spec.py --check

# Validate with the pinned upstream parser (requires a GNN source or runtime)
GNN_ROOT=/path/to/Generalized_Notation_Notation \
GNN_PYTHON=/path/to/parser-runtime/bin/python \
uv run python scripts/validate_gnn_spec.py --json

# Installed pinned runtime (what CI uses)
uv run python scripts/validate_gnn_spec.py --gnn-runtime /path/to/venv/bin/python --json
```

`GNN_ROOT` is a `Generalized_Notation_Notation` checkout: the validator resolves
the pinned commit with `git archive`, verifies every recorded parser digest, and
runs the parser from that materialized tree, so a live working copy under
development is never imported. `GNN_PYTHON` (or `--gnn-runtime`) selects the
interpreter that has the parser dependencies; the GNN checkout's own `.venv`
works. Exit codes are `0` pass, `1` conformance failure, and `2` pinned runtime
or checkout unavailable — an unavailable run is reported as not run, never as a
pass.

CI's `gnn-conformance` job (display name `gnn conformance (pinned)`) runs the
full `all` selection on Python 3.12, provisioning a separate pinned runtime
because GNN requires NumPy 2.x while the library's committed lock resolves
NumPy 1.x. Locally, the `gnn` arm always runs both specification drift and real
parser conformance, using `--gnn-runtime`, `GNN_RUNTIME_PYTHON`, or `GNN_ROOT`.
An unavailable runtime fails the arm and `all`; no opt-in requirement flag or
skip mode is needed. See [`gate.md`](gate.md) for the canonical gate commands
and exact CI map.

## What the validation checks

1. The pin: the resolved commit equals `specs/gnn-pin.json` and every recorded
   parser file digest matches.
2. The document parses with the pinned `gnn` parser with no errors or warnings,
   and the parser version equals the pinned tag.
3. The declared `ModelParameters` are read back and used to build the runtime
   model, whose `A`/`B`/`C`/`D`/`E` arrays must equal the declared matrices
   exactly; declared `StateSpaceBlock` dimensions must match the array shapes.
4. The pinned semantic validator accepts the document, the B-tensor orientation
   diagnostic reports `canonical`, and the consistency checker reports a
   complete score.
5. The committed document equals `blockference.gnn.render_spec` output.

The receipt is written to `docs/_build/gnn/validation-receipt.json`
(`--receipt` overrides it).

## Python API

`blockference.gnn` exposes the shared construction:

* `spec_parameters()` / `GridWorldSpecParameters` — the canonical fixture.
* `build_spec_model(parameters)` — the runtime model for those parameters.
* `spec_arrays(parameters)` — the declared `A`/`B`/`C`/`D`/`E` arrays.
* `render_spec(parameters)` — the document text.
* `parameters_from_parsed(payload)` / `declared_arrays(payload)` — read a
  pinned parser payload back into parameters and matrices.
* `load_pin()` — the recorded upstream pin, requiring schema
  `gnn-importer-pin/v1`, tag `v5.0.3`, and a full 40-hex commit. Invalid records
  raise `ValueError` naming the field. This interim local reader remains in use
  because the estate pin predates `gnn.importers`; shared-reader adoption requires
  a pin containing that upstream API, not a vendored copy.
  CI provisioning uses this same reader in the locked package environment,
  rather than independently loading unvalidated JSON.

## Scope

The committed document covers the canonical 3x3 fixture. Other grid sizes,
affordance subsets, or preferred/initial cells are described by the same
construction (`GridWorldSpecParameters`) and are not separately committed as
documents. GNN rendering to other frameworks is not used here: this repository
implements and validates the specification, it does not render it to a
framework runtime.

## Related work

[GNN](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation)
owns the notation, parser, and renderers, and its
[framework integration guide](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/blob/main/docs/gnn/integration/framework_integration_guide.md)
documents the render/execute pipeline; the optional `BlockferenceAgent` adapter
connects this package to the `pymdp` ecosystem that GNN also targets.
