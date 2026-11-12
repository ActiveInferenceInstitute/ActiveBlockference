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
  upstream repository, tag `v5.1.0`, commit
  `75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf`, recorded date, and SHA-256
  digests of the parser files that validate the document.

The document is generated from the implementation by
`blockference.gnn.render_spec`; the declared matrices come from
`ActiveGridference`, so the document cannot describe a model the package does
not build.

## The pin is read by `gnn.importers`, not here

The pin record is read and validated by the estate's shared reader,
`gnn.importers` in the pinned upstream revision. This repository keeps no copy
of that logic:

* the artifact uses `read_pin`, `find_pin`, `check_pointer` and
  `importer_variables` from that module;
* the release gate's `gnn` arm calls `python -m gnn.importers check --repo .`
  from the pinned runtime;
* `scripts/gnn_importer_report.py` is the only bridge: it runs in an
  interpreter where `gnn` is importable and prints the normalized pin and the
  `gnn-importer-variables/v1` map as JSON for
  `scripts/validate_gnn_spec.py` and `scripts/manuscript_variables.py`.

The cutover replaced the interim local reader. Until this change `specs/gnn-pin.json`
pinned `v5.0.3` at `8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`, a revision that
predates `gnn.importers`, so `blockference.gnn.load_pin` rejected records outside
`gnn-importer-pin/v1`, required tag `v5.0.3`, and required a full 40-hex commit.
The pin now names `v5.1.0` at
`75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf`, the first release that ships
`src/gnn/importers.py`, and `load_pin` is deleted rather than reimplemented.
`python -m gnn.importers` cannot run against the old revision, which is why the
adoption follows the pin and not the contract date.

## Commands

```bash
# The shared reader, from the pinned runtime (release gate, `gnn` arm)
"$GNN_RUNTIME_PYTHON" -m gnn.importers check --repo . --require-pointer \
    --workflow .github/workflows/ci.yml --job gnn-conformance

# Regenerate the document, or verify it is current (release gate)
uv run python scripts/export_gnn_spec.py
uv run python scripts/export_gnn_spec.py --check

# Validate with the pinned upstream parser (requires a GNN source or runtime)
GNN_ROOT=/path/to/Generalized_Notation_Notation \
GNN_PYTHON=/path/to/parser-runtime/bin/python \
uv run python scripts/validate_gnn_spec.py --json

# Installed pinned runtime (what CI uses); --verify-tag-remote resolves the
# pin's tag upstream with `git ls-remote` and needs network access
uv run python scripts/validate_gnn_spec.py --gnn-runtime /path/to/venv/bin/python \
  --verify-tag-remote --json
```

`GNN_ROOT` is a `Generalized_Notation_Notation` checkout: the validator resolves
the pinned commit with `git archive`, verifies every recorded parser digest, and
runs the parser from that materialized tree, so a live working copy under
development is never imported. `GNN_PYTHON` (or `--gnn-runtime`) selects the
interpreter that has the parser dependencies; the GNN checkout's own `.venv`
works. Exit codes are `0` pass, `1` conformance failure, and `2` pinned runtime,
checkout, or requested remote tag resolution unavailable — an unavailable run is reported as not run, never as a
pass. With `GNN_ROOT` set and no runtime named, the gate uses the checkout's own
`.venv/bin/python` for both the reader and the pin.

CI's `gnn-conformance` job (display name `gnn conformance (pinned)`) runs the
full `all` selection on Python 3.12, provisioning a separate pinned runtime
because GNN requires NumPy 2.x while the library's committed lock resolves
NumPy 1.x. That runtime is also what the `manuscript` arm uses for the `gnn_*`
documented variables, whose values come from `gnn.importers.importer_variables`.
An unavailable runtime fails the `gnn` arm and `all`; no opt-in requirement flag
or skip mode is needed. See [`gate.md`](gate.md) for the canonical gate commands
and exact CI map.

## What the validation checks

1. The pin and the pointer: `gnn.importers` reports the pin file, its schema,
   `repository`, `tag` and 40-hex `commit`, and requires
   [`.github/GNN.md`](../.github/GNN.md) to name the pin file, the pinned
   revision, `.github/workflows/ci.yml`, and job `gnn-conformance`.
2. The document parses with the pinned `gnn` parser with no errors or warnings,
   and the parser version equals the pinned tag; every recorded parser digest
   matches the installed (or archived) parser files, and for an installed
   runtime that check runs *before* the document is parsed and the parser must be
   imported from the verified package.
   The pin's `tag` must resolve to the pinned `commit` (`tag-commit-mismatch`):
   locally in a `GNN_ROOT` checkout, or upstream with `--verify-tag-remote`
   (`git ls-remote`, the peeled commit of an annotated tag; the release gate's
   `gnn` arm passes the flag, so it needs network access). The commit the
   installer recorded for the runtime (`direct_url.json` `vcs_info.commit_id`)
   must equal the pin's commit. Without a checkout or the flag, and for an
   install with no recorded VCS commit, the receipt reports `tag_binding` /
   `revision` as `unresolvable-offline` rather than a pass; with
   `--verify-tag-remote`, a tag that cannot be resolved ends the run as *not
   run* (exit `2`).
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

Reading the pin is not part of this package: `gnn.importers` owns that, and the
gate calls it from the pinned runtime.

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
connects this package to the `pymdp` ecosystem that GNN also targets. The
GNN–FEP workflow-group contract that binds the twenty-two importing repositories
to this one reader is
[`docs/workflows/gnn-fep.md`](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/blob/main/docs/workflows/gnn-fep.md).
