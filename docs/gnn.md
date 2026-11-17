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
  prior `D`, and a numeric uniform policy prior `E` over 25 length-two policies.
* `specs/gnn-pin.json` — the pin record (schema `gnn-importer-pin/v1`):
  upstream repository, tag `v5.1.0`, commit
  `75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf`, recorded date, and SHA-256
  digests of the parser files that validate the document.

The document is generated from the implementation by
`blockference.gnn.render_spec`; the declared matrices come from
`ActiveGridference`, so the document cannot describe a model the package does
not build.

Exported `E`, `pi`, and `G` have complete-policy cardinality; `B` retains
action cardinality. Runtime `ActiveGridference.E` remains the ordered action
labels. Three actions at horizon two produce nine policies. Float text keeps
round-trip precision, and a singleton vector remains a tuple rather than a
scalar. The pinned parser canonicalizes source `pi` to `π`; parsed-payload
dimension checks use `π`.

## Shared producer admission

`blockference/gnn_source.py` is a stdlib-only external worker, launched directly
with `-I -B` in the selected GNN interpreter. It first checks and executes the
independently reviewed producer bootstrap, without importing target `gnn`.
The bootstrap owns pin discovery, reading, and manuscript metadata inspection;
inspection reports `qualified=false` and is not a parser verdict.

Every actual parser, semantic, orientation, consistency, and shared
`gnn.importers` check runs inside producer admission through final import/content
revalidation. The project validator owns these separate guarded lifetimes.
There is no direct unchecked `-m gnn.importers` gate command, copied local pin
reader, or surviving `gnn_importer_report.py`/`gnn_parser_probe.py` entry point.

Supply `GNN_BOOTSTRAP` and `GNN_SOURCE_MANIFEST` from independently reviewed
producer artifacts. The source-owned expected SHA-256 values are:

* bootstrap: `f638d4c01fe7de31e43d793e59bf382d0c3d1a82f365c59238334a2b40adfb37`;
* full v5.1.0 manifest: `43b33c0b2b40386afa3f306fec4cc13bc1fd7c3b570fbf11d5008bad5c8e6b51`.

Do not derive expected identity from the target package or replace complete
content/module ownership with a version string or the five parser digests.
The hosted workflow acquires these two files from the published immutable
[producer resource revision `66d491f`](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/tree/66d491f86895443cf588011cc6f40c37b8bae8fd/interop),
checks both digests before use, and forwards their paths to the release gate.
The full manifest is for v5.1.0, not the separately published v5.0.3 manifest;
the resource revision does not change `specs/gnn-pin.json`.

The frozen bootstrap has a known contextual origin-normalization defect: a
non-GitHub URL can be misclassified as the canonical repository identity.
Existing installed proof with the actual canonical HTTPS GitHub origin remains
valid only in its recorded scope; source-artifact proof remains
`vcs_verified=false`. Resource delivery does not repair or waive this defect.
Adopting a strict-origin replacement requires the producer's completed native
qualification and a newly published immutable resource identity. Do not adopt
an unpublished private candidate or modify the historical artifact.

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
# Select existing independently reviewed resources and a prepared interpreter.
export GNN_BOOTSTRAP=/reviewed/gnn_bootstrap.py
export GNN_SOURCE_MANIFEST=/reviewed/v5.1.0-source-manifest.json
export GNN_RUNTIME_PYTHON=/prepared/gnn/bin/python

# Regenerate the document, or verify it is current (release gate)
uv run python scripts/export_gnn_spec.py
uv run python scripts/export_gnn_spec.py --check

# Explicit source checkout: materialize the exact pin, then validate.
uv run python scripts/validate_gnn_spec.py \
  --gnn-root /path/to/Generalized_Notation_Notation --json

# Installed mode: requires real matching PEP610 Git origin and full content.
# --verify-tag-remote additionally needs network access.
uv run python scripts/validate_gnn_spec.py --verify-tag-remote --json
```

Interpreter precedence is explicit `--gnn-runtime`, `GNN_RUNTIME_PYTHON`,
explicit legacy `--python`, `GNN_PYTHON`, then the selected `GNN_ROOT/.venv`
launcher. A missing explicit selection fails; it never substitutes project
Python. Venv launcher paths are preserved. The project and GNN NumPy
environments remain separate.

For a Git checkout, the project validator resolves and archives the exact pin
before execution. A separately admitted plain source tree is explicitly
source-only: complete-content/import proof does not invent Git or installed
provenance, and plain-artifact receipts retain `vcs_verified=false`.
Installed mode requires actual distribution ownership, canonical PEP610 Git
origin with the pinned commit, and complete package content. An editable-local
installation or missing origin is not an installed pass and cannot fall back
to source.

Exit codes are `0` pass, `1` conformance failure, and `2` unavailable required
runtime/resources/origin or requested tag resolution. Missing prerequisites
are not passes. CI provisions the checksum-verified reviewed bootstrap and full
manifest alongside its pinned runtime; this wiring is not itself hosted,
installed, or full-release acceptance. See [`gate.md`](gate.md).

## What the validation checks

1. The shared producer reports the pin and requires
   [`.github/GNN.md`](../.github/GNN.md) to name its file, revision, workflow
   `.github/workflows/ci.yml`, and job `gnn-conformance`.
2. Independent bootstrap/full-manifest admission precedes all target GNN
   execution. Actual imported modules and verdict exit revalidation must
   remain qualified. Parsing has no errors or warnings and the version
   matches the pinned tag.
   The tag must resolve to the pinned commit: locally for a Git checkout, or
   upstream when `--verify-tag-remote` is requested. Remote verification
   always queries upstream rather than trusting a stale local tag. Missing
   offline tag/revision observations remain `unresolvable-offline`; they are
   not invented VCS proof or a substitute for installed-origin admission.
3. The declared `ModelParameters` are read back and used to build the runtime
   model, whose `A`/`B`/`C`/`D`/`E` arrays must equal the declared matrices
   exactly; state-space dimensions and redundant action/policy counts must
   agree. Parsed `π` and `G` have complete-policy cardinality.
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

The producer owns pin inspection and execution admission. Manuscript `gnn_*`
metadata is inspection only, not an installed, source-execution, or scientific
qualification result.

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
