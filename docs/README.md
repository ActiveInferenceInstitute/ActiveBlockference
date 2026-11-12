# Documentation

Start here if you are new to the repository:

1. [`../README.md`](../README.md) — installation, quick start, public API, and
   repository map.
2. [`documentation-index.md`](documentation-index.md) — flat index of every
   long-lived document.
3. [`tutorials/`](tutorials/) — short runnable examples.
4. [`api.md`](api.md) — public Python and CLI contracts.
5. [`pipeline.md`](pipeline.md) — configuration, output tree, and validation.

Generated facts live under [`_generated/`](_generated/README.md). Measured gate
numbers are in [`_generated/gate-metrics.md`](_generated/gate-metrics.md); do
not copy those numbers into prose.
The receipts retain failed and non-equivalent workloads, including the offline
package baseline cache miss; they do not establish a speedup.

## Concepts and design

* [`theory.md`](theory.md) — the discrete Active Inference loop and matrix
  conventions.
* [`architecture.md`](architecture.md) — module boundaries and data flow.
* [`gnn.md`](gnn.md) — the GNN declaration of the grid-world model, the pinned
  parser validation, and its commands.
* [`design-contract.md`](design-contract.md) — scope, constraints, and release
  gates.
* [`glossary.md`](glossary.md) — recurring terms and symbols.
* [`migration-v1.md`](migration-v1.md) — compatibility and upgrade boundaries.

## Working in the repository

* [`development.md`](development.md) — locked development and verification
  commands.
* [`gate.md`](gate.md) — gate arms, CI job map, and where the measured
  numbers live.
* [`contributing.md`](contributing.md) — contribution flow and issue reporting.
* [`../configs/README.md`](../configs/README.md) — configuration examples and
  schema notes.
* [`../notebooks/README.md`](../notebooks/README.md) — executable notebook
  examples.
* [`../scripts/README.md`](../scripts/README.md) — release and publication tools.
* [`../tests/README.md`](../tests/README.md) — test layout and gate coverage.
* [`../.github/workflows/README.md`](../.github/workflows/README.md) — workflow
  entry point; [`gate.md`](gate.md) owns the exact two-job CI map.
* [`../SECURITY.md`](../SECURITY.md) — private security reporting.
* [`../TODO.md`](../TODO.md) — documentation completion record.
* [`../GRTs/README.md`](../GRTs/README.md) — optional local research orchestration.
* [`../.aii/README.md`](../.aii/README.md) — InstituteOS metadata sidecar.

## Publication source

* [`manuscript/README.md`](manuscript/README.md) — ordered publication source,
  formal definitions, numbered equations, bibliography, and build commands.
* [`manuscript/SYNTAX.md`](manuscript/SYNTAX.md) — publication syntax and variables.
* [`manuscript/figure-registry.json`](manuscript/figure-registry.json) — figure
  sources and artifact mapping.

The software verdicts described throughout these documents certify software and
artefact integrity only; they do not establish empirical adequacy of a
scientific hypothesis.
