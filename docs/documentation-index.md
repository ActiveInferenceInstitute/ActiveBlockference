# Documentation index

Flat index of the repository's long-lived documents. Generated facts live under
[`_generated/`](_generated/README.md) and are never edited by hand.

## Entry points

- [`../README.md`](../README.md) — installation, quick start, public API, and repository map.
- [`README.md`](README.md) — documentation hub and reading order.
- [`../.github/README.md`](../.github/README.md) — GitHub-facing signpost.
- [`contributing.md`](contributing.md) — contribution flow.
- [`../AGENTS.md`](../AGENTS.md) — repository contributor guidance; folder-local `AGENTS.md` files refine it.
- [`../SECURITY.md`](../SECURITY.md) — private security reporting.
- [`../LICENSE`](../LICENSE) — license terms.
- [`../TODO.md`](../TODO.md) — documentation findings and completion record.

## Concepts and contracts

- [`theory.md`](theory.md) — discrete Active Inference loop and matrix conventions.
- [`architecture.md`](architecture.md) — module boundaries and data flow.
- [`api.md`](api.md) — public Python and CLI contracts.
- [`pipeline.md`](pipeline.md) — configuration, output tree, and validation.
- [`design-contract.md`](design-contract.md) — scope, constraints, and release gates.
- [`glossary.md`](glossary.md) — recurring terms and symbols.
- [`migration-v1.md`](migration-v1.md) — compatibility and upgrade boundaries.

## Specification and conformance

- [`gnn.md`](gnn.md) — GNN declaration, pinned parser validation, and commands.
- [`../specs/README.md`](../specs/README.md) — GNN document and pin record.
- [`../.github/GNN.md`](../.github/GNN.md) — what the conformance job parses.
- [`../specs/gridworld_3x3.gnn.md`](../specs/gridworld_3x3.gnn.md) — canonical generated model declaration.
- [`../specs/gnn-pin.json`](../specs/gnn-pin.json) — authoritative upstream parser pin and digests.

## Development and operations

- [`development.md`](development.md) — locked development workflow.
- [`gate.md`](gate.md) — gate arms, CI job map, and generated metrics.
- [`../scripts/README.md`](../scripts/README.md) — gate and generator scripts.
- [`../tests/README.md`](../tests/README.md) — test layout and gate equivalence.
- [`../.github/workflows/README.md`](../.github/workflows/README.md) — workflow entry point.
- [`../tests/envs/README.md`](../tests/envs/README.md) — environment test conventions.
- [`../tests/utils/README.md`](../tests/utils/README.md) — utility test conventions.
- [`_generated/README.md`](_generated/README.md) — generated-document producers and receipt imports.
- [`_generated/gate-metrics.md`](_generated/gate-metrics.md) — measured arm runtimes and test counts (generated).
- [`_generated/manuscript-variables.json`](_generated/manuscript-variables.json) — declared manuscript variables (generated).
- [`_generated/gate-baseline.json`](_generated/gate-baseline.json) — one-run historical baseline receipt, including failures.
- [`_generated/gate-timings.json`](_generated/gate-timings.json) — recorded current gate receipts.

## Tutorials and examples

- [`tutorials/README.md`](tutorials/README.md) — tutorial index.
- [`tutorials/01_single_agent.md`](tutorials/01_single_agent.md)
- [`tutorials/02_multi_agent_dict.md`](tutorials/02_multi_agent_dict.md)
- [`tutorials/03_cadcad_pipeline.md`](tutorials/03_cadcad_pipeline.md)
- [`../notebooks/README.md`](../notebooks/README.md) — executable notebooks.
- [`../notebooks/simple_gridworld/README.md`](../notebooks/simple_gridworld/README.md) — grid-world notebook guide.
- [`../notebooks/ants/README.md`](../notebooks/ants/README.md) — multi-agent notebook guide.
- [`../configs/README.md`](../configs/README.md) — configuration examples.
- [`../GRTs/README.md`](../GRTs/README.md) — explicit local and optional provider-based research.
- [`../GRTs/data3/README.md`](../GRTs/data3/README.md) — research workspace contents.

## Publication source

- [`manuscript/README.md`](manuscript/README.md) — ordered manuscript source, build, and render.
- [`manuscript/SYNTAX.md`](manuscript/SYNTAX.md) — labels, cross-references, citations, variables.
- [`manuscript/preamble.md`](manuscript/preamble.md) — renderer expectations.
- [`manuscript/00_abstract.md`](manuscript/00_abstract.md) — abstract.
- [`manuscript/01_introduction.md`](manuscript/01_introduction.md) — introduction.
- [`manuscript/02_formal_model.md`](manuscript/02_formal_model.md) — formal model.
- [`manuscript/03_deterministic_dynamics.md`](manuscript/03_deterministic_dynamics.md) — deterministic dynamics.
- [`manuscript/04_implementation.md`](manuscript/04_implementation.md) — implementation architecture.
- [`manuscript/05_validation.md`](manuscript/05_validation.md) — validation.
- [`manuscript/06_results.md`](manuscript/06_results.md) — results.
- [`manuscript/07_related_work.md`](manuscript/07_related_work.md) — related work.
- [`manuscript/08_limitations.md`](manuscript/08_limitations.md) — limitations.
- [`manuscript/09_reproducibility.md`](manuscript/09_reproducibility.md) — reproducibility.
- [`manuscript/99_references.md`](manuscript/99_references.md) — references heading.
- [`manuscript/config.yaml`](manuscript/config.yaml) — publication metadata and experiment configuration.
- [`manuscript/references.bib`](manuscript/references.bib) — citation database.
- [`manuscript/figure-registry.json`](manuscript/figure-registry.json) — figure sources and artifact mapping.

## Package guides

- [`../blockference/README.md`](../blockference/README.md) — package layout.
- [`../blockference/config/README.md`](../blockference/config/README.md) — configuration helpers.
- [`../blockference/envs/README.md`](../blockference/envs/README.md) — environments.
- [`../blockference/io/README.md`](../blockference/io/README.md) — persistence.
- [`../blockference/simulations/README.md`](../blockference/simulations/README.md) — backend adapters.
- [`../blockference/utils/README.md`](../blockference/utils/README.md) — utility helpers.
- [`../blockference/viz/README.md`](../blockference/viz/README.md) — visualization.

## InstituteOS metadata and historical review

- [`../.aii/README.md`](../.aii/README.md) — metadata sidecar.
- [`../.aii/docs/README.md`](../.aii/docs/README.md) — sidecar document index.
- [`../.aii/docs/INTEGRATION.md`](../.aii/docs/INTEGRATION.md) — federation and metadata operations.
- [`../REVIEW_LOG_2026-08-02.md`](../REVIEW_LOG_2026-08-02.md) — historical review record, not current gate evidence.
