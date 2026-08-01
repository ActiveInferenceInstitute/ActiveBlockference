# ActiveBlockference TODO

Owner: Active Inference Institute
Status: all scoped roadmap items closed (release gate green on this interpreter)
Last reviewed: 2026-08-01

This file is the forward-looking execution source of truth. The prior
hardening, release-contract, and refinement work is intentionally absent; an
item belongs here only when implementation, tests, documentation, generated
artifacts, or verification still remains.

## Completed / Closed (completion pass of 2026-08-01)

The following roadmap items were fully implemented in source, tests, docs,
manuscript, and CI this pass, and are closed.

### Cross-interpreter + clean-install release verification (P0)
- [x] CI matrix already runs the canonical release gate on Python 3.10, 3.11,
      and 3.12; added a mypy `--python-version 3.10` type-check step.
- [x] Added deliberately interrupted / partial-tree and reused-run tests so
      stale manifests, partial trees, and old validation reports can never
      yield a passing release verdict.
- [x] Clean wheel-install smoke (import + quick start without the optional
      `pymdp` extra) is exercised by `scripts/release_check.py` and CI.

### Atomic publication across all artefact types (P1)
- [x] JSON, config, CSV, NPZ, per-step, manifest, and every render (PNG/GIF)
      are now published atomically through a sibling temporary file + rename;
      a leftover temp file fails validation as drift. `run.log` remains
      append-only (a live log cannot be atomic, per the contract).
- [x] Added interruption tests: a failed writer leaves the previous file and
      cleans its temp; a partial tree and a stale temp file both fail closed.

### `validate_run_outputs` stage split (P1)
- [x] Split the monolith into focused public stage validators
      (tree structure, artifact presence, config, trajectory, model, manifest,
      policies, per-step, and config↔trajectory / config↔model agreement) with
      `validate_run_outputs` as a thin fail-closed aggregator.

### Normalized typed trajectory schema (P1)
- [x] Added `blockference.io.parse_trajectory_records` returning typed,
      deterministically ordered `TrajectoryRecord` values; backend (radCAD vs
      cadCAD) parity and persistence round-trips now compare bytes-as-values.

### Formalized step/update protocol + 3.10 type check (P1)
- [x] Added `CoreAgentUpdate` required-keys TypedDict and `CORE_UPDATE_FIELDS` /
      `validate_update_envelope`; all adapters emit the same ten-field envelope
      and are validated fail-closed. Mypy now checks 3.10-compatible stubs in CI.

### Grid index / transition optimization (P1)
- [x] `ActiveGridference` uses a precomputed coordinate→index dict (O(1) lookup)
      instead of repeated `list.index` scans, with a brute-force parity test
      proving exact matrix and coordinate-index agreement.

### Publication drift baseline gate (P1)
- [x] `build_manuscript.py --check` now regenerates figures into a temp dir and
      compares bytes against the committed figures (checkout-independent), in
      addition to the manuscript-text diff in `release_check.py`.

### Generic discrete-environment protocol (P2)
- [x] Added the `blockference.envs.DiscreteEnvironment` runtime-checkable
      protocol (state identity, observations, actions, collisions,
      stochasticity, serialization); `GridWorld` conforms and adds lossless
      `serialize()` / `load()`.

### Scientific-adequacy claims kept out of the software verdict (P2)
- [x] README, `docs/pipeline.md`, `docs/design-contract.md`, and the manuscript
      now state explicitly that `ValidationReport.ok` / `PipelineResult.ok`
      certify software and artefact integrity only, never empirical adequacy.

### Multi-agent start correctness (Minor/Medium hardening)
- [x] A multi-agent run without distinct `simulation.initial_states` now fails
      fast with an explicit error instead of crashing inside a simulation
      backend; README, `docs/api.md`, `docs/pipeline.md`, `docs/migration-v1.md`,
      the manuscript, and the example configs were corrected.
- [x] Added request timeout to the OpenAI GRTs provider (default 60 s).

## Forward-looking (not started)

- [ ] Nothing scoped. Future work (additional environments, stochastic
      transitions, learned `A`/`B`/`C`, factored state spaces, larger policy
      search, network-backed research, and any empirical scientific-adequacy
      claim) must enter as its own validated contract with independent
      provenance and sensitivity analyses, per the design contract.

## Exit criteria (all currently met on this interpreter; CI confirms 3.10/3.11/3.12)

- [x] The canonical release gate passes on all supported Python versions (CI).
- [x] A clean wheel install runs the documented quick start without `pymdp` and
      the optional adapter fails with its documented installation message.
- [x] A fresh and an interrupted run both validate fail-closed, with exact
      agreement among JSON, NPZ, CSV, summary, manifest, renders, and report.
- [x] All tracked notebooks and the manuscript pass their mandatory gates;
      optional renderers pass when present.
- [x] `ValidationReport.ok` and `PipelineResult.ok` remain the only release
      verdicts as future environments and research workflows are added.
