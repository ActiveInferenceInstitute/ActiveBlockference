# ActiveBlockference TODO

This file is the forward-looking execution source of truth. The prior
hardening and release-contract work is intentionally absent; an item belongs
here only when implementation, tests, documentation, generated artifacts, or
verification still remains.

## P0 — cross-interpreter release verification

- [ ] Run the locked development environment and canonical release gate on
  Python 3.10, 3.11, and 3.12; resolve any interpreter- or platform-specific
  failures without weakening required checks.
- [ ] Run the clean wheel-install smoke path on every supported interpreter,
  including the documented quick start and import without the optional
  `pymdp` extra.
- [ ] Exercise a completed-run reuse attempt and a deliberately interrupted
  run in CI so stale manifests, partial trees, and old validation reports can
  never yield a passing release verdict.

## P1 — architectural hardening

- [ ] Split `validate_run_outputs` into explicit trajectory, model, diagnostic,
  artifact, and manifest stages with focused public helpers while preserving
  `ValidationReport` and fail-closed aggregation.
- [ ] Extend atomic publication to CSV, NPZ, renders, and logs where the
  backend permits it; add interruption tests around each replacement boundary.
- [ ] Formalize the shared typed step/update protocol across single-agent,
  graph, dictionary, radCAD, and cadCAD adapters, then run the type checker on
  Python 3.10-compatible stubs as well as the current interpreter.
- [ ] Add a normalized trajectory schema object so backend parity compares
  typed values rather than stringified DataFrame cells, including all
  diagnostics, policy order, row indices, and configured starts.
- [ ] Optimize grid index lookup and transition construction using precomputed
  coordinate indexes; prove exact matrix and trajectory parity before and
  after the optimization.

## P1 — publication and usability completion

- [ ] Make the release command compare generated files against a clean
  checkout-independent baseline and fail on any new manuscript Markdown,
  figure, or metadata drift.
- [ ] Update every teaching notebook and tutorial to show explicit multi-agent
  starts, seeded runs, backend parity, `ValidationReport`, and inspection of a
  persisted artifact; preserve the clean temporary-workspace gate.
- [ ] Add rendered PDF/HTML manuscript checks when the optional renderer stack
  is installed, while keeping source validation and deterministic regeneration
  network-free and mandatory.

## P2 — scoped long-horizon extensions

- [ ] Define and document a generic discrete-environment protocol covering
  state identity, observations, simultaneous actions, collisions, transition
  stochasticity, and serialization before adding another environment.
- [ ] After that protocol has independent release gates, scope obstacle maps,
  stochastic transitions, learned `A/B/C`, factored state spaces, larger
  policy search, and `AntColonyEnv` as separate validated contracts rather
  than incremental flags on the grid reference model.
- [ ] Keep empirical scientific adequacy claims outside software-integrity
  verdicts; any such claim requires independent datasets, provenance, and
  sensitivity analyses documented separately from `PipelineResult.ok`.

## Exit criteria for the remaining roadmap

- [ ] The canonical release gate passes on all supported Python versions.
- [ ] A clean wheel install runs the documented quick start without `pymdp` and
  the optional adapter fails with its documented installation message.
- [ ] A fresh and an interrupted run both validate fail-closed, with exact
  agreement among JSON, NPZ, CSV, summary, manifest, renders, and report.
- [ ] All tracked notebooks and the manuscript pass their mandatory gates;
  optional renderers pass when present.
- [ ] `ValidationReport.ok` and `PipelineResult.ok` remain the only release
  verdicts as future environments and research workflows are added.
