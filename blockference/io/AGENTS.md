# AGENTS.md — `blockference/io/`

## Hard rules

* The `RunPaths` dataclass is **frozen**. Add new artefacts as
  properties; never mutate after construction.
* `persist_*` functions take an existing `RunPaths` (already
  ensured) — they must not silently create root directories elsewhere.
* `ValidationReport.ok` is the single source of truth for "did this run
  succeed?". Every check that flips `ok=False` must also append to
  `issues` with a human-readable message.

## Soft rules

* When adding a new artefact (e.g. `trajectory.feather`), update both
  `layout.py` (path) and `validate.py` (existence check).
* Persistence helpers should silently no-op on missing optional
  dependencies (e.g. parquet writer absent) — the run must not fail
  because the user hasn't installed pyarrow.

## Things to avoid

* Pulling cadCAD/radCAD types into validation logic. The `io` layer is
  framework-agnostic — it works on any DataFrame matching the schema.
* Adding configuration here. Configs live in `blockference/config/`.
