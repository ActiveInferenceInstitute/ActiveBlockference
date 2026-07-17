# Persistence and validation

`RunPaths` defines the output tree. Call `paths.ensure()` before writing, then
use persistence functions; they require the existing tree and never create
missing directories. CSV nested values are JSON/Python-literal compatible and
are parsed again by `validate_run_outputs`.

`ValidationReport.ok` is the fail-closed verdict. A pipeline writes one
aggregate report combining trajectory, model, per-step, artefact, and rendering
checks.
