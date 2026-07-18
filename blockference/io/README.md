# Persistence and validation

`RunPaths` defines the output tree. Call `paths.ensure()` before writing, then
use persistence functions; they require the existing tree and never create
missing directories. A non-empty tree is refused unless
`paths.ensure(allow_existing=True)` is explicit. CSV nested values are
JSON/Python-literal compatible and are parsed again by `validate_run_outputs`.

`manifest.json` is the completed stable-artifact boundary. It contains relative
paths, byte sizes, and SHA-256 digests; `validation_report.json` and `run.log`
remain mutable metadata outside that digest set.

`ValidationReport.ok` is the fail-closed verdict. A pipeline writes one
aggregate report combining trajectory, model, per-step, artefact, and rendering
checks.
