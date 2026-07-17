# I/O guidance

`RunPaths` is frozen. Call `ensure()` before persistence; each `persist_*`
function then requires the complete tree and never creates missing directories.
Every failed check appends a human-readable issue. Parquet is the only optional
artefact and may be absent when its writer is not installed.
