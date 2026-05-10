# `blockference/io/` — output layout, persistence, validation

Three concerns share this folder:

| Module          | Purpose                                                           |
|-----------------|-------------------------------------------------------------------|
| `layout.py`     | `RunPaths` dataclass — typed accessors for the canonical tree.    |
| `persist.py`    | `persist_config`, `persist_dataframe`, `persist_summary`.         |
| `validate.py`   | `ValidationReport`, `validate_run_outputs`, schema checks.        |

## Canonical layout

Every experiment produces:

```
output/<run_name>/
├── config.yml
├── data/{trajectory.csv, trajectory.parquet?, summary.json}
├── viz/<plots>.png
├── animations/trajectory.gif
└── validation_report.json
```

## Surface

```python
from blockference.io import (
    RunPaths,                          # frozen dataclass with typed paths
    build_run_paths,                   # factory; auto-creates dirs
    persist_config, persist_dataframe, persist_summary,
    ValidationReport, validate_run_outputs, validate_trajectory_dataframe,
)
```

See [`AGENTS.md`](AGENTS.md) for editing rules.
