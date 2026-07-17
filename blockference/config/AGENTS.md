# Configuration guidance

The schema is strict. Every field is validated in `__post_init__`, every
nested mapping rejects unknown keys, and YAML/TOML round-trips preserve the
same semantic values. Schema changes require a test and an update to
`configs/example.yml` and `configs/README.md`.
