# `docs/` — ActiveBlockference documentation

Authoritative long-form documentation. The repository's [top-level
`README.md`](../README.md) is the elevator pitch; everything *deeper*
lives here.

## Map

| File                              | Audience                | Purpose                                            |
|-----------------------------------|-------------------------|----------------------------------------------------|
| [`theory.md`](theory.md)          | newcomers, scientists    | Active Inference primer — A, B, C, D, E.           |
| [`architecture.md`](architecture.md) | contributors, reviewers | How the package fits together.                     |
| [`pipeline.md`](pipeline.md)      | library users            | config → simulate → persist → viz → validate flow. |
| [`migration.md`](migration.md)    | upgraders                | pymdp 0.0.x → 1.0.x migration guide.               |
| [`api.md`](api.md)                | library users            | Public Python surface.                             |
| [`glossary.md`](glossary.md)      | everyone                 | One-line definitions for every recurring term.     |
| [`contributing.md`](contributing.md) | contributors            | How to make changes here.                          |
| [`development.md`](development.md)| contributors             | Local dev environment with `uv`, lint, format.     |
| [`tutorials/`](tutorials/)        | new users                | Step-by-step walkthroughs.                         |

## Building these docs

These are plain Markdown — no static-site generator required. To build a
browsable site optionally:

```bash
pip install -e .[docs]
mkdocs serve     # http://localhost:8000
```

The `[docs]` extra ships `mkdocs` + `mkdocs-material`. A future PR may
add `mkdocs.yml` for navigation.

See [`AGENTS.md`](AGENTS.md) for editing rules.
