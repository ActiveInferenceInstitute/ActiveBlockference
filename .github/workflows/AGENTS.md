# Agent Guide — `library/repos/ActiveBlockference/.github/workflows`

## Purpose

This folder is part of the instituteos tree at `library/repos/ActiveBlockference/.github/workflows`. Read [`README.md`](README.md) for a short human-facing description.

## Conventions

- Prefer editing tracked source and library files. Generated `output/` artifacts are not source, but many release artifacts are tracked; inspect any `output/` diff before committing.
- Logging: `from instituteos.config import get_logger` when touching Python under `src/`.
- Tests mirror `src/instituteos/` under `tests/`; keep zero-mock policy (see root [`AGENTS.md`](../../../../../AGENTS.md)).

## Parent context

- Repository root: [`README.md`](../../../../../README.md)
