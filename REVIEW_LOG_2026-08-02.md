# Documentation deep-review log — 2026-08-02

## Scope

Reviewed all tracked Markdown documentation (70 files), including the root
README and contributor guidance, `docs/`, the ordered manuscript source,
tutorials, package/config/test/notebook/GRTs README files, `.github/` guidance,
and documentation-facing scripts/configuration. Cross-checked public API,
configuration, CLI, pipeline, environment, persistence, visualization,
notebook, and manuscript claims against the implementation.

## Preflight

- Branch: `main`
- Remote default branch: `origin/main`
- Starting HEAD: `1fad3dfb957224afd5e73aa639c377e5b46bd2e2`
- Starting working tree: clean
- `git fetch origin` completed; branch was already current, so no pull was
  required.
- Documentation inventory: 70 tracked Markdown files; `docs/` contains a
  concept/API/pipeline/development set, tutorials, and an ordered manuscript.

## Findings

### Minor — 4 findings

1. `tests/envs/README.md` and `tests/utils/README.md` linked five directory
   levels upward for root files; those targets resolved outside the repository.
2. `configs/AGENTS.md` named a nonexistent `tests/test_config.py`; configuration
   tests are distributed across current `tests/test_*.py` modules.
3. The `scripts/build_manuscript.py` module docstring named
   `docs/_build/manuscript.md`, while the builder writes
   `docs/_build/manuscript/manuscript.md`.
4. The root README did not link its documentation index or license, despite
   both being high-value entry points.

### Medium — 4 findings

1. The root README Python API example configured two agents with only
   `initial_state`, which the current implementation intentionally rejects as
   non-distinct multi-agent starts.
2. `docs/README.md` was only a list and omitted the glossary, notebooks,
   configuration, and explicit beginner entry path.
3. The three tutorials were short fragments despite their local contract saying
   each should be a runnable end-to-end story with prerequisites and next steps.
4. `docs/contributing.md` instructed contributors to fill in a PR template, but
   no PR template exists under `.github/`.

### Major — 0 findings

The repository already has a coherent documentation structure, local guidance,
strict manuscript validation, and a release gate. No responsible major overhaul
was warranted.

## Implemented changes

- Corrected all verified broken/stale references.
- Corrected and executed the README multi-agent example.
- Expanded the documentation index and root entry-point links.
- Rewrote the tutorials around the current package API and CLI.
- Aligned contribution guidance with the repository's actual GitHub files.
- Replaced the TODO roadmap with severity sections, completed entries, and
  explicit deferred items.

## Validation performed

- Executed the README API example after correction.
- Executed the `docs/api.md` pipeline example.
- Executed the tutorial Python examples and checked the pipeline CLI commands
  against the current CLI definition.
- Ran the repo-wide relative Markdown link/anchor audit: no remaining internal
  link or anchor failures after fixes.
- Ran `uv run ruff check blockference tests scripts GRTs`: passed.
- The complete release gate is run after the final documentation commits.
