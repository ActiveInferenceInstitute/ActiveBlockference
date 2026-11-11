# ActiveBlockference TODO

Owner: Active Inference Institute
Last reviewed: 2026-11-11

This file records documentation findings from the 2026-08-02 DOCS-DEEP pass and
the 2026-11-11 completeness pass.
Items are grouped by the size of the documentation change required:

* **Minor** — typo, broken link, stale path, or formatting correction.
* **Medium** — stale section rewrite, documentation restructure, or missing guide.
* **Major** — large documentation-system overhaul or cross-cutting refactor.

## Minor

- [x] Fix four broken repository-root links in `tests/envs/README.md` and
      `tests/utils/README.md`. (e67253c)
- [x] Replace the nonexistent `tests/test_config.py` reference in
      `configs/AGENTS.md` with the current test-file convention. (e67253c)
- [x] Correct the generated-manuscript path in the `scripts/build_manuscript.py`
      module documentation. (e67253c)
- [x] Link the root README to the documentation index and MIT license.
      (e67253c)

## Medium

- [x] Correct the root README Python example so its two-agent configuration
      supplies distinct `simulation.initial_states` and executes under the
      current validation contract. (e67253c)
- [x] Expand `docs/README.md` into a usable documentation index covering
      concepts, tutorials, development, configuration, notebooks, and the
      manuscript source. (55055c4)
- [x] Rewrite the three tutorials as short runnable stories with prerequisites,
      current API/CLI examples, assertions or validation steps, and next steps.
      (55055c4)
- [x] Remove the nonexistent PR-template claim from `docs/contributing.md` and
      describe the actual pull-request information expected. (e67253c)
- [x] Add a stdlib-only Markdown link/anchor validator
      (`scripts/check_docs_links.py`), a negative-controls test, and a release
      gate step so documentation drift fails the canonical gate. (0c16ab8)
- [x] Add `SECURITY.md` with a private-advisory disclosure path and factual
      security-relevant project behaviour. (96a02e3)
- [ ] Reconcile the Python-version claim. `docs/manuscript/00_abstract.md`
      describes a "Python 3.10+" implementation and
      `requirements.txt`/`pyproject.toml` keep a
      `python_version < '3.11'` `tomli` fallback, but `pyproject.toml` declares
      `requires-python = ">=3.11"`. The manuscript is publication source, so it
      was recorded rather than edited here; decide whether to raise the
      manuscript claim to 3.11+ or lower the package floor to 3.10.

## Major

No major documentation-system overhaul was identified. The repository already
has a coherent root/docs/manuscript structure, local contributor guidance, and
an executable publication validator.

## Open / deferred

- The Python-version claim under **Medium** is open pending an owner decision on
  which side to change. All 2026-08-02 docs-deep items remain complete.

## Verification notes

The pass verified the documented single-agent and multi-agent Python examples,
the `docs/api.md` pipeline example, the tutorial snippets, `uv run ruff check
blockference tests scripts GRTs`, manuscript validation/build drift checks, and
the repository-wide relative Markdown links and anchors. Full release verification
remains the canonical `uv run python scripts/release_check.py` command.
