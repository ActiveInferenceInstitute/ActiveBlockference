# ActiveBlockference TODO

Owner: Active Inference Institute
Last reviewed: 2026-08-02

This file records documentation findings from the 2026-08-02 DOCS-DEEP pass.
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

## Major

No major documentation-system overhaul was identified. The repository already
has a coherent root/docs/manuscript structure, local contributor guidance, and
an executable publication validator.

## Open / deferred

- [ ] Add automated Markdown link and anchor validation to CI. Deferred because
      the repository currently has no Markdown tooling dependency or CI command;
      the pass used a repo-wide audit script without introducing a heavyweight
      documentation toolchain.
- [ ] Add a dedicated `SECURITY.md` policy. Deferred because no security-reporting
      channel or policy owner is defined in the repository metadata; adding one
      would require an organization-level decision rather than inventing contact
      details.

## Verification notes

The pass verified the documented single-agent and multi-agent Python examples,
the `docs/api.md` pipeline example, the tutorial snippets, `uv run ruff check
blockference tests scripts GRTs`, manuscript validation/build drift checks, and
the repository-wide relative Markdown links and anchors. Full release verification
remains the canonical `uv run python scripts/release_check.py` command.
