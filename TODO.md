# ActiveBlockference TODO

Owner: Active Inference Institute
Last reviewed: 2026-11-14

This file records documentation findings from the 2026-08-02 DOCS-DEEP pass and
the 2026-11-11 completeness pass, plus the subsequent estate alignment.
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
- [x] Reconcile the Python-version claim. `docs/manuscript/00_abstract.md` now
      injects `{{python_requires}}` from the generated manuscript variables,
      which derive it from `pyproject.toml`'s `requires-python = ">=3.11"`, so
      the manuscript and package floor can no longer disagree. The
      `python_version < '3.11'` `tomli` fallback in `requirements.txt` remains a
      data-file pin, not a second supported floor.
- [x] Reconcile the optional `pymdp` extra with its lock. The extra named
      `inferactively-pymdp>=1.0.1` while `uv.lock` resolved 1.0.4; the extra now
      requires `>=1.0.5`, the regenerated lock resolves 1.0.5, and the lock's
      `[package.metadata] requires-dist` carries the new specifier. The adapter
      only subclasses `pymdp.agent.Agent`, which 1.0.5 keeps, so the documented
      adapter surface is unchanged, and no page states a pymdp version.
      (7f8469d, #28)
- [x] Align the documentation hub and flat index with long-lived guides,
      generated receipts, and the manuscript figure registry; keep measured
      values in generated metrics rather than human-maintained prose.
- [x] Document the fail-closed GNN arm, the canonical full parser pin, the
      Python 3.11 test-only/Python 3.12 full-gate CI mapping, and the directly
      pinned uv setup action without claiming use of the Institute composite.
- [x] Document receipt imports, including the failed offline package baseline
      cache miss, without claiming a speedup; distinguish the declared NumPy
      range from the actual lockfile resolution.
- [x] Cut over to the shared GNN pin reader. The pin moved from `v5.0.3` at
      `8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1` — a revision that predates
      `src/gnn/importers.py` — to `v5.1.0` at
      `75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf`, and the interim local reader
      (`blockference.gnn.load_pin` plus its pin-shape tests) is deleted. The
      `gnn` arm now runs `python -m gnn.importers check --repo . --require-pointer
      --workflow .github/workflows/ci.yml --job gnn-conformance` from the pinned
      runtime, and the `gnn_*` manuscript variables come from
      `gnn.importers.importer_variables` through
      `scripts/gnn_importer_report.py`. No copy of the reader is vendored.
      (b621126, #27)

## Major

No major documentation-system overhaul was identified. The repository already
has a coherent root/docs/manuscript structure, local contributor guidance, and
an executable publication validator.

## Open / deferred

The current candidate is not merged or release-qualified. The dated completed
items and verification notes below remain historical, not proof for this
source. Current scope:

- [x] **B01:** retain per-run/per-agent models and policies; local real-backend
  and refreshed-manifest association controls passed.
- [x] **B02:** separate histories, diagnostics, plots, and animation labels by
  run/agent; real render and artist/data controls passed.
- [x] **B03:** reviewed artifact-validation controls passed, including six
  string/float/boolean categorical attacks across actions and observations.
  Refreshed manifests cannot hide these invalid categorical values.
- [x] **B04:** malformed artifacts and missing parsed prerequisites fail
  validation rather than escaping or becoming successful skipped boundaries.
- [x] **B05:** regenerate only through the exporter; actual pinned-source
  default, three-action, and singleton-policy parsing passed. Runtime action
  labels remain distinct from declaration-only numeric policy priors.
- [ ] **B06:** research-maintainer decision on scientific prose/definitions.
  Numerical definitions and the held proposed prose remain unchanged.
- [x] **B07:** one external interpreter-selection contract preserves venv
  launchers and separate NumPy environments without project-Python fallback.
- [ ] **B08:** source and installed-origin/content admission controls plus
  actual pinned parser/importer validation passed in their recorded scopes.
  The workflow now prepares and forwards checksum-verified immutable producer
  resources (`66d491f`, bootstrap `f638d4c`, v5.1.0 manifest `43b33c0`). Fresh
  hosted/full-release acceptance remains required. A strict-origin replacement
  needs producer qualification and a new public immutable identity; the private
  candidate is not adopted and full native-ecosystem qualification is not claimed.
- [x] **B09:** retain actual policy caps and ordered allowed policies.
- [x] **B10:** preserve built-in integer/string identities, registry order,
  and established conflict priority through typed serialization.
- [x] **B11:** normalize extreme finite weights without changing RNG ownership
  or the EFE definition.
- [x] **B12:** reject nonmapping configuration roots before output setup,
  retaining explicit null/default compatibility.
- [x] **B13:** fresh-wheel selection controls and the actual invocation-owned
  wheel build, clean-environment installation and isolated import smoke passed.
- [x] **B14:** metadata-selected confined experiment configuration reaches
  real dimension-aware rendering.
- [x] **B15:** relocated composition preserves and resolves local resources;
  code, citations, external URLs, titles, queries, and fragments are retained.
- [x] **B16:** one tracked notebook inventory feeds validation/execution;
  real prepared-kernel execution passed without source output refresh.
- [x] **B17:** the teaching example owns its seeded generator independently
  of the global NumPy stream.
- [x] **B18:** explicit corpus paths are admitted before workspace/provider
  mutation; offline/literal/default modes retain their existing contracts.
- [x] **BO1:** document independent graph worlds versus shared occupancy
  without introducing graph-world collision resolution.
- [ ] **BO2:** elective optimization/benchmark work requires owner selection,
  measured workloads, and RNG/invalid-input parity; no speedup claimed.
- [x] **BO3:** current documentation link/anchor validation passed after
  replacing retired worker instructions and obsolete GRT descriptions.
- [ ] **BO4:** optional adapter qualification remains elective and unclaimed;
  do not reopen its already-set floor or infer incompatibility.
- [x] **BT01:** retire only the lane's legacy `ignored.csv` with its matching
  ignore rule. Original bytes remain under private owner-managed custody;
  no canonical-original deletion or broader CSV cleanup is implied.

Final quality, full required gates, exact-merge-head proof, hosted CI, review,
and ordinary governed integration remain open. Existing measurements are not
rewritten, scientific/owner approvals are not inferred, and canonical figures,
manuscript outputs, private evidence, and retained storage stay preserved.

## Verification notes

The earlier completeness pass verified the documented single-agent and
multi-agent Python examples, the `docs/api.md` pipeline example, the tutorial
snippets, Ruff, manuscript validation/build drift checks, and repository-wide
relative Markdown links and anchors. These are historical verification notes,
not results from the estate alignment.

The estate documentation edits were reviewed by reasoning only; integrated
verification belongs to the canonical release selection in
[`docs/gate.md`](docs/gate.md), with a configured pinned GNN runtime. The
generated metrics and receipts record actual runs, including failures.

On 2026-11-12 the `docs` arm (`scripts/check_docs_links.py` and
`scripts/gate_metrics.py --check`) and the `gnn` arm both passed at `7f8469d`.
The `gnn` arm ran `python -m gnn.importers check --repo . --require-pointer
--workflow .github/workflows/ci.yml --job gnn-conformance` from a locally
installed pinned runtime and reported `pin: specs/gnn-pin.json -> v5.1.0 @
75c834bd85 (schema gnn-importer-pin/v1)`, `pointer: .github/GNN.md -> ok`, and
`verdict: ok`, followed by real pinned-parser conformance at `gnn_version`
5.1.0. The other arms were not re-run in this pass; their recorded observations
in `docs/_generated/` are unchanged.
