# Manuscript contributor instructions

The manuscript is a methods paper for the implementation in this repository.
Claims about behavior must be traceable to source code, tests, generated
figures, or a cited scholarly source.

Use the following conventions:

1. Add narrative files with the next ordered two-digit prefix.
2. Give every section an identifier such as `{#sec:methods}`.
3. Give every display equation an identifier such as `{#eq:belief-update}`.
4. Use `definition`, `proposition`, and `algorithm` blocks for numbered
   formalisms; the builder assigns their numbers by source order.
5. Reference formal objects with `[@eq:label]`, `[@def:label]`, `[@prop:label]`,
   `[@alg:label]`, `[@fig:label]`, `[@tbl:label]`, or `[@sec:label]`.
6. Cite scholarship with `[@BibTeXKey]` and add the key to `references.bib`.
7. Generate figures from code and close every figure on success and failure.
8. Do not hardcode a generated result when it can be derived by the builder.

Required checks are:

```bash
uv run python scripts/validate_manuscript.py
uv run python scripts/build_manuscript.py
uv run python scripts/build_manuscript.py --check
```
