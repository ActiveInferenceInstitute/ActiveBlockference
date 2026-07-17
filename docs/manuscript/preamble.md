# Rendering preamble

The manuscript source is Markdown with Pandoc-compatible mathematics,
citations, section identifiers, figure identifiers, table identifiers, and
formalism blocks. The local builder resolves identifiers before a renderer is
invoked, so the result remains readable in plain Markdown and portable across
HTML, PDF, and repository viewers.

For a PDF build on a machine with Pandoc and XeLaTeX installed:

```bash
uv run python scripts/build_manuscript.py
pandoc docs/_build/manuscript/manuscript.md \
  --from markdown+tex_math_dollars \
  --to pdf --pdf-engine=xelatex \
  --resource-path=docs/_build/manuscript \
  --bibliography=docs/manuscript/references.bib \
  --metadata-file=docs/manuscript/config.yaml \
  -o docs/_build/manuscript/activeblockference.pdf
```

The generated PDF is a build product. The ordered source, bibliography,
configuration, figure generator, and validation commands are the publication
record kept in the repository.
