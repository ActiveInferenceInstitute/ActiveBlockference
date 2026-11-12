# Manuscript syntax

## Sections and formal objects

```markdown
# Methods {#sec:methods}

::: definition {#def:state-space}
**Discrete state space.** A finite set of states is ...
:::

$$
q_{t+1} = B^{(u_t)}q_t
$$ {#eq:state-propagation}
```

Supported formal labels are `eq`, `def`, `prop`, and `alg`. The builder emits
their type and number in source order and resolves a reference such as
`[@eq:state-propagation]` into the corresponding numbered object.

## Figures and tables

```markdown
![A deterministic transition diagram.](figures/fig1_active_inference_loop.svg){#fig:loop width=90%}

| Field | Meaning |
|---|---|
| `A` | likelihood |

: Core generative-model fields. {#tbl:model-fields}
```

Figure and table numbers are also assigned by source order. The image path is
relative to this directory and must exist when validation runs.

## Variables

Write `{{token}}` where a value must come from the generated variables file
(`docs/_generated/manuscript-variables.json`), for example
`Python {{python_requires}}` or the pinned GNN commit. The file is produced by
`scripts/manuscript_variables.py` from `pyproject.toml`, `specs/gnn-pin.json`,
and `.github/workflows/ci.yml`. Do not type a pin, job name, or supported Python
version into prose; add a key to the generator instead. The builder fails on an
undeclared token, and validation reports tokens the variables file does not
declare.

## Citations

Use BibTeX keys in square brackets:

```markdown
The discrete formulation is summarized by [@Sajid2021].
```

Do not write raw citation commands or raw equation cross-reference commands in
the source. `references.bib` is the only bibliography source.
