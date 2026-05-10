# AGENTS.md — Working in the ActiveBlockference repository

This file orients any **AI coding agent** (Claude Code, Cursor, GitHub
Copilot Workspace, OpenAI Codex, etc.) and any new **human contributor**
arriving at this repo cold. It is the canonical source for *how* to make
changes here safely.

## 1. Project at a glance

* **Domain:** Active Inference (AIF), specifically discrete-state generative
  models for cadCAD/radCAD multi-agent simulations.
* **Primary upstream libraries:** `inferactively-pymdp`, `cadCAD`, `radcad`.
* **Python:** 3.9+ (CI target: 3.10).
* **Package name:** `blockference` (importable as `blockference.*`).

## 2. The mental model

When in doubt, remember the canonical AIF loop the package implements:

```
observation o_t
  → infer hidden state q(s_t) using A
  → for each policy π evaluate expected free energy G(π) using A, B, C
  → softmax(-G) gives policy posterior; sample → action a_t
  → propagate prior: prior_{t+1} = B[:,:,a_t] · q(s_t)
```

`A` (likelihood), `B` (transitions), `C` (preferences), `D` (initial prior),
`E` (affordances/actions) are the five matrices you will see everywhere.

## 3. Folder ownership map

| Path                       | Owner concept                                  |
|----------------------------|------------------------------------------------|
| `blockference/`            | Installable library — keep stable, add tests.  |
| `blockference/envs/`       | Environment classes (state space + dynamics).  |
| `blockference/simulations/`| End-to-end runnable cadCAD experiments.        |
| `blockference/utils/`      | Math, plotting, cadCAD policy helpers.         |
| `notebooks/`               | Exploratory / pedagogical — code may be rough. |
| `docs/`                    | Authoritative explanatory docs.                |
| `tests/`                   | Unit + smoke tests; mirror package layout.     |
| `GRTs/`                    | LangChain "Generative Research Team" sketch.   |

Each folder has its own `AGENTS.md` with folder-local rules.

## 4. Before you change anything

1. **Read** `docs/architecture.md` and the relevant folder-local `AGENTS.md`.
2. **Set up the env** with `uv`:
   ```bash
   uv venv --python 3.11
   uv pip install -e ".[dev]"
   ```
   Do not use `pip install` directly; CI relies on the same `uv` lockfile.
3. **Run the tests**: `uv run pytest`. They must be green at HEAD.
4. **Check open issues** — particularly any labelled `breaking-change`.

## 5. Coding conventions

* Format: `ruff format` (line length 100). `ruff check` must pass.
* Imports: absolute (`from blockference.utils import utils`), never relative
  with `..`.
* Docstrings: NumPy style, present tense, one-line summary then blank line.
* Public surface lives in each subpackage's `__init__.py` — keep it tidy.
* No `sys.path.insert` hacks in library code. (Notebooks may still do this
  pragmatically; we are gradually phasing them out.)

## 6. Things to avoid

* **Do not** import `pymdp` symbols and re-export them under a same-name
  shadowing class (`class Agent(Agent)` was the historical bug — see
  `blockference/agent.py` for the corrected pattern).
* **Do not** commit `result.csv`, `__pycache__`, `.ipynb_checkpoints`,
  `.DS_Store`, virtualenvs, or notebook outputs that include unresolved
  merge markers.
* **Do not** widen the public surface without adding a docstring + test.
* **Do not** introduce GPU / heavyweight ML dependencies; this package is
  meant to stay lean and CPU-friendly.

## 7. Dependency policy

* Runtime deps go in `pyproject.toml [project] dependencies` and stay
  mirrored in `requirements.txt`.
* Dev/docs deps go in `[project.optional-dependencies]`.
* Pin loosely (`>=`) — this is research code; we want to track upstream
  pymdp/cadCAD progress.

## 8. Notebook hygiene

* Strip outputs before commit when they exceed ~100 KB.
* Prefer importing from `blockference` over copy-pasting cells.
* New notebooks: put a one-paragraph "what this notebook teaches" markdown
  cell at the top.

## 9. AI-agent-specific notes

If you are an LLM agent operating autonomously here:

* Verify file paths with `ls`/`grep` before editing — the repo has been
  refactored and old paths in stale notebooks may not exist.
* When fixing a notebook merge conflict, take the *most-recent* side
  (typically `HEAD`) and re-validate the JSON before saving.
* Never delete `notebooks/` content without an explicit human request — those
  are reference material.
* Never invent a citation; if a paper is mentioned, verify or omit.

## 10. Where to ask questions

Open an issue at
<https://github.com/ActiveInferenceInstitute/ActiveBlockference/issues> or
join the [Active Inference Institute](https://activeinference.org/) Discord.
