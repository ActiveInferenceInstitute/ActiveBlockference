# AGENTS.md — `notebooks/`

Notebooks are *exploration*. They have looser quality bars than library
code, but a few rules keep them merge-friendly.

## Hard rules

1. **One title-and-summary markdown cell at the top.** The cell explains
   what the notebook teaches in 2–4 sentences.
2. **Strip large outputs** (`>100 KB`) before commit. `nbstripout` is the
   easiest tool. Diagrams that the README links to may stay.
3. **No unresolved merge markers.** A previous merge left
   `multi_agent_experimental.ipynb` invalid; resolve before commit.
4. **No notebook depends on another notebook's outputs.** If shared logic
   is needed, promote it into `blockference/`.

## Soft rules

* Keep cells focused: one concept per cell.
* If you call into `blockference`, prefer top-level imports
  (`from blockference import ActiveGridference`).
* If you must add `sys.path.insert(0, '../../')`, leave a `# TODO: remove
  once notebook is migrated to package import` comment.

## Promotion path

1. A useful function in a notebook → move it into `blockference/utils/` or
   `blockference/envs/`.
2. Add a docstring + a unit test.
3. Replace the notebook code with `from blockference... import`.
4. Re-run the notebook end-to-end before commit.

## Things to avoid

* Committing `.ipynb_checkpoints/` (covered by `.gitignore`).
* Hard-coded absolute paths.
* Notebooks named `Untitled*.ipynb` for more than one commit cycle.
