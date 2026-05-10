# AGENTS.md — `notebooks/simple_gridworld/`

Same rules as the parent `notebooks/AGENTS.md`, plus:

* These notebooks form a **pedagogical sequence**. Renaming or reordering
  changes the learning path — coordinate via an issue first.
* Imports should prefer `from blockference import ...`. Notebooks
  predating the package layout may still use `sys.path.insert`; flag them
  for migration.
* Coordinate convention: `(y, x)` everywhere, matching the library.
