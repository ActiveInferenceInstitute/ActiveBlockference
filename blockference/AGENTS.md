# Package guidance

Library code only. Public symbols need a docstring, an `__all__` entry, and a
test. `maths.py` stays pure NumPy. `gridference.py` owns the canonical movement
mapping. `envs/` owns world transitions, `simulations/` owns cadCAD/radCAD
execution, `io/` owns artefacts and validation, and `viz/` owns headless
rendering.

All modules import silently. Use absolute imports and keep file writes behind
explicit functions or CLI commands.
