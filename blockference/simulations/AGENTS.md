# Simulation guidance

Simulation modules import silently and expose typed `run_*` functions returning
DataFrames. radCAD and cadCAD consume the same state-update blocks and seeded
NumPy/Python random streams. The package CLI is the only command-line surface.
