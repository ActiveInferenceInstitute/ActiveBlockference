# Visualisation guidance

Renderers use the Agg backend, validate trajectories and output formats, return
written paths, and close figures in `finally` blocks. `plot_efe` consumes
persisted expected-free-energy vectors. Animation requires a positive integer
FPS and `.gif` or `.mp4` output.
