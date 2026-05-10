# AGENTS.md — `blockference/viz/`

## Hard rules

* Force `matplotlib.use("Agg")` at import time (already done) — never
  switch backends from inside a function.
* Renderers must be **pure** w.r.t. their inputs: take a DataFrame +
  path, write the file, return the path. No global state mutation.
* Always close the figure (`plt.close(fig)`) after saving — long
  cadCAD sweeps will exhaust memory otherwise.
* Coordinate convention is `(y, x)`. Display origin is top-left.

## Soft rules

* Default DPI 120; bump only if the user explicitly asks.
* Prefer `matplotlib.pyplot.get_cmap("tab10")` for per-agent colour.
* When a renderer needs a value (e.g. `n_agents`) that the DataFrame
  doesn't carry, derive it from the data, then optionally accept an
  override kwarg.

## Animation specifics

* Default writer is Pillow (`PillowWriter`) for `.gif`. MP4 needs
  ffmpeg installed at the OS level.
* Keep `interval = 1000 / fps` so changing `fps` actually changes
  playback speed.

## Things to avoid

* Heavy interactive backends (Qt, TkAgg). They block CI.
* Reading from disk inside a renderer — accept a DataFrame.
* Plotting the **true** EFE without it being in the trajectory frame.
  Today we only have the posterior; the EFE-proxy chart uses
  belief-entropy and is labelled honestly as such.
