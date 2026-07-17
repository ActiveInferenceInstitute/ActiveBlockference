# Environment guidance

Environment classes encode world dynamics, not inference. `GridWorld.step`
uses `(y, x)` coordinates, clamped boundaries, configured actions, and
deterministic simultaneous collision resolution. It never prints or plots.
