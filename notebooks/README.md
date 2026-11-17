# Notebooks

Every notebook is a small executable example of the current package API. The
examples cover the model, environment transitions, collision resolution, and
the optional local research workflow.

`blockference.notebook_inventory` owns selection, validation, and execution.
The scripts and slow tests consume one frozen tracked selection from an exact
Git root, including worktrees. Ignored/untracked files, administrative/build
trees, checkpoints, missing tracked files, and symlink escapes cannot enter
execution. Git errors fail rather than trigger a recursive-scan fallback;
a valid empty selection remains empty.

Execution uses the supplied workspace and an independently prepared
`python3` kernel. Source selection alone is not kernel qualification.
The single-agent gridference example owns `np.random.default_rng(17)` and
passes it to planning, independently of the global NumPy stream. Executing
examples does not authorize committing refreshed outputs, metadata, or
scientific claims.
