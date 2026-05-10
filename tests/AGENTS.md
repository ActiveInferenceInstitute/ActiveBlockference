# AGENTS.md — `tests/`

Tests should be small, deterministic, and quick.

## Hard rules

* Mirror the source layout: a test for `blockference/foo/bar.py` goes
  in `tests/foo/test_bar.py`.
* Seed the RNG. The `_seed_rng` fixture in `conftest.py` is autouse;
  don't override it without justification.
* Per-test budget:
  * **Unit tests** — under 2 seconds.
  * **Smoke tests** — under 10 seconds *for the file*.
* No network calls in tests. Period.

## Soft rules

* Prefer `pytest.parametrize` to copy-paste.
* Use `tmp_path` (pytest-builtin) for any file IO.
* Use `pytest.importorskip` for optional / heavy dependencies so the
  suite degrades gracefully on minimal installs.

## Things to avoid

* `time.sleep` — replace with deterministic stubs.
* Asserting against the *string* representation of NumPy arrays — compare
  numerically with `np.allclose` or `pytest.approx`.
* Testing private functions directly when a public path covers them.

## Adding a folder

If you add `blockference/<new>/`, also add `tests/<new>/__init__.py` and
at least one regression test.
