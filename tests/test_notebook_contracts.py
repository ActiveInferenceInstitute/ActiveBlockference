"""Tracked-only notebook admission and notebook-owned RNG behavior."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import nbformat
import numpy as np
import pytest

from blockference.notebook_inventory import (
    execute_notebooks,
    tracked_notebooks,
    validate_all,
    validate_notebook,
)


def _git(root, *arguments):
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update({
        "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0",
    })
    return subprocess.run(
        ["git", "--no-optional-locks", "-c", "core.fsmonitor=false", "-c", "commit.gpgsign=false",
         "-c", "user.name=Notebook Test", "-c", "user.email=notebook@example.invalid",
         "-C", str(root), *arguments],
        check=True, capture_output=True, text=True, env=environment, stdin=subprocess.DEVNULL,
    )


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "core.fsmonitor", "false")
    hooks = root / ".git" / "empty-hooks"
    hooks.mkdir()
    _git(root, "config", "core.hooksPath", str(hooks))
    return root


def _notebook(path, label):
    path.parent.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell(
            "This local teaching example introduces the supported package API with a controlled "
            "environment. It uses deterministic local state without external services."
        ),
        nbformat.v4.new_code_cell(
            "from blockference import GridWorld\n"
            "assert GridWorld(2, {0: (0, 0)}).current_state == {0: (0, 0)}\n"
        ),
    ])
    notebook.metadata["selection_label"] = label
    nbformat.write(notebook, path)


def test_only_tracked_nonignored_source_can_validate_and_reach_execution(repository, monkeypatch, tmp_path):
    first = repository / "course.ipynb"
    unusual = repository / "nb α\ncopy.ipynb"
    _notebook(first, "course")
    _notebook(unusual, "unicode-newline")
    _git(repository, "add", first.name, unusual.name)
    (repository / ".gitignore").write_text("ignored.ipynb\n", encoding="utf-8")
    ignored = repository / "ignored.ipynb"
    ignored.write_bytes(b"invalid notebook must not be parsed")
    _git(repository, "add", "-f", ignored.name)
    for directory in (".venv", "venv", "env", "cad", ".ipynb_checkpoints", "build", "dist", "output", "results"):
        path = repository / directory / "decoy.ipynb"
        path.parent.mkdir()
        path.write_bytes(b"tracked non-source decoy must not be parsed")
        _git(repository, "add", "-f", str(path.relative_to(repository)))
    (repository / "untracked.ipynb").write_bytes(b"untracked invalid notebook")
    (repository / ".git" / "decoy.ipynb").write_bytes(b"git administrative decoy")
    selected, failures = validate_all(repository)
    assert selected == (first, unusual)
    assert failures == {}
    executed = []

    class Client:
        def __init__(self, notebook, **_kwargs):
            self.notebook = notebook

        def execute(self):
            executed.append(self.notebook.metadata["selection_label"])
            return self.notebook

    monkeypatch.setattr("nbclient.NotebookClient", Client)
    assert execute_notebooks(repository, workspace=tmp_path) == (first, unusual)
    assert executed == ["course", "unicode-newline"]


def test_empty_tracked_inventory_stays_empty_even_with_invalid_disk_decoys(repository, tmp_path, monkeypatch):
    (repository / "untracked.ipynb").write_bytes(b"must not be parsed")
    assert tracked_notebooks(repository) == ()
    assert validate_all(repository) == ((), {})
    monkeypatch.setitem(sys.modules, "nbclient", None)
    assert execute_notebooks(repository, workspace=tmp_path) == ()


def test_worktree_git_file_is_a_valid_inventory_root(repository, tmp_path):
    source = repository / "tracked.ipynb"
    _notebook(source, "worktree")
    _git(repository, "add", source.name)
    _git(repository, "commit", "-qm", "fixture")
    worktree = tmp_path / "linked-worktree"
    _git(repository, "worktree", "add", "--detach", str(worktree), "HEAD")
    assert (worktree / ".git").is_file()
    assert tracked_notebooks(worktree) == (worktree / source.name,)


def test_nested_directory_cannot_inherit_an_unrelated_repository_inventory(repository):
    nested = repository / "nested"
    nested.mkdir()
    _notebook(nested / "untracked.ipynb", "not-a-root")
    with pytest.raises(ValueError, match="exact Git repository root"):
        tracked_notebooks(nested)


def test_missing_tracked_notebook_and_symlink_escape_are_explicit_errors(repository, tmp_path):
    source = repository / "tracked.ipynb"
    _notebook(source, "tracked")
    _git(repository, "add", source.name)
    source.unlink()
    with pytest.raises(FileNotFoundError):
        tracked_notebooks(repository)
    outside = tmp_path / "outside.ipynb"
    _notebook(outside, "outside")
    source.symlink_to(outside)
    _git(repository, "add", source.name)
    with pytest.raises(ValueError, match="symlink|escapes"):
        tracked_notebooks(repository)


def test_git_unavailable_or_failed_inventory_never_falls_back_to_recursive_scan(repository, monkeypatch):
    (repository / "untracked.ipynb").write_bytes(b"not source")
    with monkeypatch.context() as context:
        context.setenv("PATH", "")
        with pytest.raises(FileNotFoundError):
            tracked_notebooks(repository)
    (repository / ".git" / "index").write_bytes(b"broken owned fixture index")
    with pytest.raises(RuntimeError, match="cannot read tracked notebooks"):
        tracked_notebooks(repository)


def test_selected_hygiene_failure_prevents_any_notebook_execution(repository, tmp_path, monkeypatch):
    source = repository / "tracked.ipynb"
    _notebook(source, "invalid-summary")
    notebook = nbformat.read(source, as_version=4)
    notebook.cells[0].source = "Too short."
    nbformat.write(notebook, source)
    _git(repository, "add", source.name)
    assert validate_notebook(source)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("execution began despite selected-source hygiene failure")

    monkeypatch.setattr("nbclient.NotebookClient", forbidden)
    with pytest.raises(ValueError, match="hygiene violations"):
        execute_notebooks(repository, workspace=tmp_path)


@pytest.mark.slow
def test_single_agent_notebook_source_owns_a_seeded_stream_independent_of_global_rng():
    path = Path(__file__).resolve().parents[1] / "notebooks" / "simple_gridworld" / "gridference_single.ipynb"
    notebook = nbformat.read(path, as_version=4)
    source = next(cell.source for cell in notebook.cells if cell.cell_type == "code")
    original = np.random.get_state()
    updates = []
    try:
        for seed in (13, 971):
            np.random.seed(seed)
            np.random.random(37)
            before = np.random.get_state()
            namespace = {}
            exec(compile(source, str(path), "exec"), namespace)
            updates.append(namespace["update"])
            after = np.random.get_state()
            assert before[0] == after[0]
            assert np.array_equal(before[1], after[1])
            assert before[2:] == after[2:]
        assert updates[0]["update_action"] == updates[1]["update_action"]
        assert updates[0]["update_env"] == updates[1]["update_env"]
        for name in ("update_prior", "update_inference", "update_efe", "update_q_pi", "update_p_u"):
            assert np.array_equal(updates[0][name], updates[1][name])
    finally:
        np.random.set_state(original)
