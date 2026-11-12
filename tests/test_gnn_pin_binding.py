"""Offline tests for the pin-binding checks in ``scripts/validate_gnn_spec.py``.

No test here uses the network or an installed GNN runtime: remote tag resolution
is exercised against a stubbed ``git ls-remote``, the checkout path uses a
throwaway local git repository, and the runtime is a stub.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import scripts.validate_gnn_spec as vgs

COMMIT = "a" * 40
OTHER = "b" * 40


def _pin(**overrides) -> dict:
    pin = {
        "repository": "https://example.invalid/Generalized_Notation_Notation",
        "tag": "v9.9.9",
        "commit": COMMIT,
        "parser_files": {"src/gnn/__init__.py": "0" * 64},
    }
    pin.update(overrides)
    return pin


def _git(cwd: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(cwd), "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


@pytest.fixture
def checkout(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "gnn"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "f").write_text("x", encoding="utf-8")
    _git(repo, "add", "f")
    _git(repo, "commit", "-q", "-m", "c")
    return repo, _git(repo, "rev-parse", "HEAD")


def _stub_ls_remote(monkeypatch: pytest.MonkeyPatch, stdout: str, returncode: int = 0) -> list:
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(list(command))
        return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr="boom")

    monkeypatch.setattr(vgs.subprocess, "run", fake_run)
    return calls


def test_local_annotated_tag_resolves_to_the_peeled_commit(checkout) -> None:
    repo, head = checkout
    _git(repo, "tag", "-a", "-m", "annotated", "v9.9.9")
    pin = _pin(commit=head)
    binding = vgs.tag_binding(repo, pin)

    assert binding["status"] == "verified"
    assert binding["resolved"] == head
    assert vgs.tag_failures(binding, pin) == []


def test_local_tag_pointing_elsewhere_is_a_named_failure(checkout) -> None:
    repo, head = checkout
    _git(repo, "tag", "v9.9.9")
    pin = _pin(commit=OTHER)
    failures = vgs.tag_failures(vgs.tag_binding(repo, pin), pin)

    assert len(failures) == 1
    assert failures[0].startswith(vgs.TAG_COMMIT_MISMATCH)
    assert head in failures[0] and OTHER in failures[0]


def test_offline_without_a_store_is_explicitly_unresolved(
    monkeypatch: pytest.MonkeyPatch, checkout
) -> None:
    repo, _head = checkout  # a checkout that lacks the tag

    def no_network(command, **_kwargs):
        assert "ls-remote" not in command, "offline binding must not use the network"
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="")

    monkeypatch.setattr(vgs.subprocess, "run", no_network)
    for store in (None, repo):
        binding = vgs.tag_binding(store, _pin())
        assert binding["status"] == "unresolvable-offline"
        assert binding["resolved"] is None
        assert vgs.tag_failures(binding, _pin()) == []


def test_remote_prefers_the_peeled_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub_ls_remote(
        monkeypatch, f"{OTHER}\trefs/tags/v9.9.9\n{COMMIT}\trefs/tags/v9.9.9^{{}}\n"
    )
    binding = vgs.tag_binding(None, _pin(), remote=True)

    assert calls[0][:3] == ["git", "ls-remote", "--tags"]
    assert binding["status"] == "verified"
    assert binding["resolved"] == COMMIT
    assert vgs.tag_failures(binding, _pin()) == []


def test_remote_falls_back_when_the_checkout_lacks_the_tag(
    monkeypatch: pytest.MonkeyPatch, checkout
) -> None:
    repo, _head = checkout
    real_run = subprocess.run

    def run(command, **kwargs):
        if command[:2] == ["git", "ls-remote"]:
            return subprocess.CompletedProcess(command, 0, f"{COMMIT}\trefs/tags/v9.9.9\n", "")
        return real_run(command, **kwargs)

    monkeypatch.setattr(vgs.subprocess, "run", run)
    binding = vgs.tag_binding(repo, _pin(), remote=True)

    assert binding["status"] == "verified"
    assert binding["detail"] == "git ls-remote"


def test_remote_tag_resolving_elsewhere_fails_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_ls_remote(monkeypatch, f"{OTHER}\trefs/tags/v9.9.9\n")
    binding = vgs.tag_binding(None, _pin(), remote=True)

    assert [f.split(":")[0] for f in vgs.tag_failures(binding, _pin())] == [vgs.TAG_COMMIT_MISMATCH]


def test_remote_ignores_look_alike_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_ls_remote(monkeypatch, f"{OTHER}\trefs/tags/v9.9.9-rc1\n")
    binding = vgs.tag_binding(None, _pin(), remote=True)

    assert binding["status"] == "unresolvable-offline"
    assert "not listed" in binding["detail"]


def test_failed_remote_is_unresolved_never_a_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_ls_remote(monkeypatch, "", returncode=128)
    binding = vgs.tag_binding(None, _pin(), remote=True)

    assert binding["status"] == "unresolvable-offline"
    assert "ls-remote failed" in binding["detail"]


def test_installed_commit_must_match_the_pin() -> None:
    pin = _pin()

    assert vgs.revision_failures({"status": "verified", "commit": COMMIT}, pin) == []
    assert vgs.revision_failures({"status": "unresolvable-offline", "commit": None}, pin) == []
    assert vgs.revision_failures({"status": "verified", "commit": OTHER}, pin)[0].startswith(
        vgs.TAG_COMMIT_MISMATCH
    )
    assert vgs.revision_failures({"status": "conflicting", "detail": "two"}, pin)[0].startswith(
        vgs.TAG_COMMIT_MISMATCH
    )


def _run_main(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *extra: str):
    probed: list[str] = []

    def fake_probe(*_args, **_kwargs):
        probed.append("probe")
        raise AssertionError("the parser must not run before the digests verify")

    monkeypatch.setattr(vgs, "run_probe", fake_probe)
    monkeypatch.setattr(
        vgs, "read_importer_report", lambda *_a, **_k: {"pin": _pin(), "variables": {}}
    )
    code = vgs.main(
        ["--gnn-runtime", sys.executable, "--receipt", str(tmp_path / "r.json"), *extra]
    )
    return code, probed


def test_digests_are_verified_before_the_document_is_parsed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    package = tmp_path / "gnn"
    package.mkdir()
    (package / "__init__.py").write_text("tampered", encoding="utf-8")
    monkeypatch.setattr(vgs, "locate_package", lambda *_a, **_k: package)

    code, probed = _run_main(monkeypatch, tmp_path)

    assert code == 2
    assert probed == []
    assert "digest mismatch" in capsys.readouterr().err


def test_a_missing_pinned_package_is_not_run_before_parsing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def absent(*_args, **_kwargs):
        raise vgs.Unavailable("the gnn package is not installed")

    monkeypatch.setattr(vgs, "locate_package", absent)
    code, probed = _run_main(monkeypatch, tmp_path)

    assert code == 2
    assert probed == []


def test_verify_tag_remote_that_cannot_answer_is_not_run_not_a_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_ls_remote(monkeypatch, "", returncode=128)
    code, probed = _run_main(monkeypatch, tmp_path, "--verify-tag-remote")

    assert code == 2
    assert probed == []
    assert "--verify-tag-remote" in capsys.readouterr().err


def test_a_remote_tag_moved_off_the_pinned_commit_reaches_the_conformance_failures() -> None:
    binding = {"status": "verified", "tag": "v9.9.9", "commit": COMMIT, "resolved": OTHER}
    payload = {
        "gnn_version": "9.9.9",
        "gnn_module": "/nowhere/gnn/__init__.py",
        "parser": {
            "success": True,
            "errors": [],
            "warnings": [],
            "model_name": "x",
            "variables": [],
            "version": "1.1",
        },
        "semantic": {"is_valid": True},
        "orientation": {},
        "consistency": {"consistent": True},
    }
    failures = vgs.conformance_failures(
        payload, pin=_pin(), source=None, spec_text="", tag=binding
    )

    assert any(f.startswith(vgs.TAG_COMMIT_MISMATCH) for f in failures)


def test_normalized_owner_name_pins_become_github_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    assert vgs.remote_url("Org/Repo") == "https://github.com/Org/Repo"
    assert vgs.remote_url("https://example.invalid/x") == "https://example.invalid/x"
    calls = _stub_ls_remote(monkeypatch, "")
    vgs.tag_binding(None, _pin(repository="Org/Repo"), remote=True)
    assert "https://github.com/Org/Repo" in calls[0]
