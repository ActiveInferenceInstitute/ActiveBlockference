"""Credential-free tests for the optional research workflow."""

import json
from pathlib import Path

import pytest

from GRTs.agents import (
    OfflineCorpusProvider,
    ResearchOrchestrator,
    ResearchWorkspace,
    build_provider,
    process_csv,
)
from GRTs.cli import main


def test_offline_orchestrator_writes_report(tmp_path):
    workspace = ResearchWorkspace(tmp_path)
    workspace.project_description.write_text("Active inference uses beliefs and actions.", encoding="utf-8")
    workspace.requests.write_text("Explain beliefs.", encoding="utf-8")
    report = ResearchOrchestrator(OfflineCorpusProvider(workspace.project_description), workspace).run()
    assert report.is_file()
    assert "Findings" in report.read_text(encoding="utf-8")


def test_csv_analysis_is_deterministic(tmp_path):
    source = tmp_path / "data.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")
    result = json.loads(process_csv(source, "count rows"))
    assert result["rows"] == 1
    assert result["columns"] == ["a", "b"]


def test_openai_provider_fails_clearly_without_key(monkeypatch):
    """The OpenAI provider is network-free to validate: missing key raises up front."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        build_provider("openai")


@pytest.mark.parametrize("fault", ["missing", "directory", "broken-link", "unreadable", "utf8"])
@pytest.mark.parametrize("existing_workspace", [False, True])
def test_explicit_corpus_path_failure_precedes_all_workspace_mutation(
    tmp_path, monkeypatch, fault, existing_workspace,
):
    corpus = tmp_path / "corpus.txt"
    if fault == "directory":
        corpus.mkdir()
    elif fault == "broken-link":
        corpus.symlink_to(tmp_path / "absent.txt")
    elif fault == "utf8":
        corpus.write_bytes(b"\xff")
    elif fault == "unreadable":
        corpus.write_text("private corpus", encoding="utf-8")
        read_text = Path.read_text

        def denied(path, *args, **kwargs):
            if path == corpus:
                raise PermissionError("controlled corpus admission failure")
            return read_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", denied)
    workspace = tmp_path / "workspace"
    sentinels = {}
    if existing_workspace:
        workspace.mkdir()
        for name in ("requests.txt", "research_data.txt", "report.txt"):
            path = workspace / name
            sentinels[path] = f"preserve {name}\n".encode()
            path.write_bytes(sentinels[path])
    with pytest.raises((OSError, UnicodeError)):
        provider = OfflineCorpusProvider(corpus)
        ResearchOrchestrator(provider, ResearchWorkspace(workspace)).run()
    with pytest.raises((OSError, UnicodeError)):
        main(["run", "--workspace", str(workspace), "--corpus", str(corpus)])
    if existing_workspace:
        assert {path: path.read_bytes() for path in sentinels} == sentinels
        assert {path.name for path in workspace.iterdir()} == {path.name for path in sentinels}
    else:
        assert not workspace.exists()


def test_missing_corpus_refuses_before_optional_provider_construction(tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("optional provider constructed before explicit Path admission")

    monkeypatch.setattr("GRTs.agents.OpenAIProvider", forbidden)
    with pytest.raises(FileNotFoundError):
        main([
            "run", "--workspace", str(tmp_path / "workspace"),
            "--provider", "openai", "--corpus", str(tmp_path / "missing.txt"),
        ])
    assert not (tmp_path / "workspace").exists()


def test_readable_corpus_is_admitted_before_workspace_and_keeps_its_snapshot(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("Beliefs guide actions.", encoding="utf-8")
    provider = OfflineCorpusProvider(corpus)
    corpus.write_text("Unrelated later replacement.", encoding="utf-8")
    workspace = ResearchWorkspace(tmp_path / "new-workspace")
    assert not workspace.root.exists()
    report = ResearchOrchestrator(provider, workspace).run()
    text = report.read_text(encoding="utf-8")
    assert "Beliefs guide actions." in text
    assert "Unrelated later replacement." not in text
    assert main([
        "run", "--workspace", str(tmp_path / "cli-workspace"), "--corpus", str(corpus),
    ]) == 0
    assert "Unrelated later replacement." in (
        tmp_path / "cli-workspace" / "report.txt"
    ).read_text(encoding="utf-8")


def test_literal_default_and_existing_filename_string_modes_remain_distinct(tmp_path):
    corpus = tmp_path / "source.txt"
    corpus.write_text("A literal result from an existing file.", encoding="utf-8")
    assert "existing file" in OfflineCorpusProvider(str(corpus)).complete("result")
    assert "Absent filename text" in OfflineCorpusProvider("Absent filename text").complete("text")
    long_text = "Belief " * 1000
    assert "Belief Belief" in OfflineCorpusProvider(long_text).complete("belief")
    assert "Active inference" in OfflineCorpusProvider().complete("active")
    assert "Active inference" not in OfflineCorpusProvider("").complete("active")
