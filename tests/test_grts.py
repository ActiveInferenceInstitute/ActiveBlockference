"""Credential-free tests for the optional research workflow."""

import json

from GRTs.agents import OfflineCorpusProvider, ResearchOrchestrator, ResearchWorkspace, process_csv


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
