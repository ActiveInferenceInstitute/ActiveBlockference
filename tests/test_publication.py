"""Tests for reproducible publication-asset handling."""

from pathlib import Path

import pytest

import scripts.build_manuscript as builder
import scripts.check_docs_links as docs_links
import scripts.validate_manuscript as validator
from scripts.build_manuscript import _normalise_svg


def test_svg_normalisation_removes_build_time_and_hash_variation(tmp_path):
    path = tmp_path / "figure.svg"
    path.write_text(
        "<dc:date>2026-07-17T14:25:06.833820</dc:date>\n"
        '<path id="p4f37ad28c3" clip-path="url(#p4f37ad28c3)"/>\n',
        encoding="utf-8",
    )
    _normalise_svg(path)
    first = path.read_text(encoding="utf-8")
    _normalise_svg(path)
    second = path.read_text(encoding="utf-8")

    assert first == second
    assert "1970-01-01T00:00:00" in first
    assert "generated_1" in first
    assert "p4f37ad28c3" not in first


def test_manuscript_builder_rejects_duplicate_and_missing_labels():
    with pytest.raises(ValueError, match="duplicate"):
        builder._collect_labels(
            [(Path("01_a.md"), "#sec:duplicate"), (Path("02_b.md"), "#sec:duplicate")]
        )
    with pytest.raises(ValueError, match="unresolved"):
        builder._resolve_references("See [@fig:missing].", {})


def test_manuscript_validator_reports_negative_controls(monkeypatch, tmp_path):
    for index in range(1, 10):
        (tmp_path / f"{index:02d}_section.md").write_text(
            "#sec:duplicate\n\n[@MissingCitation] ![figure](figures/missing.svg) {{TOKEN}} \\ref{bad}\n",
            encoding="utf-8",
        )
    (tmp_path / "references.bib").write_text("", encoding="utf-8")
    monkeypatch.setattr(validator, "MANUSCRIPT", tmp_path)
    errors = validator.validate()
    assert any("labels" in error for error in errors)
    assert any("bibliography" in error for error in errors)
    assert any("figure" in error for error in errors)
    assert any("token" in error for error in errors)
    assert any("LaTeX" in error for error in errors)


def test_docs_link_checker_reports_broken_links_and_anchors(tmp_path):
    (tmp_path / "a.md").write_text(
        "# Heading\n\n"
        "[ok](#heading)\n"
        "[missing](nope.md)\n"
        "[anchor](b.md#nope)\n"
        "[escape](../../outside.md)\n"
        "```\nfenced [fake](fake.md)\n```\n"
        "[external](https://example.com/page)\n",
        encoding="utf-8",
    )
    (tmp_path / "b.md").write_text("# Title\n", encoding="utf-8")
    failures = docs_links.validate(tmp_path)
    messages = [message for issues in failures.values() for message in issues]
    assert any("nope.md" in message for message in messages)
    assert any("#nope" in message and "b.md" in message for message in messages)
    assert any("outside" in message for message in messages)
    assert not any("fake.md" in message for message in messages)
    assert not any("example.com" in message for message in messages)
    assert not any("same-file anchor" in message for message in messages)
