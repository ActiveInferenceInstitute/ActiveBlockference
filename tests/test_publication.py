"""Tests for reproducible publication-asset handling."""

import re
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


def test_generated_svgs_are_byte_stable_across_builds(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    builder.build_figures(first)
    builder.build_figures(second)

    svgs = sorted(first.glob("*.svg"))
    assert svgs
    for svg in svgs:
        assert svg.read_bytes() == (second / svg.name).read_bytes()


def test_generated_pngs_carry_no_toolchain_metadata(tmp_path):
    builder.build_figures(tmp_path)

    pngs = sorted(tmp_path.glob("*.png"))
    assert pngs
    for png in pngs:
        assert b"Matplotlib version" not in png.read_bytes()


def test_generated_svgs_carry_no_toolchain_metadata(tmp_path):
    builder.build_figures(tmp_path)

    svgs = sorted(tmp_path.glob("*.svg"))
    assert svgs
    for svg in svgs:
        text = svg.read_text(encoding="utf-8")
        assert "Matplotlib v" not in text
        assert "<dc:date>1970-01-01T00:00:00</dc:date>" in text


def test_committed_figures_are_structurally_valid():
    svgs = sorted(builder.FIGURE_DIR.glob("*.svg"))
    assert svgs
    for svg in svgs:
        png = svg.with_suffix(".png")
        assert png.is_file(), f"missing raster companion for {svg.name}"
        assert builder.svg_structure_failures(svg) == []
        assert builder.png_structure_failures(png) == []


def test_svg_structure_failures_report_damaged_figures(tmp_path):
    source = sorted(builder.FIGURE_DIR.glob("*.svg"))[0]
    original = source.read_text(encoding="utf-8")
    assert builder.svg_structure_failures(source) == []

    def failures(name: str, text: str) -> list[str]:
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        return builder.svg_structure_failures(path)

    assert "not an SVG document" in failures("blank.svg", "")[0]
    assert "well-formed XML" in failures("truncated.svg", original[: len(original) // 2])[0]
    versioned = re.sub(r"<dc:title>[^<]*</dc:title>", "<dc:title>Matplotlib v9.9.9</dc:title>", original)
    assert "toolchain version" in failures("versioned.svg", versioned)[0]
    untitled = re.sub(r"<dc:title>[^<]*</dc:title>", "", original)
    assert "Dublin Core title" in failures("untitled.svg", untitled)[0]
    text_element = original.replace("</defs>", '</defs><text x="1" y="1">label</text>')
    assert "draw text as paths" in failures("text.svg", text_element)[0]
    empty_box = re.sub(r'viewBox="[^"]*"', 'viewBox="0 0 0 0"', original)
    assert "non-positive viewBox" in failures("empty.svg", empty_box)[0]
    no_box = re.sub(r'viewBox="[^"]*"', "", original)
    assert "no usable viewBox" in failures("nobox.svg", no_box)[0]
