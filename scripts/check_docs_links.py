"""Validate internal Markdown links and anchors across tracked documentation.

For every tracked ``*.md`` file, verify that:

* every relative link resolves to an existing path inside the repository;
* every ``file.md#anchor`` reference names a real GitHub-Flavored-Markdown
  heading slug in the target file; and
* every same-file ``#anchor`` reference matches a heading slug in that file.

External URLs are skipped, and fenced code blocks are ignored so illustrative
snippets cannot cause false failures. The check is deterministic and performs
no network access; it is part of the canonical release gate.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*)$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_IGNORED_PARTS = {".git", ".venv", "docs/_build", ".ipynb_checkpoints"}


def _slugify(heading: str) -> str:
    """Return the GitHub-Flavored-Markdown anchor slug for a heading."""

    slug = heading.strip().lower()
    slug = re.sub(r"[`*_]", "", slug)
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    return re.sub(r"\s+", "-", slug.strip())


def _headings(path: Path) -> set[str]:
    """Return heading slugs for a Markdown file, ignoring fenced blocks."""

    slugs: set[str] = set()
    in_fence = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = _HEADING_RE.match(line)
        if match:
            slugs.add(_slugify(match.group(2)))
    return slugs


def _markdown_files(root: Path) -> list[Path]:
    """Return tracked Markdown files, falling back to a directory scan."""

    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "*.md"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return sorted(Path(name) for name in result.stdout.splitlines() if name)
    return sorted(
        path
        for path in root.rglob("*.md")
        if not any(part in _IGNORED_PARTS for part in path.parts)
    )


def _check_file(path: Path, doc_dir: Path, root: Path) -> list[str]:
    """Return link and anchor issues found in one Markdown file."""

    issues: list[str] = []
    slugs = _headings(path)
    in_fence = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in _LINK_RE.finditer(line):
            target = match.group(1).strip()
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            if target.startswith("#"):
                anchor = target[1:]
                if anchor and anchor not in slugs:
                    issues.append(f"same-file anchor '#{anchor}' not found")
                continue
            part = target.split("#", 1)[0].split("?", 1)[0]
            if not part:
                continue
            resolved = (doc_dir / part).resolve()
            try:
                resolved.relative_to(root)
            except ValueError:
                issues.append(f"link '{target}' resolves outside the repository")
                continue
            if not resolved.exists():
                issues.append(f"link '{target}' target does not exist: {resolved}")
                continue
            if "#" in target and resolved.suffix == ".md":
                anchor = target.split("#", 1)[1].split("?", 1)[0]
                if anchor and anchor not in _headings(resolved):
                    issues.append(f"anchor '#{anchor}' not found in {resolved.name}")
    return issues


def validate(root: str | Path) -> dict[Path, list[str]]:
    """Return ``{relative_path: [issue, ...]}`` for every documentation file."""

    root_path = Path(root).resolve()
    failures: dict[Path, list[str]] = {}
    for entry in _markdown_files(root_path):
        path = entry if entry.is_absolute() else root_path / entry
        relative = entry.relative_to(root_path) if entry.is_absolute() else entry
        if not path.is_file():
            failures.setdefault(relative, []).append("tracked Markdown file is missing")
            continue
        for issue in _check_file(path, path.parent, root_path):
            failures.setdefault(relative, []).append(issue)
    return failures


def main() -> int:
    """Validate repository Markdown and print a concise report."""

    root = Path(__file__).resolve().parents[1]
    failures = validate(root)
    if failures:
        for relative, issues in sorted(failures.items()):
            for issue in issues:
                print(f"ERROR: {relative}: {issue}")
        return 1
    print("markdown link and anchor validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
