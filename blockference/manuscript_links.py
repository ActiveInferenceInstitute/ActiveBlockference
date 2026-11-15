"""Relocate local Markdown resources while retaining source-document semantics."""

from __future__ import annotations

import os
import re
import string
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

__all__ = ["rebase_local_links"]

REFERENCE_RE = re.compile(r"^ {0,3}\[(?!\^)[^\]]+\]:[ \t]*")
FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


def _unescape(value: str) -> str:
    output = []
    index = 0
    while index < len(value):
        if value[index] == "\\" and index + 1 < len(value) and value[index + 1] in string.punctuation:
            index += 1
        output.append(value[index])
        index += 1
    return "".join(output)


def _destination_span(line: str, start: int) -> tuple[int, int]:
    while start < len(line) and line[start] in " \t":
        start += 1
    if start < len(line) and line[start] == "<":
        end = start + 1
        while end < len(line):
            if line[end] == "\\":
                end += 2
            elif line[end] == ">":
                return start + 1, end
            else:
                end += 1
        return start, start
    end, depth = start, 0
    while end < len(line):
        char = line[end]
        if char == "\\":
            end += 2
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            if depth == 0:
                break
            depth -= 1
        elif char.isspace() and depth == 0:
            break
        end += 1
    return start, end


def _relocate(value: str, source: Path, output_dir: Path, copied: Mapping[Path, Path]) -> str:
    if not value:
        return value
    decoded = _unescape(value)
    parsed = urlsplit(decoded)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return value
    target = (source.parent / unquote(parsed.path)).resolve()
    if target not in copied and not target.exists():
        raise FileNotFoundError(f"missing manuscript resource {value!r} in {source}")
    destination = copied.get(target, target)
    # Explicit copied destinations are publication-owned plans; the builder
    # generates/validates registered figures and copies resources before success.
    relative = Path(os.path.relpath(destination, output_dir)).as_posix()
    # Preserve the original query/fragment bytes; only the path changes location.
    suffix = decoded[len(parsed.path):]
    return quote(relative, safe="/-._~") + suffix


def rebase_local_links(
    text: str, *, source: Path, output_dir: Path, copied: Mapping[Path, Path] | None = None,
) -> str:
    """Rebase inline/image and reference-definition destinations, not code or URIs.

    Destination parsing retains balanced/escaped parentheses, angle delimiters,
    optional titles and query/fragment suffixes. Fenced and inline code, scholarly
    citations and same-document fragments are left intact.
    """
    assets = {path.resolve(): destination for path, destination in (copied or {}).items()}
    lines = []
    fence: str | None = None
    fence_length = 0
    for original in text.splitlines(keepends=True):
        marker = FENCE_RE.match(original)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence, fence_length = token[0], len(token)
            elif token[0] == fence and len(token) >= fence_length:
                fence = None
            lines.append(original)
            continue
        if fence is not None:
            lines.append(original)
            continue
        line = original
        definition = REFERENCE_RE.match(line)
        if definition:
            start, end = _destination_span(line, definition.end())
            line = line[:start] + _relocate(line[start:end], source, output_dir, assets) + line[end:]
            lines.append(line)
            continue
        index = 0
        while index < len(line):
            if line[index] == "\\":
                index += 2
                continue
            if line[index] == "`":
                end = index
                while end < len(line) and line[end] == "`":
                    end += 1
                closing = line.find(line[index:end], end)
                index = len(line) if closing < 0 else closing + end - index
                continue
            if line[index] != "[":
                index += 1
                continue
            end, depth = index + 1, 1
            while end < len(line) and depth:
                if line[end] == "\\":
                    end += 2
                    continue
                if line[end] == "[":
                    depth += 1
                elif line[end] == "]":
                    depth -= 1
                end += 1
            if depth or end >= len(line) or line[end] != "(":
                index = end
                continue
            start, finish = _destination_span(line, end + 1)
            replacement = _relocate(line[start:finish], source, output_dir, assets)
            line = line[:start] + replacement + line[finish:]
            index = start + len(replacement)
        lines.append(line)
    return "".join(lines)
