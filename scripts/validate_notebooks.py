"""Validate tracked notebook hygiene before executing notebooks."""

from __future__ import annotations

from pathlib import Path

from blockference.notebook_inventory import validate_all


def main() -> int:
    """Validate repository notebooks and print a concise report."""

    root = Path(__file__).resolve().parents[1]
    notebooks, failures = validate_all(root)
    if failures:
        for path, errors in failures.items():
            for error in errors:
                print(f"ERROR: {path}: {error}")
        return 1
    print(f"notebook hygiene validation passed: {len(notebooks)} tracked notebooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
