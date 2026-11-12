"""Validate the committed GNN specification with the pinned upstream parser.

The canonical GNN pin is recorded in ``specs/gnn-pin.json`` (repository, tag,
commit, and SHA-256 digests of the parser files). The pin is read by the
upstream ``gnn.importers`` reader through ``scripts/gnn_importer_report.py``, so
this repository carries no copy of the pin shape. This script supports two
verified sources for that pin:

* a GNN git checkout (``--gnn-root`` or ``GNN_ROOT``): the pinned commit is
  materialized with ``git archive`` and every recorded digest is verified, so
  the live working tree is never imported;
* an installed pinned runtime (``--gnn-runtime`` or ``GNN_RUNTIME_PYTHON``):
  an interpreter whose ``gnn`` package is installed at the pinned revision; the
  installed parser file digests are verified *before* the document is parsed
  (the package is located with the interpreter's import machinery, never
  imported to parse), the VCS commit the installer recorded
  (``direct_url.json``) is compared with the pin, and the parser must be imported
  from the verified package.

The pin's ``tag`` and ``commit`` name one revision. A GNN checkout resolves
``refs/tags/<tag>^{commit}`` locally; ``--verify-tag-remote`` resolves it upstream
with ``git ls-remote`` (the peeled commit of an annotated tag), and always asks
upstream: a tag that moved after a checkout fetched it fails as
``tag-commit-mismatch`` instead of passing on the stale local tag. A tag that
resolves elsewhere fails the same way. Without a checkout or the
flag the binding is recorded in the receipt as ``unresolvable-offline``, never
as a pass, and no network is used. With the flag, a tag that cannot be resolved
ends the run as *not run* (exit ``2``).

Either way the document is parsed by ``scripts/gnn_parser_probe.py`` in a
subprocess and checked for conformance: the declared fixture parameters, the
declared ``A``/``B``/``C``/``D``/``E`` matrices against the runtime model built
by ``blockference.gnn``, the pinned semantic validator, the B-tensor
orientation diagnostic, and the consistency checker.

Exit codes: ``0`` pass, ``1`` conformance failure, ``2`` pinned runtime,
checkout, or requested remote tag resolution unavailable (the run is then
explicitly *not run*, never a pass).

Usage:

    # Local checkout (strongest: archive of the pinned commit)
    GNN_ROOT=/path/to/Generalized_Notation_Notation \\
    GNN_PYTHON=/path/to/parser-runtime/bin/python \\
    uv run python scripts/validate_gnn_spec.py --json

    # Pinned installed runtime (used by CI)
    uv run python scripts/validate_gnn_spec.py --gnn-runtime /path/to/venv/bin/python \\
        --verify-tag-remote
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from blockference.gnn import (
    GNN_SPEC_PATH,
    declared_arrays,
    parameters_from_parsed,
    render_spec,
    spec_arrays,
    spec_parameters,
)

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "gnn_parser_probe.py"
REPORT = ROOT / "scripts" / "gnn_importer_report.py"
DEFAULT_RECEIPT = ROOT / "docs" / "_build" / "gnn" / "validation-receipt.json"
PACKAGE_PREFIX = "src/gnn/"

#: Named failure: the pin's tag and commit are two spellings of one revision.
TAG_COMMIT_MISMATCH = "tag-commit-mismatch"

# Locating the installed package and reading its installer metadata never parses
# a document, so the digest and revision checks run before the probe.
LOCATE_EXPRESSION = (
    "import importlib.util, json; "
    "spec = importlib.util.find_spec('gnn'); "
    "print(json.dumps({'module': spec.origin if spec else None, "
    "'locations': [str(path) for path in (spec.submodule_search_locations or [])] "
    "if spec else []}))"
)
REVISION_PROGRAM = """
import importlib.metadata as md
import json

records = []
for name in md.packages_distributions().get("gnn", []):
    try:
        distribution = md.distribution(name)
    except md.PackageNotFoundError:
        continue
    record = {"name": name, "version": distribution.version, "commit": None}
    raw = distribution.read_text("direct_url.json")
    if raw:
        try:
            direct = json.loads(raw)
        except ValueError:
            direct = {}
        record["commit"] = (direct.get("vcs_info") or {}).get("commit_id")
    records.append(record)
print(json.dumps(records))
"""


class Unavailable(RuntimeError):
    """Raised when the pinned parser runtime or checkout is not available."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_importer_report(python: str, *, source: Path | None = None) -> dict[str, Any]:
    """Return the shared ``gnn.importers`` reading of this repository's pin.

    ``source`` puts a GNN source tree first on ``PYTHONPATH``, which is how a
    checkout supplies the reader when no installed distribution is present.
    """

    environment = dict(os.environ)
    if source is not None:
        existing = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = str(source) + (
            os.pathsep + existing if existing else ""
        )
    completed = subprocess.run(
        [python, str(REPORT), "--repo", str(ROOT)],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        tail = " | ".join(detail[-3:]) if detail else "no output"
        raise Unavailable(
            f"the shared gnn.importers reader could not be run with {python}: {tail}. "
            "Supply --gnn-runtime/GNN_RUNTIME_PYTHON, or --python naming an interpreter "
            "with the pinned GNN distribution, or a GNN checkout via GNN_ROOT."
        )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict) or not isinstance(payload.get("pin"), dict):
        raise Unavailable(f"the shared gnn.importers reader returned no pin: {payload!r}")
    return payload


def _verify_digests(package_dir: Path, digests: dict[str, str], *, origin: str) -> dict[str, str]:
    """Verify every recorded parser digest under an installed ``gnn`` package."""

    observed: dict[str, str] = {}
    for relative, expected in sorted(digests.items()):
        suffix = relative[len(PACKAGE_PREFIX) :] if relative.startswith(PACKAGE_PREFIX) else relative
        path = package_dir / suffix
        if not path.is_file():
            raise Unavailable(f"pinned parser file is missing from {origin}: {suffix}")
        observed[relative] = _sha256(path)
        if observed[relative] != expected:
            raise Unavailable(
                f"pinned parser file digest mismatch for {suffix} in {origin}: "
                f"expected {expected}, observed {observed[relative]}"
            )
    return observed


def _run_text(command: list[str], *, what: str) -> str:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise Unavailable(f"{what}: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        tail = " | ".join(detail[-3:]) if detail else "no output"
        raise Unavailable(f"{what}: {tail}")
    return completed.stdout


def locate_package(python: str) -> Path:
    """Return the ``gnn`` package directory ``python`` would import, without importing it."""

    output = _run_text(
        [python, "-c", LOCATE_EXPRESSION], what=f"cannot locate the gnn package in {python}"
    )
    try:
        located = json.loads(output)
    except json.JSONDecodeError as exc:
        raise Unavailable(f"cannot locate the gnn package in {python}: {exc}") from exc
    if not located.get("module"):
        raise Unavailable(
            f"the gnn package is not installed in {python}; provision the pinned revision "
            "or pass --gnn-root"
        )
    locations = located.get("locations") or []
    return (
        Path(locations[0]).resolve() if locations else Path(located["module"]).resolve().parent
    )


def installed_revision(python: str) -> dict[str, Any]:
    """Return the VCS commit the installer recorded for the ``gnn`` distribution.

    ``direct_url.json`` carries ``vcs_info.commit_id`` only for an install from a
    git URL; an editable or index install has no commit to compare and is
    reported as ``unresolvable-offline`` rather than assumed to match.
    """

    output = _run_text(
        [python, "-c", REVISION_PROGRAM], what=f"cannot read the installed gnn metadata in {python}"
    )
    try:
        records = json.loads(output)
    except json.JSONDecodeError as exc:
        raise Unavailable(f"cannot read the installed gnn metadata in {python}: {exc}") from exc
    commits = sorted({record["commit"] for record in records if record.get("commit")})
    if not commits:
        return {
            "status": "unresolvable-offline",
            "commit": None,
            "detail": "the installer recorded no VCS commit (not installed from a git URL)",
        }
    if len(commits) > 1:
        return {
            "status": "conflicting",
            "commit": None,
            "detail": f"the installer recorded several commits {commits}",
        }
    return {"status": "verified", "commit": commits[0], "detail": "direct_url.json vcs_info.commit_id"}


def revision_failures(revision: dict[str, Any], pin: dict[str, Any]) -> list[str]:
    """Return failures when the installed runtime's recorded commit is not the pin."""

    if revision.get("status") == "verified" and revision.get("commit") != pin["commit"]:
        return [
            f"{TAG_COMMIT_MISMATCH}: the installed runtime records commit "
            f"{revision.get('commit')} but the pin records {pin['commit']}"
        ]
    if revision.get("status") == "conflicting":
        return [
            f"{TAG_COMMIT_MISMATCH}: the installed runtime metadata is ambiguous: "
            f"{revision.get('detail')}"
        ]
    return []


def remote_url(repository: str) -> str:
    """Return a git URL for the pin's repository.

    ``gnn.importers`` normalizes the pin's repository to ``owner/name`` for a
    GitHub URL; ``git ls-remote`` needs the full URL.
    """

    if "://" in repository or repository.startswith("git@") or repository.startswith("/"):
        return repository
    return f"https://github.com/{repository}"


def tag_binding(
    gnn_root: Path | None, pin: dict[str, Any], *, remote: bool = False
) -> dict[str, Any]:
    """Bind the pin's ``tag`` to its ``commit`` in an available object store.

    A GNN checkout resolves the tag locally; ``remote`` resolves it upstream with
    ``git ls-remote``. An explicit upstream request is never answered by a local
    tag: a tag that moved upstream after this checkout fetched it is stale here,
    so ``remote`` overrides the local reading instead of accepting it. When
    neither store can answer the status says so instead of assuming the binding
    holds.
    """

    tag = pin["tag"]
    record: dict[str, Any] = {
        "status": "unresolvable-offline",
        "tag": tag,
        "commit": pin["commit"],
        "resolved": None,
        "detail": "no GNN object store was supplied and --verify-tag-remote was not requested",
    }
    local: dict[str, Any] | None = None
    if gnn_root is not None and (gnn_root / ".git").exists():
        try:
            local_run = subprocess.run(
                [
                    "git", "-C", str(gnn_root), "rev-parse", "--verify", "-q",
                    f"refs/tags/{tag}^{{commit}}",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            record["detail"] = f"git could not run: {exc}"
        else:
            if local_run.returncode == 0 and local_run.stdout.strip():
                local = {
                    "resolved": local_run.stdout.strip(),
                    "detail": f"local object store {gnn_root}",
                }
            else:
                record["detail"] = f"tag {tag} is not present in {gnn_root}"
    if not remote:
        if local is not None:
            record.update(status="verified", **local)
        return record
    try:
        completed = subprocess.run(
            ["git", "ls-remote", "--tags", remote_url(pin["repository"]), f"refs/tags/{tag}*"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        record["detail"] = f"git ls-remote could not run: {exc}"
        return record
    if completed.returncode != 0:
        record["detail"] = f"git ls-remote failed: {completed.stderr.strip()}"
        return record
    pairs = [line.split() for line in completed.stdout.splitlines() if len(line.split()) == 2]
    # An annotated tag lists the tag object and, peeled, the commit it names; the
    # peeled entry is the commit.
    for wanted in (f"refs/tags/{tag}^{{}}", f"refs/tags/{tag}"):
        for sha, name in pairs:
            if name == wanted:
                record.update(status="verified", resolved=sha, detail="git ls-remote")
                return record
    record["detail"] = f"tag {tag} is not listed by {pin['repository']}"
    return record


def tag_failures(tag: dict[str, Any] | None, pin: dict[str, Any]) -> list[str]:
    """Return the named failure when the pin's tag resolves to a different commit."""

    if tag is not None and tag.get("status") == "verified" and tag.get("resolved") != pin["commit"]:
        return [
            f"{TAG_COMMIT_MISMATCH}: tag {pin['tag']} resolves to {tag.get('resolved')} "
            f"but the pin records {pin['commit']}"
        ]
    return []


def materialize_pin(gnn_root: Path, pin: dict[str, Any], target: Path) -> tuple[Path, str]:
    """Materialize the pinned commit into ``target`` and return it with its origin.

    ``target`` belongs to the caller, so a run leaves no materialized tree behind.
    """

    if not gnn_root.is_dir():
        raise Unavailable(f"GNN checkout is not a directory: {gnn_root}")
    if not (gnn_root / ".git").exists():
        _verify_digests(gnn_root / "src" / "gnn", pin["parser_files"], origin=str(gnn_root))
        return gnn_root, f"digest-verified source tree {gnn_root} (no git metadata)"

    commit = pin["commit"]
    resolved = subprocess.run(
        ["git", "-C", str(gnn_root), "rev-parse", f"{commit}^{{commit}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if resolved.returncode != 0:
        raise Unavailable(f"pinned commit {commit} is not present in {gnn_root}")
    if resolved.stdout.strip() != commit:
        raise Unavailable(
            f"resolved commit {resolved.stdout.strip()} differs from the pin {commit}"
        )

    target.mkdir(parents=True, exist_ok=True)
    archive = target / "pinned.tar"
    completed = subprocess.run(
        ["git", "-C", str(gnn_root), "archive", "--format=tar", f"--output={archive}", commit],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise Unavailable(f"git archive failed for {commit}: {completed.stderr.strip()}")
    with tarfile.open(archive) as handle:
        try:
            handle.extractall(target, filter="data")
        except TypeError:  # pragma: no cover - Python < 3.12 fallback
            handle.extractall(target)
    archive.unlink()
    _verify_digests(target / "src" / "gnn", pin["parser_files"], origin=f"archive of {commit}")
    return target, f"git archive of {commit}"


def run_probe(
    python: str, spec: Path, output: Path, *, source: Path | None
) -> dict[str, Any]:
    """Run the pinned parser probe in a subprocess and return its payload."""

    environment = dict(os.environ)
    if source is not None:
        existing = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = str(source / "src") + (
            os.pathsep + existing if existing else ""
        )
    completed = subprocess.run(
        [python, str(PROBE), "--spec", str(spec), "--output", str(output)],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        tail = " | ".join(detail[-3:]) if detail else "no output"
        raise Unavailable(
            f"pinned parser probe failed with {python}: {tail}. Provide an interpreter that "
            "has the GNN parser dependencies (the GNN checkout's own .venv, or a runtime "
            "installed from the pinned revision) via --python/--gnn-runtime."
        )
    return json.loads(output.read_text(encoding="utf-8"))


def _variable_dimensions(parser: dict[str, Any]) -> dict[str, list[int]]:
    return {
        variable["name"]: list(variable["dimensions"]) for variable in parser["variables"]
    }


def conformance_failures(
    payload: dict[str, Any],
    *,
    pin: dict[str, Any],
    source: Path | None,
    spec_text: str,
    package_dir: Path | None = None,
    tag: dict[str, Any] | None = None,
) -> list[str]:
    """Return every conformance failure between the document and the model."""

    failures: list[str] = []
    parser = payload["parser"]
    expected_version = pin["tag"].lstrip("v")

    if payload["gnn_version"] != expected_version:
        failures.append(
            f"parser version {payload['gnn_version']} differs from the pinned {expected_version}"
        )
    failures.extend(tag_failures(tag, pin))
    if source is not None:
        module = Path(str(payload["gnn_module"])).resolve()
        if not module.is_relative_to((source / "src").resolve()):
            failures.append(f"parser was imported from {module}, outside the pinned tree {source}")
    elif package_dir is not None:
        module = Path(str(payload["gnn_module"])).resolve()
        if not module.is_relative_to(package_dir):
            failures.append(
                f"parser was imported from {module}, outside the verified package {package_dir}"
            )
    if not parser["success"]:
        failures.append("pinned parser reported failure")
    for message in parser["errors"]:
        failures.append(f"parser error: {message}")
    for message in parser["warnings"]:
        failures.append(f"parser warning: {message}")

    parameters = spec_parameters()
    try:
        parsed_parameters = parameters_from_parsed(parser)
    except (KeyError, TypeError, ValueError) as exc:
        failures.append(f"cannot read fixture parameters from the document: {exc}")
    else:
        if parsed_parameters != parameters:
            failures.append(
                f"declared parameters {parsed_parameters} differ from the canonical fixture "
                f"{parameters}"
            )
    if parser["model_name"] != parameters.model_name:
        failures.append(
            f"declared model name {parser['model_name']!r} differs from "
            f"{parameters.model_name!r}"
        )

    if spec_text != render_spec(parameters):
        failures.append("committed document differs from blockference.gnn.render_spec output")

    expected = spec_arrays(parameters)
    try:
        declared = declared_arrays(parser)
    except (KeyError, TypeError, ValueError) as exc:
        failures.append(f"cannot read declared matrices from the document: {exc}")
    else:
        dimensions = _variable_dimensions(parser)
        for key, array in expected.items():
            observed = declared[key]
            if observed.shape != array.shape:
                failures.append(
                    f"declared {key} shape {observed.shape} differs from the runtime {array.shape}"
                )
                continue
            if not np.array_equal(observed, array):
                failures.append(f"declared {key} differs from the runtime model")
            if dimensions.get(key) != list(array.shape):
                failures.append(
                    f"StateSpaceBlock {key} dimensions {dimensions.get(key)} differ from "
                    f"the declared matrix shape {list(array.shape)}"
                )

    semantic = payload["semantic"]
    if not semantic.get("is_valid"):
        failures.append(
            f"pinned semantic validator rejected the document: {semantic.get('errors')}"
        )
    for message in semantic.get("warnings", []):
        failures.append(f"semantic warning: {message}")

    orientation = payload["orientation"]
    for message in orientation.get("warnings", []):
        failures.append(f"B-orientation warning: {message}")
    for tensor in orientation.get("tensors", []):
        if tensor.get("orientation") != "canonical":
            failures.append(
                f"B tensor {tensor.get('variable')} orientation is {tensor.get('orientation')}, "
                "not canonical"
            )

    consistency = payload["consistency"]
    if not consistency.get("consistent"):
        failures.append(f"pinned consistency checker reported: {consistency.get('warnings')}")

    return failures


def main(argv: list[str] | None = None) -> int:
    """Validate the committed specification against the pinned parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gnn-root",
        type=Path,
        default=Path(os.environ["GNN_ROOT"]) if os.environ.get("GNN_ROOT") else None,
        help="Generalized_Notation_Notation checkout (default: $GNN_ROOT)",
    )
    parser.add_argument(
        "--gnn-runtime",
        default=os.environ.get("GNN_RUNTIME_PYTHON"),
        help="interpreter with the pinned GNN distribution installed",
    )
    parser.add_argument(
        "--python",
        default=os.environ.get("GNN_PYTHON"),
        help="parser interpreter used with --gnn-root (default: $GNN_PYTHON or this interpreter)",
    )
    parser.add_argument(
        "--verify-tag-remote",
        action="store_true",
        help="resolve the pin's tag upstream with git ls-remote (needs network access)",
    )
    parser.add_argument("--spec", type=Path, default=GNN_SPEC_PATH)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--json", action="store_true", help="print the receipt to stdout")
    args = parser.parse_args(argv)

    if args.gnn_root is None and not args.gnn_runtime:
        print(
            "gnn validation not run: supply --gnn-root/GNN_ROOT or --gnn-runtime/"
            "GNN_RUNTIME_PYTHON",
            file=sys.stderr,
        )
        return 2

    reader_python = args.gnn_runtime or args.python or sys.executable
    try:
        pin = read_importer_report(
            reader_python,
            source=args.gnn_root / "src" if args.gnn_root is not None else None,
        )["pin"]
    except Unavailable as exc:
        print(f"gnn validation not run: {exc}", file=sys.stderr)
        return 2

    tag = tag_binding(args.gnn_root, pin, remote=args.verify_tag_remote)
    if args.verify_tag_remote and tag["status"] != "verified":
        # An explicit request to verify the tag must not degrade into a silent
        # pass when nothing could answer (offline, tag not listed).
        print(f"gnn validation not run: --verify-tag-remote: {tag['detail']}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="activeblockference-gnn-") as directory:
        output = Path(directory) / "probe.json"
        try:
            if args.gnn_root is not None:
                source, origin = materialize_pin(
                    args.gnn_root, pin, Path(directory) / "gnn-pin"
                )
                package_dir = None
                if (args.gnn_root / ".git").exists():
                    revision = {
                        "status": "verified",
                        "commit": pin["commit"],
                        "detail": "git archive of the pinned commit",
                    }
                else:
                    revision = {
                        "status": "unresolvable-offline",
                        "commit": None,
                        "detail": "digest-verified source tree without git metadata",
                    }
                payload = run_probe(
                    args.python or sys.executable, args.spec, output, source=source
                )
            else:
                source = None
                # The digest and revision checks come before the parser runs, so a
                # tampered parser is never reached by parsing first.
                package_dir = locate_package(args.gnn_runtime)
                _verify_digests(package_dir, pin["parser_files"], origin=args.gnn_runtime)
                revision = installed_revision(args.gnn_runtime)
                payload = run_probe(args.gnn_runtime, args.spec, output, source=None)
                origin = f"installed runtime {args.gnn_runtime} ({package_dir})"
        except Unavailable as exc:
            print(f"gnn validation not run: {exc}", file=sys.stderr)
            return 2

        failures = conformance_failures(
            payload,
            pin=pin,
            source=source,
            spec_text=args.spec.read_text(encoding="utf-8"),
            package_dir=package_dir,
            tag=tag,
        )
        failures.extend(revision_failures(revision, pin))

    receipt = {
        "ok": not failures,
        "spec": str(args.spec),
        "pin": pin,
        "pin_source": origin,
        "gnn_version": payload["gnn_version"],
        "gnn_module": payload["gnn_module"],
        "tag_binding": tag,
        "revision": revision,
        "parser_success": payload["parser"]["success"],
        "semantic_valid": payload["semantic"].get("is_valid"),
        "orientation": [
            tensor.get("orientation") for tensor in payload["orientation"].get("tensors", [])
        ],
        "consistency_score": payload["consistency"].get("consistency_score"),
        "failures": failures,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(receipt, sort_keys=True))
    if failures:
        print("gnn conformance failures:", file=sys.stderr)
        for message in failures:
            print(f"  {message}", file=sys.stderr)
        return 1
    print(
        f"gnn validation passed: {args.spec} parsed by gnn {payload['gnn_version']} "
        f"({origin}); tag {tag['tag']} -> {tag['status']}; "
        f"revision {revision['status']}; receipt {args.receipt}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
