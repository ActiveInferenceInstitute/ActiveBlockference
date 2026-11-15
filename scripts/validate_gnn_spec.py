"""Validate the committed GNN specification with the pinned upstream parser.

The canonical pin is read by the independently hash-checked producer bootstrap
in the selected external interpreter. No target ``gnn`` code runs to discover
its expected identity. ``GNN_BOOTSTRAP`` and ``GNN_SOURCE_MANIFEST`` select the
reviewed bootstrap and independently frozen complete v5.1.0 package manifest.

A selected checkout is materialized with ``git archive`` at the exact pin, or
explicitly admitted as a source artifact when no Git metadata exists. Installed
mode requires the producer's actual matching distribution/PEP610 Git origin and
complete package content; an unavailable origin never falls back to source.
``blockference/gnn_source.py`` runs parser, semantic, orientation, consistency
and shared importer checks in fresh isolated external processes. Every GNN
import and verdict stays inside producer qualification through exit revalidation.

The pin's ``tag`` and ``commit`` name one revision. A GNN checkout resolves
``refs/tags/<tag>^{commit}`` locally; ``--verify-tag-remote`` resolves it upstream
with ``git ls-remote`` (the peeled commit of an annotated tag), and always asks
upstream: a tag that moved after a checkout fetched it fails as
``tag-commit-mismatch`` instead of passing on the stale local tag. A tag that
resolves elsewhere fails the same way. Without a checkout or the
flag the binding is recorded in the receipt as ``unresolvable-offline``, never
as a pass, and no network is used. With the flag, a tag that cannot be resolved
ends the run as *not run* (exit ``2``).

The qualified ``blockference/gnn_source.py`` worker parses the document in a
separate interpreter and checks conformance: the declared fixture parameters, the
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
from blockference.gnn_runtime import resolve_gnn_runtime
from blockference.gnn_source import worker_command

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECEIPT = ROOT / "docs" / "_build" / "gnn" / "validation-receipt.json"

#: Named failure: the pin's tag and commit are two spellings of one revision.
TAG_COMMIT_MISMATCH = "tag-commit-mismatch"


class Unavailable(RuntimeError):
    """Raised when the pinned parser runtime or checkout is not available."""


def read_pin_report(python: str) -> dict[str, Any]:
    """Read metadata with checked producer bytes, without importing target GNN."""
    output = _run_text(
        worker_command(python, "inspect", "--repo", str(ROOT)),
        what=f"cannot run the reviewed GNN pin reader in {python}",
    )
    try:
        payload = json.loads(output)
    except ValueError as exc:
        raise Unavailable(f"the reviewed GNN pin reader returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("pin"), dict):
        raise Unavailable("the reviewed GNN pin reader returned no pin")
    return payload


def _run_text(command: list[str], *, what: str) -> str:
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False, stdin=subprocess.DEVNULL,
        )
    except OSError as exc:
        raise Unavailable(f"{what}: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        tail = " | ".join(detail[-3:]) if detail else "no output"
        raise Unavailable(f"{what}: {tail}")
    return completed.stdout


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
                stdin=subprocess.DEVNULL,
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
            stdin=subprocess.DEVNULL,
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
        return gnn_root, f"explicit source tree {gnn_root} (no Git metadata)"

    commit = pin["commit"]
    resolved = subprocess.run(
        ["git", "-C", str(gnn_root), "rev-parse", f"{commit}^{{commit}}"],
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
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
        stdin=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        raise Unavailable(f"git archive failed for {commit}: {completed.stderr.strip()}")
    with tarfile.open(archive) as handle:
        try:
            handle.extractall(target, filter="data")
        except TypeError:  # pragma: no cover - Python < 3.12 fallback
            handle.extractall(target)
    archive.unlink()
    return target, f"git archive of {commit}"


def _run_worker(
    python: str, action: str, output: Path, *, source: Path | None, spec: Path | None = None,
) -> dict[str, Any]:
    arguments = ["--repo", str(ROOT), "--output", str(output)]
    if spec is not None:
        arguments += ["--spec", str(spec)]
    if source is not None:
        arguments += ["--source-root", str(source)]
    command = worker_command(python, action, *arguments)
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False, stdin=subprocess.DEVNULL,
        )
    except OSError as exc:
        raise Unavailable(f"cannot run the qualified GNN {action} worker in {python}: {exc}") from exc
    if completed.returncode not in ({0, 1} if action == "check" else {0}):
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        tail = " | ".join(detail[-3:]) if detail else "no output"
        raise Unavailable(f"qualified GNN {action} worker failed in {python}: {tail}")
    try:
        payload = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise Unavailable(f"qualified GNN {action} worker returned no readable JSON: {exc}") from exc
    qualification = payload.get("source_qualification", {})
    if not qualification.get("ok") or not qualification.get("imports_verified"):
        raise Unavailable(f"GNN {action} worker did not complete source/origin revalidation")
    return payload


def run_probe(
    python: str, spec: Path, output: Path, *, source: Path | None,
) -> dict[str, Any]:
    """Run the fully qualified parser/verdict worker and return only JSON data."""
    return _run_worker(python, "probe", output, source=source, spec=spec)


def run_importer_check(
    python: str, output: Path, *, source: Path | None,
) -> dict[str, Any]:
    """Run the producer's qualified syntax/pointer check in a separate lifetime."""
    return _run_worker(python, "check", output, source=source)


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

    dimensions = _variable_dimensions(parser)
    for name in ("π", "G"):
        if dimensions.get(name) != [parameters.n_policies]:
            failures.append(
                f"StateSpaceBlock {name} must have {parameters.n_policies} policy entries"
            )
    declared_parameters = parser.get("parameters", {})
    for name, value in (
        ("n_actions", parameters.n_actions), ("n_policies", parameters.n_policies),
    ):
        observed = declared_parameters.get(name) if isinstance(declared_parameters, dict) else None
        if isinstance(observed, bool) or not isinstance(observed, (int, float)) or observed != value:
            failures.append(f"declared {name} {observed!r} differs from {value}")
    expected = spec_arrays(parameters)
    try:
        declared = declared_arrays(parser)
    except (KeyError, TypeError, ValueError) as exc:
        failures.append(f"cannot read declared matrices from the document: {exc}")
    else:
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
        help="interpreter with the pinned GNN distribution installed",
    )
    parser.add_argument(
        "--python",
        help="legacy external interpreter selector after --gnn-runtime/$GNN_RUNTIME_PYTHON",
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

    try:
        interpreter = resolve_gnn_runtime(
            args.gnn_runtime, checkout=args.gnn_root, python=args.python,
        )
    except (OSError, ValueError) as exc:
        print(f"gnn validation not run: {exc}", file=sys.stderr)
        return 2
    try:
        pin = read_pin_report(interpreter)["pin"]
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
                    # Materialization selects bytes; only the external producer
                    # guard below can admit them for execution.
                    args.gnn_root, pin, Path(directory) / "gnn-pin"
                )
                archived = (args.gnn_root / ".git").exists()
                if archived:
                    revision = {
                        "status": "verified",
                        "commit": pin["commit"],
                        "detail": "git archive of the pinned commit",
                    }
                else:
                    revision = {
                        "status": "unresolvable-offline",
                        "commit": None,
                        "detail": "explicit complete-manifest source without Git metadata",
                    }
                package_dir = None
                payload = run_probe(interpreter, args.spec, output, source=source)
            else:
                source = None
                payload = run_probe(interpreter, args.spec, output, source=None)
                package = payload["source_qualification"]
                package_dir = Path(package["package_root"])
                revision = {
                    "status": "verified",
                    "commit": package["origin"]["commit"],
                    "detail": "producer-qualified installed distribution/PEP610 Git origin",
                }
                origin = f"installed runtime {interpreter} ({package_dir})"
            importer = run_importer_check(
                interpreter, Path(directory) / "importer.json", source=source,
            )
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
        if not importer["ok"]:
            failures.extend(
                f"shared importer {item['code']}: {item['message']}"
                for item in importer["findings"] if item["level"] == "error"
            )

    receipt = {
        "ok": not failures,
        "runtime_python": interpreter,
        "spec": str(args.spec),
        "pin": pin,
        "pin_source": origin,
        "gnn_version": payload["gnn_version"],
        "gnn_module": payload["gnn_module"],
        "tag_binding": tag,
        "revision": revision,
        "source_qualification": payload["source_qualification"],
        "importer_check": importer,
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
