"""Run exact-pin GNN operations without importing the project NumPy environment.

Execute this file directly with the selected external interpreter and ``-I -B``.
``GNN_BOOTSTRAP`` selects the independently reviewed producer bootstrap;
``GNN_SOURCE_MANIFEST`` selects its independently frozen complete v5.1.0 manifest.
Their expected hashes are source-owned constants, never target-derived inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

__all__ = [
    "BOOTSTRAP_SHA256",
    "SOURCE_MANIFEST_SHA256",
    "PINNED_COMMIT",
    "worker_command",
    "pin_report",
    "probe",
    "check_importer",
    "main",
]

BOOTSTRAP_SHA256 = "f638d4c01fe7de31e43d793e59bf382d0c3d1a82f365c59238334a2b40adfb37"
SOURCE_MANIFEST_SHA256 = "43b33c0b2b40386afa3f306fec4cc13bc1fd7c3b570fbf11d5008bad5c8e6b51"
PINNED_COMMIT = "75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf"
_BOOTSTRAP_NAME = "_blockference_reviewed_gnn_bootstrap"


def worker_command(python: str, action: str, *arguments: str) -> list[str]:
    """Build an isolated external invocation without resolving the venv launcher."""
    return [python, "-I", "-B", str(Path(__file__).resolve()), action, *arguments]


def _resource(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} must select the independently reviewed producer resource")
    return Path(value).expanduser().absolute()


@contextmanager
def _bootstrap() -> Iterator[ModuleType]:
    # Check once, then compile precisely that buffer. A loader/file reread or
    # cached bytecode must never choose the code that executes.
    path = _resource("GNN_BOOTSTRAP")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != BOOTSTRAP_SHA256:
        raise ValueError("bootstrap-hash-mismatch: GNN_BOOTSTRAP is not the reviewed bootstrap")
    if _BOOTSTRAP_NAME in sys.modules:
        raise ValueError("a reviewed GNN bootstrap session is already active")
    module = ModuleType(_BOOTSTRAP_NAME)
    module.__file__ = str(path)
    sys.modules[_BOOTSTRAP_NAME] = module
    try:
        exec(compile(data, str(path), "exec", dont_inherit=True), module.__dict__)
        yield module
        if path.read_bytes() != data:
            raise ValueError("bootstrap-changed: reviewed bootstrap changed during the operation")
    finally:
        del sys.modules[_BOOTSTRAP_NAME]


def _read_pin(bootstrap: ModuleType, repo_root: Path) -> Any:
    path = bootstrap.find_pin(repo_root)
    if path is None:
        raise ValueError(f"no GNN pin file found under {repo_root}")
    pin = bootstrap.read_pin(path)
    if (
        pin.commit != PINNED_COMMIT
        or pin.tag != "v5.1.0"
        or pin.repository != bootstrap.UPSTREAM_REPOSITORY
    ):
        raise ValueError("source-pin-mismatch: consumer pin differs from the reviewed v5.1.0 input")
    return pin


@contextmanager
def _source_path(source_root: Path | None) -> Iterator[Path | None]:
    # Source mode is explicit. Installed mode neither reads GNN_ROOT nor honors
    # inherited PYTHONPATH in the isolated worker, and never retries as source.
    previous = sys.path[:]
    try:
        package_root = None
        if source_root is not None:
            source_root = source_root.absolute()
            sys.path.insert(0, str(source_root / "src"))
            package_root = source_root / "src" / "gnn"
        yield package_root
    finally:
        sys.path[:] = previous


def pin_report(
    repo_root: Path, *, job: str | None = None, job_name: str | None = None
) -> dict[str, Any]:
    """Read pin metadata with the checked producer reader, without importing GNN.

    This is inspection only, including for manuscript variables. Actual parser
    and importer verdicts require complete source/origin admission separately.
    """
    repo_root = repo_root.resolve()
    with _bootstrap() as bootstrap:
        pin = _read_pin(bootstrap, repo_root)
        payload = {
            "mode": "inspection",
            "qualified": False,
            "runtime_python": sys.executable,
            "bootstrap_sha256": BOOTSTRAP_SHA256,
            "pin": {
                "path": pin.path.relative_to(repo_root).as_posix(),
                "schema": pin.schema,
                "repository": pin.repository,
                "tag": pin.tag,
                "commit": pin.commit,
                "recorded": pin.recorded,
                "spec_files": list(pin.spec_files),
                "parser_files": dict(pin.raw.get("parser_files", {})),
            },
            "variables": bootstrap.importer_variables(
                pin, repo_root=repo_root, job=job, job_name=job_name,
            ),
        }
    return payload


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def probe(repo_root: Path, spec: Path, *, source_root: Path | None = None) -> dict[str, Any]:
    """Parse and run all pinned validators inside one producer-qualified lifetime.

    Only JSON data escapes. Imports, parser objects, validators and conversions
    all finish before the producer revalidates the complete inventory/origin.
    """
    with _bootstrap() as bootstrap:
        pin = _read_pin(bootstrap, repo_root.resolve())
        manifest = _resource("GNN_SOURCE_MANIFEST").read_bytes()
        with _source_path(source_root) as package_root:
            with bootstrap.qualified_source(
                pin,
                manifest=manifest,
                manifest_sha256=SOURCE_MANIFEST_SHA256,
                package_root=package_root,
                origin_mode="source" if source_root is not None else "installed",
            ) as qualification:
                import gnn
                from gnn.parsers.markdown_parser import MarkdownGNNParser
                from gnn.validation.consistency_checker import check_consistency
                from gnn.validation.orientation import check_b_orientation
                from gnn.validation.semantic_validator import SemanticValidator

                result = MarkdownGNNParser().parse_file(str(spec))
                model = result.model
                semantic = SemanticValidator().validate(spec.read_text(encoding="utf-8"))
                orientation = check_b_orientation(spec)
                consistency = check_consistency(spec)
                payload = {
                    "runtime_python": sys.executable,
                    "gnn_version": gnn.__version__,
                    "gnn_module": gnn.__file__,
                    "parser": {
                        "success": bool(result.success),
                        "errors": list(result.errors),
                        "warnings": list(result.warnings),
                        "model_name": model.model_name,
                        "version": model.version,
                        "variables": [
                            {
                                "name": variable.name,
                                "dimensions": list(variable.dimensions),
                                "data_type": str(variable.data_type),
                            }
                            for variable in model.variables
                        ],
                        "connections": [
                            {
                                "sources": list(connection.source_variables),
                                "targets": list(connection.target_variables),
                                "type": str(connection.connection_type),
                            }
                            for connection in model.connections
                        ],
                        "ontology_mappings": [
                            {"variable": mapping.variable_name, "term": mapping.ontology_term}
                            for mapping in model.ontology_mappings
                        ],
                        "parameters": {
                            parameter.name: _jsonable(parameter.value)
                            for parameter in model.parameters
                        },
                        "sections": sorted(model.raw_sections),
                        "time_specification": _jsonable(model.time_specification),
                    },
                    "semantic": _jsonable(semantic),
                    "orientation": _jsonable(orientation),
                    "consistency": _jsonable(consistency),
                }
                # Freeze primitive data while GNN-derived values are still in
                # scope; no lazy objects/callables cross the admission boundary.
                payload = json.loads(json.dumps(payload))
        payload["source_qualification"] = qualification
    return payload


def check_importer(
    repo_root: Path, *, source_root: Path | None = None,
    workflow: str = ".github/workflows/ci.yml", job: str = "gnn-conformance",
) -> dict[str, Any]:
    """Run the producer's real pin/pointer/syntax check with full source admission."""
    repo_root = repo_root.resolve()
    with _bootstrap() as bootstrap:
        _read_pin(bootstrap, repo_root)
        manifest = _resource("GNN_SOURCE_MANIFEST").read_bytes()
        with _source_path(source_root) as package_root:
            payload = bootstrap.check_importer(
                repo_root,
                expected_commit=PINNED_COMMIT,
                expected_tag="v5.1.0",
                require_pointer=True,
                pointer_workflow=workflow,
                pointer_job=job,
                source_manifest=manifest,
                source_manifest_sha256=SOURCE_MANIFEST_SHA256,
                package_root=package_root,
                origin_mode="source" if source_root is not None else "installed",
            )
        payload["runtime_python"] = sys.executable
        # The producer refuses to call an empty/inspection-only result qualified.
        # Preserve that distinction even when no syntax error was encountered.
        payload["ok"] = bool(payload["ok"] and payload["qualified"])
    return payload


def main(argv: list[str] | None = None) -> int:
    """Run one fresh-process operation and publish only after exit revalidation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "probe", "check"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workflow", default=".github/workflows/ci.yml")
    parser.add_argument("--job")
    parser.add_argument("--job-name")
    args = parser.parse_args(argv)
    if args.action == "probe" and args.spec is None:
        parser.error("probe requires --spec")
    try:
        if args.action == "inspect":
            payload = pin_report(args.repo, job=args.job, job_name=args.job_name)
        elif args.action == "probe":
            payload = probe(args.repo, args.spec, source_root=args.source_root)
        else:
            payload = check_importer(
                args.repo, source_root=args.source_root,
                workflow=args.workflow, job=args.job or "gnn-conformance",
            )
    except (OSError, ValueError, ImportError) as exc:
        code = getattr(exc, "code", "source-unavailable")
        print(f"gnn operation not run: {code}: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    if args.action == "check" and not payload["ok"]:
        qualification = payload["source_qualification"]
        return 1 if qualification["ok"] and qualification["imports_verified"] else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
