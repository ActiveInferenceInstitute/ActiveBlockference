"""In-process control-flow tests for ``blockference.gnn_source``.

The admission cohort in ``test_gnn_source_admission.py`` drives the worker in
fresh external interpreters, which coverage cannot observe. These tests drive
the same functions in-process against a *test-double* bootstrap and a
test-double ``gnn`` package so the wrapper's own logic is exercised on every
run: hash binding, single-session guard, mid-operation bootstrap changes, pin
checks, sys.path restoration, JSON freezing, ok/qualified folding and the CLI
exit-code contract.

Nothing here is an admission proof. The double is accepted only because each
test rebinds ``BOOTSTRAP_SHA256`` to the double's own digest; the reviewed
constant is never satisfied by these files, and real-producer verdicts remain
the job of the external cohort.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import blockference.gnn_source as worker

ROOT = Path(__file__).resolve().parents[1]

_DOUBLE_BOOTSTRAP = """
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

UPSTREAM_REPOSITORY = "ActiveInferenceInstitute/Generalized_Notation_Notation"
CALLS = []


def find_pin(repo_root):
    path = Path(repo_root) / "specs" / "gnn-pin.json"
    return path if path.is_file() else None


def read_pin(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    repository = raw["repository"].removeprefix("https://github.com/")
    return SimpleNamespace(
        path=Path(path), schema=raw["schema"], repository=repository, tag=raw["tag"],
        commit=raw["commit"], recorded=raw["recorded"],
        spec_files=tuple(raw.get("spec_files", ())), raw=raw,
    )


def importer_variables(pin, *, repo_root, job, job_name):
    return {"gnn_commit": pin.commit, "gnn_job_key": job, "gnn_job_name": job_name}


@contextmanager
def qualified_source(pin, *, manifest, manifest_sha256, package_root, origin_mode):
    CALLS.append(("qualified_source", manifest, manifest_sha256, package_root, origin_mode))
    yield {"ok": True, "imports_verified": True, "origin_mode": origin_mode}


def check_importer(repo_root, **options):
    CALLS.append(("check_importer", options))
    return dict(RESULT, source_qualification=dict(QUALIFICATION))


RESULT = {"ok": True, "qualified": True}
QUALIFICATION = {"ok": True, "imports_verified": True}
"""

_DOUBLE_GNN = {
    "__init__.py": '__version__ = "double"\n',
    "parsers/__init__.py": "",
    "parsers/markdown_parser.py": """
from types import SimpleNamespace


class MarkdownGNNParser:
    def parse_file(self, path):
        model = SimpleNamespace(
            model_name="double", version="1",
            variables=[SimpleNamespace(name="s", dimensions=(3, 1), data_type="categorical")],
            connections=[SimpleNamespace(
                source_variables=("s",), target_variables=("o",), connection_type="directed",
            )],
            ontology_mappings=[SimpleNamespace(variable_name="s", ontology_term="HiddenState")],
            parameters=[SimpleNamespace(name="A", value=((1, 0), (0, 1)))],
            raw_sections={"StateSpaceBlock": "", "Connections": ""},
            time_specification={1: object()},
        )
        return SimpleNamespace(success=True, errors=(), warnings=["w"], model=model)
""",
    "validation/__init__.py": "",
    "validation/semantic_validator.py": """
class SemanticValidator:
    def validate(self, text):
        return {"valid": True, "length": len(text)}
""",
    "validation/orientation.py": "def check_b_orientation(path):\n    return ('ok', None)\n",
    "validation/consistency_checker.py": "def check_consistency(path):\n    return []\n",
}


def _drop_gnn_modules() -> None:
    for name in [name for name in sys.modules if name == "gnn" or name.startswith("gnn.")]:
        del sys.modules[name]


@pytest.fixture
def double(tmp_path, monkeypatch):
    """Select a hash-bound test-double bootstrap and a consumer repository."""
    assert "gnn" not in sys.modules, "the project environment must not import GNN"
    bootstrap = tmp_path / "double_bootstrap.py"
    bootstrap.write_text(_DOUBLE_BOOTSTRAP, encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(b'{"double": true}\n')
    monkeypatch.setattr(
        worker,
        "BOOTSTRAP_SHA256",
        hashlib.sha256(bootstrap.read_bytes()).hexdigest(),
    )
    monkeypatch.setenv("GNN_BOOTSTRAP", str(bootstrap))
    monkeypatch.setenv("GNN_SOURCE_MANIFEST", str(manifest))
    repo = tmp_path / "consumer"
    (repo / "specs").mkdir(parents=True)
    (repo / "specs" / "gnn-pin.json").write_bytes((ROOT / "specs" / "gnn-pin.json").read_bytes())
    spec = repo / "specs" / "model.gnn.md"
    spec.write_text("## ModelName\ndouble\n", encoding="utf-8")
    source = tmp_path / "source"
    for relative, text in _DOUBLE_GNN.items():
        path = source / "src" / "gnn" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    yield SimpleNamespace(
        bootstrap=bootstrap, manifest=manifest, repo=repo, spec=spec, source=source
    )
    _drop_gnn_modules()
    assert worker._BOOTSTRAP_NAME not in sys.modules


def test_worker_command_is_isolated_and_targets_this_file():
    command = worker.worker_command("/runtime/python", "probe", "--spec", "x.gnn.md")
    assert command[:3] == ["/runtime/python", "-I", "-B"]
    assert Path(command[3]) == Path(worker.__file__).resolve()
    assert command[4:] == ["probe", "--spec", "x.gnn.md"]


def test_resource_must_be_selected(monkeypatch, tmp_path):
    monkeypatch.delenv("GNN_SOURCE_MANIFEST", raising=False)
    with pytest.raises(ValueError, match="GNN_SOURCE_MANIFEST must select"):
        worker._resource("GNN_SOURCE_MANIFEST")
    monkeypatch.setenv("GNN_SOURCE_MANIFEST", "relative/manifest.json")
    monkeypatch.chdir(tmp_path)
    assert worker._resource("GNN_SOURCE_MANIFEST") == tmp_path / "relative" / "manifest.json"


def test_unreviewed_bootstrap_bytes_are_never_executed(tmp_path, monkeypatch):
    marker = tmp_path / "executed"
    bootstrap = tmp_path / "unreviewed.py"
    bootstrap.write_text(f"open({str(marker)!r}, 'w').write('x')\n", encoding="utf-8")
    monkeypatch.setenv("GNN_BOOTSTRAP", str(bootstrap))
    with pytest.raises(ValueError, match="bootstrap-hash-mismatch"):
        with worker._bootstrap():
            pass  # pragma: no cover - the context must refuse before yielding
    assert not marker.exists()
    assert worker._BOOTSTRAP_NAME not in sys.modules


def test_one_bootstrap_session_at_a_time(double):
    with worker._bootstrap() as module:
        assert sys.modules[worker._BOOTSTRAP_NAME] is module
        assert module.__file__ == str(double.bootstrap)
        with pytest.raises(ValueError, match="already active"):
            with worker._bootstrap():
                pass  # pragma: no cover
        # The refused nested entry must not tear down the active session.
        assert sys.modules[worker._BOOTSTRAP_NAME] is module
    assert worker._BOOTSTRAP_NAME not in sys.modules


def test_bootstrap_changed_during_operation_is_refused(double):
    with pytest.raises(ValueError, match="bootstrap-changed"):
        with worker._bootstrap():
            double.bootstrap.write_text(_DOUBLE_BOOTSTRAP + "\n# changed\n", encoding="utf-8")
    assert worker._BOOTSTRAP_NAME not in sys.modules


def test_missing_pin_is_refused(double, tmp_path):
    empty = tmp_path / "no-pin"
    empty.mkdir()
    with worker._bootstrap() as bootstrap:
        with pytest.raises(ValueError, match="no GNN pin file"):
            worker._read_pin(bootstrap, empty)


@pytest.mark.parametrize(
    ("field", "value"),
    [("commit", "0" * 40), ("tag", "v5.0.0"), ("repository", "https://github.com/other/gnn")],
)
def test_pin_must_match_the_reviewed_input(double, field, value):
    pin_path = double.repo / "specs" / "gnn-pin.json"
    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    pin[field] = value
    pin_path.write_text(json.dumps(pin), encoding="utf-8")
    with worker._bootstrap() as bootstrap:
        with pytest.raises(ValueError, match="source-pin-mismatch"):
            worker._read_pin(bootstrap, double.repo)


def test_pin_report_is_inspection_only(double):
    payload = worker.pin_report(double.repo, job="gnn-conformance", job_name="Pinned")
    assert payload["mode"] == "inspection" and payload["qualified"] is False
    assert payload["runtime_python"] == sys.executable
    assert payload["bootstrap_sha256"] == worker.BOOTSTRAP_SHA256
    assert payload["pin"]["path"] == "specs/gnn-pin.json"
    assert payload["pin"]["commit"] == worker.PINNED_COMMIT
    assert payload["pin"]["tag"] == "v5.1.0"
    assert payload["pin"]["spec_files"] == []
    assert set(payload["pin"]["parser_files"]) >= {"src/gnn/__init__.py"}
    assert payload["variables"] == {
        "gnn_commit": worker.PINNED_COMMIT,
        "gnn_job_key": "gnn-conformance",
        "gnn_job_name": "Pinned",
    }
    assert "gnn" not in sys.modules


def test_jsonable_freezes_nested_values():
    marker = object()
    value = {1: (1, 2.5, [True, None]), "nested": {"object": marker}}
    assert worker._jsonable(value) == {
        "1": [1, 2.5, [True, None]],
        "nested": {"object": str(marker)},
    }


def test_source_path_is_explicit_and_restored(tmp_path):
    before = sys.path[:]
    with worker._source_path(None) as package_root:
        assert package_root is None
        assert sys.path == before
    with worker._source_path(tmp_path / "source") as package_root:
        assert package_root == tmp_path / "source" / "src" / "gnn"
        assert sys.path[0] == str(tmp_path / "source" / "src")
    assert sys.path == before
    with pytest.raises(RuntimeError):
        with worker._source_path(tmp_path / "source"):
            raise RuntimeError("operation failed")
    assert sys.path == before


def _record_qualified_source(monkeypatch, calls):
    """Wrap the double's qualified_source as each session executes it."""
    original = worker._bootstrap

    def recording():
        context = original()
        module = context.__enter__()
        inner = module.qualified_source

        def wrapped(*args, **kwargs):
            calls.append(kwargs)
            return inner(*args, **kwargs)

        module.qualified_source = wrapped
        return _Entered(context, module)

    monkeypatch.setattr(worker, "_bootstrap", recording)


class _Entered:
    def __init__(self, context, module):
        self._context, self._module = context, module

    def __enter__(self):
        return self._module

    def __exit__(self, *exc):
        return self._context.__exit__(*exc)


def test_probe_returns_frozen_json_from_one_qualified_lifetime(double, monkeypatch):
    calls: list[dict] = []
    _record_qualified_source(monkeypatch, calls)
    before = sys.path[:]
    payload = worker.probe(double.repo, double.spec, source_root=double.source)
    assert sys.path == before
    assert calls == [
        {
            "manifest": double.manifest.read_bytes(),
            "manifest_sha256": worker.SOURCE_MANIFEST_SHA256,
            "package_root": double.source.absolute() / "src" / "gnn",
            "origin_mode": "source",
        }
    ]
    assert payload["source_qualification"] == {
        "ok": True,
        "imports_verified": True,
        "origin_mode": "source",
    }
    assert payload["gnn_version"] == "double"
    parser = payload["parser"]
    assert parser["success"] is True and parser["errors"] == [] and parser["warnings"] == ["w"]
    assert parser["variables"] == [
        {"name": "s", "dimensions": [3, 1], "data_type": "categorical"},
    ]
    assert parser["connections"] == [{"sources": ["s"], "targets": ["o"], "type": "directed"}]
    assert parser["ontology_mappings"] == [{"variable": "s", "term": "HiddenState"}]
    assert parser["parameters"] == {"A": [[1, 0], [0, 1]]}
    assert parser["sections"] == ["Connections", "StateSpaceBlock"]
    assert set(parser["time_specification"]) == {"1"}
    assert payload["semantic"] == {"valid": True, "length": len(double.spec.read_text())}
    assert payload["orientation"] == ["ok", None]
    assert payload["consistency"] == []
    # Only primitive data escapes the admission boundary.
    assert json.loads(json.dumps(payload)) == payload


def test_installed_probe_never_falls_back_to_source(double, monkeypatch):
    if importlib.util.find_spec("gnn") is not None:
        pytest.skip("this environment provides an importable gnn; installed-mode refusal is moot")
    calls: list[dict] = []
    _record_qualified_source(monkeypatch, calls)
    monkeypatch.setenv("GNN_ROOT", str(double.source))
    with pytest.raises(ImportError):
        worker.probe(double.repo, double.spec)
    assert calls[0]["origin_mode"] == "installed" and calls[0]["package_root"] is None
    assert "gnn" not in sys.modules


def _set_check_result(monkeypatch, result, qualification):
    original = worker._bootstrap

    def selecting():
        context = original()
        module = context.__enter__()
        module.RESULT = result
        module.QUALIFICATION = qualification
        return _Entered(context, module)

    monkeypatch.setattr(worker, "_bootstrap", selecting)


@pytest.mark.parametrize(
    ("ok", "qualified", "expected"),
    [(True, True, True), (True, False, False), (False, True, False)],
)
def test_check_importer_requires_ok_and_qualified(double, monkeypatch, ok, qualified, expected):
    _set_check_result(
        monkeypatch,
        {"ok": ok, "qualified": qualified},
        {"ok": True, "imports_verified": True},
    )
    payload = worker.check_importer(double.repo, source_root=double.source)
    assert payload["ok"] is expected
    assert payload["runtime_python"] == sys.executable


def test_main_inspect_publishes_output(double, tmp_path, capsys):
    output = tmp_path / "out" / "inspect.json"
    code = worker.main(["inspect", "--repo", str(double.repo), "--output", str(output)])
    assert code == 0
    written = output.read_text(encoding="utf-8")
    assert capsys.readouterr().out == written
    assert json.loads(written)["mode"] == "inspection"


def test_main_probe_requires_a_spec(double):
    with pytest.raises(SystemExit) as raised:
        worker.main(["probe", "--repo", str(double.repo)])
    assert raised.value.code == 2


def test_main_probe_success(double, capsys):
    code = worker.main(
        [
            "probe",
            "--repo",
            str(double.repo),
            "--spec",
            str(double.spec),
            "--source-root",
            str(double.source),
        ]
    )
    assert code == 0
    assert json.loads(capsys.readouterr().out)["parser"]["success"] is True


@pytest.mark.parametrize(
    ("result", "qualification", "code"),
    [
        ({"ok": True, "qualified": True}, {"ok": True, "imports_verified": True}, 0),
        ({"ok": False, "qualified": True}, {"ok": True, "imports_verified": True}, 1),
        ({"ok": False, "qualified": False}, {"ok": False, "imports_verified": False}, 2),
        ({"ok": True, "qualified": False}, {"ok": True, "imports_verified": False}, 2),
    ],
)
def test_main_check_exit_codes(double, monkeypatch, capsys, result, qualification, code):
    _set_check_result(monkeypatch, result, qualification)
    assert worker.main(["check", "--repo", str(double.repo), "--job", "gnn-conformance"]) == code
    assert json.loads(capsys.readouterr().out)["ok"] is (code == 0)


def test_main_refusal_keeps_previous_output(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("GNN_BOOTSTRAP", raising=False)
    output = tmp_path / "previous.json"
    output.write_bytes(b"retain")
    assert worker.main(["inspect", "--repo", str(tmp_path), "--output", str(output)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "gnn operation not run: source-unavailable" in captured.err
    assert output.read_bytes() == b"retain"


def test_main_reports_a_coded_refusal(tmp_path, monkeypatch, capsys):
    class Coded(ValueError):
        code = "source-content-mismatch"

    def refuse(*args, **kwargs):
        raise Coded("changed member")

    monkeypatch.setattr(worker, "probe", refuse)
    spec = tmp_path / "x.gnn.md"
    assert worker.main(["probe", "--repo", str(tmp_path), "--spec", str(spec)]) == 2
    assert "gnn operation not run: source-content-mismatch: changed member" in (
        capsys.readouterr().err
    )
