"""Real-file/process controls for exact-pin GNN source admission.

The source cohort requires an explicitly supplied reviewed artifact in
GNN_TEST_SOURCE_ROOT, the reviewed GNN_BOOTSTRAP/GNN_SOURCE_MANIFEST resources,
and the existing selected external interpreter. It never installs a parser or
invents a source manifest/distribution. Installed success is a separate opt-in
cohort with GNN_TEST_INSTALLED_PYTHON and actual PEP610 Git provenance.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.manuscript_variables as variables
import scripts.validate_gnn_spec as validator
from blockference.gnn import render_spec
from blockference.gnn_runtime import resolve_gnn_runtime
from blockference.gnn_source import (
    BOOTSTRAP_SHA256,
    PINNED_COMMIT,
    SOURCE_MANIFEST_SHA256,
    worker_command,
)

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "blockference" / "gnn_source.py"


def _process(command, *, cwd, environment=None):
    return subprocess.run(
        command, cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, check=False, timeout=120,
    )


def _worker(runtime, action, repo, *, cwd, source=None, spec=None, output=None):
    arguments = ["--repo", str(repo)]
    if source is not None:
        arguments += ["--source-root", str(source)]
    if spec is not None:
        arguments += ["--spec", str(spec)]
    if output is not None:
        arguments += ["--output", str(output)]
    return _process(worker_command(runtime, action, *arguments), cwd=cwd)


@pytest.mark.parametrize("selection", ["missing", "corrupt"])
@pytest.mark.parametrize("action", ["inspect", "probe", "check"])
def test_unchecked_bootstrap_cannot_execute_or_publish(tmp_path, monkeypatch, selection, action):
    marker = tmp_path / "unchecked-bootstrap-executed"
    if selection == "missing":
        monkeypatch.delenv("GNN_BOOTSTRAP", raising=False)
    else:
        bootstrap = tmp_path / "unchecked.py"
        bootstrap.write_text(
            f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("GNN_BOOTSTRAP", str(bootstrap))
    output = tmp_path / "previous-output.json"
    output.write_bytes(b"retain previous output")
    result = _worker(
        sys.executable, action, ROOT, cwd=tmp_path,
        spec=tmp_path / "unread-spec.gnn.md", output=output,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "GNN_BOOTSTRAP" in result.stderr
    assert not marker.exists()
    assert output.read_bytes() == b"retain previous output"


def _consumer(root):
    specs = root / "specs"
    specs.mkdir(parents=True)
    (specs / "gnn-pin.json").write_bytes((ROOT / "specs" / "gnn-pin.json").read_bytes())
    spec = specs / "gridworld_3x3.gnn.md"
    spec.write_text(render_spec(), encoding="utf-8")
    pointer = root / ".github" / "GNN.md"
    pointer.parent.mkdir()
    pointer.write_text(
        f"specs/gnn-pin.json v5.1.0 {PINNED_COMMIT} ci.yml gnn-conformance\n",
        encoding="utf-8",
    )
    return spec


@pytest.fixture
def reviewed_resources(tmp_path, monkeypatch):
    names = ("GNN_BOOTSTRAP", "GNN_SOURCE_MANIFEST")
    if not all(os.environ.get(name) for name in names):
        if os.environ.get("GNN_TEST_SOURCE_ROOT") or os.environ.get("GNN_TEST_INSTALLED_PYTHON"):
            pytest.fail("the explicitly selected external cohort needs both reviewed resources")
        pytest.skip("reviewed producer resources were not selected; no admission proof")
    bootstrap_bytes = Path(os.environ["GNN_BOOTSTRAP"]).read_bytes()
    manifest_bytes = Path(os.environ["GNN_SOURCE_MANIFEST"]).read_bytes()
    assert hashlib.sha256(bootstrap_bytes).hexdigest() == BOOTSTRAP_SHA256
    assert hashlib.sha256(manifest_bytes).hexdigest() == SOURCE_MANIFEST_SHA256
    bootstrap = tmp_path / "reviewed-bootstrap.py"
    manifest = tmp_path / "reviewed-manifest.json"
    bootstrap.write_bytes(bootstrap_bytes)
    manifest.write_bytes(manifest_bytes)
    monkeypatch.setenv("GNN_BOOTSTRAP", str(bootstrap))
    monkeypatch.setenv("GNN_SOURCE_MANIFEST", str(manifest))
    repo = tmp_path / "consumer"
    spec = _consumer(repo)
    return SimpleNamespace(bootstrap=bootstrap, manifest=manifest, repo=repo, spec=spec)


@pytest.fixture
def source_inputs(reviewed_resources, tmp_path):
    selected = os.environ.get("GNN_TEST_SOURCE_ROOT")
    if not selected:
        pytest.skip("explicit reviewed source artifact not selected; corruption cohort not qualified")
    runtime = resolve_gnn_runtime()
    package = Path(selected) / "src" / "gnn"
    assert package.is_dir(), f"GNN_TEST_SOURCE_ROOT has no src/gnn: {selected}"
    source = tmp_path / "artifact"
    shutil.copytree(
        package, source / "src" / "gnn",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    return SimpleNamespace(**vars(reviewed_resources), source=source, runtime=runtime)


@pytest.mark.gnn
@pytest.mark.parametrize("action", ["probe", "check"])
def test_real_source_verdict_finishes_guarded_imports_and_revalidation(source_inputs, tmp_path, action):
    inputs = source_inputs
    output = tmp_path / "verdict.json"
    result = _worker(
        inputs.runtime, action, inputs.repo, cwd=tmp_path,
        source=inputs.source, spec=inputs.spec, output=output,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    qualification = payload["source_qualification"]
    assert qualification["ok"] and qualification["imports_verified"]
    assert qualification["commit"] == PINNED_COMMIT
    assert qualification["manifest_sha256"] == SOURCE_MANIFEST_SHA256
    assert qualification["package_root"] == str((inputs.source / "src" / "gnn").resolve())
    assert qualification["origin"]["kind"] == "explicit-source"
    assert qualification["origin"]["vcs_verified"] is False
    assert qualification["execution_verified"] is False
    modules = qualification["loaded_modules"]
    if action == "probe":
        assert payload["parser"]["success"], payload["parser"]
        assert {
            "gnn.validation.semantic_validator",
            "gnn.validation.orientation",
            "gnn.validation.consistency_checker",
        } <= modules.keys()
    else:
        assert payload["qualified"] and payload["specs"]["validated"] == 1
        assert "gnn.parsers.basic" in modules


@pytest.mark.gnn
@pytest.mark.parametrize("member", [
    "importers.py", "validation/semantic_validator.py", "validation/orientation.py",
    "validation/consistency_checker.py", "parsers/basic.py",
])
@pytest.mark.parametrize("action", ["probe", "check", "validator"])
def test_changed_verdict_members_refuse_before_execution(source_inputs, tmp_path, member, action):
    inputs = source_inputs
    marker = tmp_path / "unauthorized-module-executed"
    target = inputs.source / "src" / "gnn" / member
    target.write_bytes(
        target.read_bytes()
        + f"\nfrom pathlib import Path as _P\n_P({str(marker)!r}).write_text('executed')\n".encode()
    )
    # These verdict modules are outside the historical five-file parser pin.
    pin = json.loads((inputs.repo / "specs" / "gnn-pin.json").read_text(encoding="utf-8"))
    for relative, expected in pin["parser_files"].items():
        assert hashlib.sha256((inputs.source / relative).read_bytes()).hexdigest() == expected
    output = tmp_path / "verdict.json"
    if action == "validator":
        result = _process(
            [sys.executable, "-B", str(ROOT / "scripts" / "validate_gnn_spec.py"),
             "--gnn-runtime", inputs.runtime, "--gnn-root", str(inputs.source),
             "--spec", str(inputs.spec), "--receipt", str(output), "--json"],
            cwd=ROOT,
        )
    else:
        result = _worker(
            inputs.runtime, action, inputs.repo, cwd=tmp_path,
            source=inputs.source, spec=inputs.spec, output=output,
        )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "source-content-mismatch" in result.stdout + result.stderr
    assert not marker.exists()
    if output.exists():
        failed = json.loads(output.read_text(encoding="utf-8"))
        assert failed["ok"] is False and failed["qualified"] is False


@pytest.mark.gnn
@pytest.mark.parametrize("action", ["probe", "check"])
def test_changed_manifest_is_not_replaced_by_a_target_inventory(source_inputs, tmp_path, action):
    inputs = source_inputs
    inputs.manifest.write_bytes(inputs.manifest.read_bytes() + b" ")
    result = _worker(
        inputs.runtime, action, inputs.repo, cwd=tmp_path,
        source=inputs.source, spec=inputs.spec,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "source-manifest-hash-mismatch" in result.stdout + result.stderr


@pytest.mark.gnn
@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_complete_manifest_rejects_package_membership_changes(source_inputs, tmp_path, mutation):
    inputs = source_inputs
    package = inputs.source / "src" / "gnn"
    if mutation == "missing":
        (package / "validation" / "orientation.py").unlink()
    else:
        (package / "unreviewed-resource.txt").write_text("not reviewed\n", encoding="utf-8")
    result = _worker(
        inputs.runtime, "probe", inputs.repo, cwd=tmp_path,
        source=inputs.source, spec=inputs.spec,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "source-content-mismatch" in result.stdout + result.stderr


@pytest.mark.gnn
def test_bootstrap_execution_uses_the_checked_buffer_not_a_file_reread(reviewed_resources, tmp_path):
    inputs = reviewed_resources
    marker = tmp_path / "unchecked-reread-executed"
    program = f'''
import runpy
import sys
from pathlib import Path
bootstrap = Path({str(inputs.bootstrap)!r})
original = Path.read_bytes
changed = False
def read_bytes(path):
    global changed
    data = original(path)
    if path == bootstrap and not changed:
        changed = True
        path.write_text("from pathlib import Path\\nPath({str(marker)!r}).write_text('executed')\\n")
    return data
Path.read_bytes = read_bytes
sys.argv = [{str(WORKER)!r}, "inspect", "--repo", {str(inputs.repo)!r}]
runpy.run_path({str(WORKER)!r}, run_name="__main__")
'''
    result = _process([sys.executable, "-I", "-B", "-c", program], cwd=tmp_path)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "bootstrap-changed" in result.stderr
    assert not marker.exists()


@pytest.mark.gnn
def test_probe_revalidates_after_the_actual_parser_and_verdict_calls(source_inputs, tmp_path):
    inputs = source_inputs
    published = tmp_path / "incorrect-success.json"
    member = inputs.source / "src" / "gnn" / "validation" / "semantic_validator.py"
    program = f'''
import runpy
from pathlib import Path
worker = runpy.run_path({str(WORKER)!r})
class ChangingSpec(type(Path())):
    def read_text(self, *args, **kwargs):
        text = super().read_text(*args, **kwargs)
        member = Path({str(member)!r})
        member.write_bytes(member.read_bytes() + b"\\n# changed during actual verdict\\n")
        return text
try:
    worker["probe"](
        Path({str(inputs.repo)!r}), ChangingSpec({str(inputs.spec)!r}),
        source_root=Path({str(inputs.source)!r}),
    )
except ValueError as exc:
    raise RuntimeError(getattr(exc, "code", "unexpected-admission-failure")) from exc
Path({str(published)!r}).write_text("should never publish")
'''
    result = _process([inputs.runtime, "-I", "-B", "-c", program], cwd=tmp_path)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "source-content-mismatch" in result.stderr
    assert not published.exists()
    assert member.read_bytes().endswith(b"# changed during actual verdict\n")


@pytest.mark.gnn
def test_already_imported_real_gnn_cannot_start_a_new_verdict(source_inputs, tmp_path):
    inputs = source_inputs
    program = f'''
import runpy
import sys
sys.path.insert(0, {str(inputs.source / "src")!r})
import gnn
sys.argv = [{str(WORKER)!r}, "probe", "--repo", {str(inputs.repo)!r},
            "--source-root", {str(inputs.source)!r}, "--spec", {str(inputs.spec)!r}]
runpy.run_path({str(WORKER)!r}, run_name="__main__")
'''
    result = _process([inputs.runtime, "-I", "-B", "-c", program], cwd=tmp_path)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "source-already-imported" in result.stderr


@pytest.mark.gnn
def test_parent_helpers_use_the_selected_external_runtime_for_real_verdicts(
    source_inputs, tmp_path, monkeypatch,
):
    inputs = source_inputs
    monkeypatch.setattr(validator, "ROOT", inputs.repo)
    monkeypatch.setenv("GNN_RUNTIME_PYTHON", sys.executable)
    selected = resolve_gnn_runtime(inputs.runtime)
    report = validator.read_pin_report(selected)
    probe = validator.run_probe(
        selected, inputs.spec, tmp_path / "probe.json", source=inputs.source,
    )
    importer = validator.run_importer_check(
        selected, tmp_path / "importer.json", source=inputs.source,
    )
    assert report["mode"] == "inspection" and report["qualified"] is False
    for payload in (report, probe, importer):
        assert Path(payload["runtime_python"]).absolute() == Path(selected).absolute()
    assert probe["parser"]["success"]
    assert importer["qualified"]


@pytest.mark.gnn
def test_manuscript_metadata_uses_checked_reader_without_executing_target_gnn(
    source_inputs, tmp_path, monkeypatch,
):
    inputs = source_inputs
    marker = tmp_path / "target-importer-executed"
    target = inputs.source / "src" / "gnn" / "importers.py"
    target.write_bytes(
        target.read_bytes()
        + f"\nfrom pathlib import Path as _P\n_P({str(marker)!r}).write_text('executed')\n".encode()
    )
    monkeypatch.setattr(variables, "ROOT", inputs.repo)
    monkeypatch.setenv("PYTHONPATH", str(inputs.source / "src"))
    payload = variables._gnn_variables(
        "gnn-conformance", "Pinned conformance", gnn_runtime=inputs.runtime,
    )
    assert payload["gnn_commit"] == PINNED_COMMIT
    assert payload["gnn_job_key"] == "gnn-conformance"
    assert not marker.exists()


@pytest.mark.gnn
def test_installed_mode_does_not_fall_back_to_environment_source(source_inputs, tmp_path, monkeypatch):
    inputs = source_inputs
    empty = tmp_path / "unprovisioned-runtime"
    venv.EnvBuilder(with_pip=False).create(empty)
    python = empty / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    monkeypatch.setenv("GNN_ROOT", str(inputs.source))
    monkeypatch.setenv("PYTHONPATH", str(inputs.source / "src"))
    result = _worker(str(python), "probe", inputs.repo, cwd=tmp_path, spec=inputs.spec)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "source-origin-unavailable" in result.stderr


@pytest.mark.gnn
def test_wrong_explicit_runtime_is_not_substituted_after_source_admission(
    source_inputs, tmp_path, monkeypatch,
):
    inputs = source_inputs
    empty = tmp_path / "unprovisioned-runtime"
    venv.EnvBuilder(with_pip=False).create(empty)
    python = empty / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    monkeypatch.setenv("GNN_RUNTIME_PYTHON", inputs.runtime)
    selected = resolve_gnn_runtime(str(python))
    monkeypatch.setattr(validator, "ROOT", inputs.repo)
    output = tmp_path / "never-published.json"
    with pytest.raises(validator.Unavailable, match="worker failed"):
        validator.run_probe(selected, inputs.spec, output, source=inputs.source)
    assert not output.exists()


@pytest.mark.gnn
def test_prepared_installed_runtime_qualifies_actual_distribution_origin(reviewed_resources, tmp_path):
    runtime = os.environ.get("GNN_TEST_INSTALLED_PYTHON")
    if not runtime:
        pytest.skip("actual installed-origin cohort not selected; source proof is not installed proof")
    inputs = reviewed_resources
    output = tmp_path / "installed.json"
    result = _worker(
        resolve_gnn_runtime(runtime), "probe", inputs.repo,
        cwd=tmp_path, spec=inputs.spec, output=output,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    qualification = payload["source_qualification"]
    assert qualification["ok"] and qualification["imports_verified"]
    origin = qualification["origin"]
    assert origin["kind"] == "installed-vcs" and origin["vcs_verified"]
    assert origin["distribution"] == "generalized-notation-notation"
    assert origin["commit"] == PINNED_COMMIT
    assert origin["repository"] == "ActiveInferenceInstitute/Generalized_Notation_Notation"
    assert payload["parser"]["success"]


def _installed_command(inputs, action, *, output=None):
    arguments = [
        str(WORKER), action, "--repo", str(inputs.repo), "--spec", str(inputs.spec),
    ]
    if output is not None:
        arguments += ["--output", str(output)]
    program = (
        "import runpy, sys; "
        f"sys.path.insert(0, {str(inputs.site)!r}); "
        f"sys.argv = {arguments!r}; "
        f"runpy.run_path({str(WORKER)!r}, run_name='__main__')"
    )
    return [inputs.runtime, "-I", "-B", "-c", program]


@pytest.fixture
def installed_inputs(reviewed_resources, tmp_path):
    selected = os.environ.get("GNN_TEST_INSTALLED_PYTHON")
    if not selected:
        pytest.skip("real installed-origin corruption cohort not selected; no installed proof")
    runtime = resolve_gnn_runtime(selected)
    # Read only actual distribution metadata in the prepared interpreter. Never
    # create a pretend direct_url, version, RECORD or successful parser module.
    locate = """
import importlib.metadata as md
import json
from pathlib import Path
distribution = md.distribution("generalized-notation-notation")
metadata = [
    Path(distribution.locate_file(item)).parent
    for item in distribution.files or ()
    if item.name == "METADATA" and item.parent.name.endswith(".dist-info")
]
if len(metadata) != 1:
    raise ValueError("installed distribution has no unique real metadata directory")
print(json.dumps({
    "package": str(Path(distribution.locate_file("gnn/__init__.py")).parent),
    "metadata": str(metadata[0]),
}))
"""
    located = _process([runtime, "-I", "-B", "-c", locate], cwd=tmp_path)
    assert located.returncode == 0, located.stdout + located.stderr
    paths = json.loads(located.stdout)
    site = tmp_path / "private-installed-copy"
    shutil.copytree(
        paths["package"], site / "gnn",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    metadata = Path(paths["metadata"])
    shutil.copytree(metadata, site / metadata.name)
    inputs = SimpleNamespace(
        **vars(reviewed_resources), runtime=runtime, site=site,
        metadata=site / metadata.name,
    )
    # The unchanged genuine distribution copy must run a real admitted verdict
    # before its negative control can claim an origin/content refusal.
    baseline = tmp_path / "installed-baseline.json"
    result = _process(_installed_command(inputs, "probe", output=baseline), cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(baseline.read_text(encoding="utf-8"))
    proof = payload["source_qualification"]
    assert proof["ok"] and proof["imports_verified"]
    assert proof["origin"]["kind"] == "installed-vcs"
    assert proof["origin"]["commit"] == PINNED_COMMIT
    assert payload["parser"]["success"]
    return inputs


@pytest.mark.gnn
@pytest.mark.parametrize("member", [
    "importers.py", "validation/semantic_validator.py", "validation/orientation.py",
    "validation/consistency_checker.py",
])
@pytest.mark.parametrize("action", ["probe", "check"])
def test_installed_verdict_member_corruption_refuses_before_execution(
    installed_inputs, tmp_path, member, action,
):
    inputs = installed_inputs
    marker = tmp_path / "unreviewed-installed-code-executed"
    target = inputs.site / "gnn" / member
    target.write_bytes(
        target.read_bytes()
        + f"\nfrom pathlib import Path as _P\n_P({str(marker)!r}).write_text('executed')\n".encode()
    )
    result = _process(_installed_command(inputs, action), cwd=tmp_path)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "source-content-mismatch" in result.stdout + result.stderr
    assert not marker.exists()


@pytest.mark.gnn
@pytest.mark.parametrize("change", ["missing", "wrong-revision", "wrong-repository", "unowned"])
@pytest.mark.parametrize("action", ["probe", "check"])
def test_actual_installed_origin_and_module_ownership_are_required(
    installed_inputs, tmp_path, monkeypatch, change, action,
):
    inputs = installed_inputs
    direct_url = inputs.metadata / "direct_url.json"
    if change == "missing":
        direct_url.unlink()
    elif change == "unowned":
        # Keep genuine metadata, but relocate only the package away from its
        # owner's locate_file root. A shadow package is never distribution-owned.
        shadow = tmp_path / "shadow"
        shutil.copytree(inputs.site / "gnn", shadow / "gnn")
        inputs.site = shadow
    else:
        origin = json.loads(direct_url.read_text(encoding="utf-8"))
        if change == "wrong-revision":
            origin["vcs_info"]["commit_id"] = "0" * 40
        else:
            origin["url"] = "https://github.com/not-the-producer/Generalized_Notation_Notation"
        direct_url.write_text(json.dumps(origin), encoding="utf-8")
    # Environment source inputs cannot rescue an installed-origin failure.
    monkeypatch.setenv("GNN_ROOT", str(tmp_path / "unselected-source"))
    result = _process(_installed_command(inputs, action), cwd=tmp_path)
    assert result.returncode == 2, result.stdout + result.stderr
    expected = "source-origin-unavailable" if change == "missing" else "source-origin-mismatch"
    assert expected in result.stdout + result.stderr


@pytest.mark.gnn
def test_source_changed_after_admission_is_checked_again_before_import(source_inputs, tmp_path):
    inputs = source_inputs
    target = inputs.source / "src" / "gnn" / "__init__.py"
    marker = tmp_path / "late-unchecked-import-executed"
    replacement = (
        target.read_bytes()
        + f"\nfrom pathlib import Path as _P\n_P({str(marker)!r}).write_text('executed')\n".encode()
    )
    program = f'''
import builtins
import runpy
from pathlib import Path
worker = runpy.run_path({str(WORKER)!r})
original_import = builtins.__import__
changed = False
def importing(name, *args, **kwargs):
    global changed
    if name == "gnn" and not changed:
        changed = True
        Path({str(target)!r}).write_bytes({replacement!r})
    return original_import(name, *args, **kwargs)
builtins.__import__ = importing
try:
    worker["probe"](
        Path({str(inputs.repo)!r}), Path({str(inputs.spec)!r}),
        source_root=Path({str(inputs.source)!r}),
    )
except ValueError as exc:
    raise RuntimeError(getattr(exc, "code", "unexpected-admission-failure")) from exc
'''
    result = _process([inputs.runtime, "-I", "-B", "-c", program], cwd=tmp_path)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "source-content-mismatch" in result.stderr
    assert target.read_bytes() == replacement
    assert not marker.exists()
