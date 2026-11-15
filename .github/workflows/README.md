# workflows

[`ci.yml`](ci.yml) installs the locked environment and calls the canonical
release gate in two jobs:

| Job key | Python | Selection |
| --- | --- | --- |
| `test` | matrix: 3.11 only | `--arm test-full` |
| `gnn-conformance` | 3.12 | `--arm all` with `GNN_RUNTIME_PYTHON` set to the provisioned runtime |

The second job retains the display name `gnn conformance (pinned)` and installs
the revision in [`../../specs/gnn-pin.json`](../../specs/gnn-pin.json) into a
separate runtime. Its `gnn` arm invokes the project validator for guarded deep
conformance and the shared importer check. Independently reviewed bootstrap
and full-manifest job inputs must be delivered before hosted acceptance;
no machine-local path or invented publication URL substitutes for them.
The full suite is selected on both Python versions; other release arms run
on Python 3.12. Missing prerequisites fail. There is no separate `gate` job;
PDF rendering remains an explicit local arm.

The workflow uses pinned `astral-sh/setup-uv` directly, not the Institute
composite. [`../../docs/gate.md`](../../docs/gate.md) is authoritative for the
exact action pin, gate commands, and measured-receipt workflow.
