# workflows

[`ci.yml`](ci.yml) installs the locked environment and calls the canonical
release gate in two jobs:

| Job key | Python | Selection |
| --- | --- | --- |
| `test` | matrix: 3.11 only | `--arm test-full` |
| `gnn-conformance` | 3.12 | `--arm all` with `GNN_RUNTIME_PYTHON` set to the provisioned runtime |

The second job retains the display name `gnn conformance (pinned)` and installs
the revision in [`../../specs/gnn-pin.json`](../../specs/gnn-pin.json) into a
separate runtime; its `gnn` arm then calls the shared
`python -m gnn.importers check --repo .` from that runtime. The full suite runs on both Python versions; all other release
arms run with conformance on Python 3.12. Missing pinned-runtime validation
fails the gate. There is no separate `gate` job, and PDF rendering remains an
explicit local arm.

The workflow uses pinned `astral-sh/setup-uv` directly, not the Institute
composite. [`../../docs/gate.md`](../../docs/gate.md) is authoritative for the
exact action pin, gate commands, and measured-receipt workflow.
