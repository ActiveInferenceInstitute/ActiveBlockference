# `docs/_generated/`

Generated facts here are written by named scripts and checked by gate arms;
do not edit them by hand. This README explains the producers.

| File | Producer | Check |
| --- | --- | --- |
| [`gate-metrics.md`](gate-metrics.md) | `scripts/gate_metrics.py` from [`gate-timings.json`](gate-timings.json) and [`gate-baseline.json`](gate-baseline.json) | `scripts/release_check.py --arm docs` |
| [`gate-baseline.json`](gate-baseline.json) | `scripts/gate_metrics.py --record-baseline PATH` imports real pre-change gate receipts | `scripts/gate_metrics.py --check` |
| [`gate-timings.json`](gate-timings.json) | `scripts/gate_metrics.py --record-timings PATH` imports real current gate receipts | `scripts/gate_metrics.py --check` |
| [`manuscript-variables.json`](manuscript-variables.json) | `scripts/manuscript_variables.py` | `scripts/release_check.py --arm manuscript` |

`release_check.py --timings PATH` records actual gate commands, statuses, and
elapsed times. Repeat `--record-baseline PATH` for disjoint pre-change selections;
`--record-timings PATH` imports one combined current receipt into the canonical
record before generating `gate-metrics.md`. See
[`../gate.md`](../gate.md) for the authoritative recording and import commands.

The before/after records are one-run, host-specific evidence, not a controlled
benchmark or a claimed speedup. The offline package baseline cache miss is a
failed measurement, not a successful comparison. Missing arms remain unrecorded;
failed and non-equivalent workloads keep their status and qualifications.
Measured values belong in generated metrics only.
