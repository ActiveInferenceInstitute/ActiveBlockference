# `docs/_generated/`

Generated documentation. Every file here is written by a named script and
checked by a gate arm; do not edit these files by hand.

| File | Producer | Check |
| --- | --- | --- |
| [`gate-metrics.md`](gate-metrics.md) | `scripts/gate_metrics.py` from [`gate-timings.json`](gate-timings.json) | `scripts/release_check.py --arm docs` |
| [`manuscript-variables.json`](manuscript-variables.json) | `scripts/manuscript_variables.py` | `scripts/release_check.py --arm manuscript` |

`gate-timings.json` is a recorded receipt from one `--timings` run of the gate;
it is evidence for the measured numbers, not a claim about another host.
