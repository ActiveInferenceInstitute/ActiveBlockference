# Local research orchestration

`GRTs` provides a small, explicit provider interface for corpus-backed research workflows.
The default provider is deterministic and local, so it requires no credentials or network.
The workspace is always supplied by the caller; `ResearchWorkspace.prepare()` creates only
the documented request, corpus, research-data, and report files beneath that root.

```bash
uv run grts run --workspace GRTs/data3 --provider offline
uv run grts csv output/run/data/trajectory.csv --output /tmp/analysis.json
```

The optional OpenAI provider is selected explicitly with `--provider openai` and requires the
`active-blockference[research]` extra plus `OPENAI_API_KEY`. Its workspace and model are
configured by the caller; there are no implicit global paths or provider instances. Offline
tests and release gates never make network requests. A provider response is research input,
not a software-integrity or scientific-validity verdict; preserve its provenance separately.
