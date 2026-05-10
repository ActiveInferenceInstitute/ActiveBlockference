# `GRTs/` — Generative Research Teams

A small experimental playground for **LangChain-powered "research teams"**
collaborating on Active Inference content.

## Layout

```
GRTs/
├── agents.py         — Defines two AutoGPT-style agents:
│                       Professor Karl (writes report)
│                       Research Assistant Joe (gathers data)
├── GRTsV2.ipynb      — Notebook orchestrating the agents
└── data3/            — Working corpus shared between the two agents
    ├── O_Project_description.txt
    ├── requests.txt
    ├── research_data.txt
    └── report.txt
```

## What it does

* `Professor Karl` is told to draft a report on Active Inference, posting
  research questions to `requests.txt`.
* `Research Assistant Joe` reads `requests.txt`, gathers data, writes
  `research_data.txt`.
* The Professor merges results into `report.txt`.

## Status

* Experimental. Depends on a recent enough `langchain` (with
  `langchain.experimental.autonomous_agents.autogpt.agent.AutoGPT`) and
  an `OPENAI_API_KEY` in the environment.
* Pinned only loosely; running this folder is **not** part of the
  package's CI. Treat it as research-grade.

## Quick run

```bash
pip install langchain openai duckduckgo-search faiss-cpu beautifulsoup4 \
            playwright nest_asyncio
playwright install chromium
export OPENAI_API_KEY=sk-...

# from a notebook or python REPL:
from GRTs.agents import setup_agents
from langchain.chat_models import ChatOpenAI
admin, research = setup_agents(ChatOpenAI(model="gpt-4o-mini"), folder="GRTs/data3")
```

See [`AGENTS.md`](AGENTS.md) for editing rules.
