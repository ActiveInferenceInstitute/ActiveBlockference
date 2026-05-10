# AGENTS.md — `GRTs/`

This folder is a **separate playground**, not part of the
`blockference` package. The CI suite does not run anything here.

## Hard rules

* **No** API keys committed. Read them from environment variables
  (`OPENAI_API_KEY`) or a local `.env` ignored by git.
* **No** writing outside `GRTs/data3/` from the agents' tools — both
  `WriteFileTool` and `ReadFileTool` are scoped to that folder.
* **No** importing from `GRTs/` inside `blockference/` — dependency must
  flow one direction only.

## Soft rules

* Treat `data3/` as the agents' shared working corpus. Resetting it
  between runs is fine; consider committing only the seed
  `O_Project_description.txt`.
* If you upgrade to a newer LangChain, update the import paths in
  `agents.py` (LangChain has reorganised submodules several times).

## Things to avoid

* Calling LangChain at module import time. Wrap everything in
  `setup_agents(...)` so the file is import-safe.
* Pulling in heavy ML weights — keep this folder cheap.
