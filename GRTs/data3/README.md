# `GRTs/data3/` — shared working corpus

These checked-in files are retained historical corpus/workspace material, not
a live LangChain persona protocol. Current runtime locations come from an
explicit `ResearchWorkspace(root)`.

| File | Current role |
| --- | --- |
| `O_Project_description.txt` | Optional prompt context. |
| `requests.txt` | Human input, seeded only when missing. |
| `research_data.txt` | Research data written by `ResearchOrchestrator`. |
| `report.txt` | Report written by `ResearchOrchestrator`. |

`OfflineCorpusProvider` is deterministic and network-free. OpenAI is an
explicit optional provider requiring external credentials. The implementation
has no Research Joe/Professor Karl agents or `WriteFileTool` calls.
Preserve the corpus, reports, and historical communication files; updating
this description does not authorize running a provider or rewriting them.
