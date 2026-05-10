# `GRTs/data3/` — shared working corpus

Files in this folder are **the agents' communication channel**. The
LangChain agents in `GRTs/agents.py` read and write here.

| File                          | Read by         | Written by         | Purpose                                    |
|-------------------------------|-----------------|--------------------|--------------------------------------------|
| `O_Project_description.txt`   | both            | human (seed)       | The original task description.             |
| `requests.txt`                | Research Joe    | Professor Karl     | Outstanding research questions.            |
| `research_data.txt`           | Professor Karl  | Research Joe       | Gathered evidence / sources.               |
| `report.txt`                  | human           | Professor Karl     | The final write-up.                        |

## Conventions

* Files are plain UTF-8 text.
* The agents append/overwrite with `WriteFileTool`; treat any file as
  potentially mutated by the next run.
* Keep the seed `O_Project_description.txt` checked in; treat the rest as
  ephemeral artefacts (do not rely on their git history).
