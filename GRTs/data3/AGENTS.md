# AGENTS.md — `GRTs/data3/`

This folder is the **shared filesystem** for the GRT agents. Treat it
like a message bus.

## Hard rules

* Files here may be created, overwritten, or truncated by either agent.
* Do not commit anything you can't afford to lose. The seed
  `O_Project_description.txt` is the only stable file.
* Never put secrets here.

## Conventions

* UTF-8 plain text.
* Agents address each other by writing files; do not introduce a fancier
  protocol without updating both `agents.py` and the README table.
