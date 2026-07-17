# Research workflow guidance

The local corpus provider is deterministic and requires no network. The OpenAI
provider is opt-in, reads `OPENAI_API_KEY`, and reports missing optional
dependencies clearly. Workspace paths are explicit and all generated files
stay inside the selected workspace.
