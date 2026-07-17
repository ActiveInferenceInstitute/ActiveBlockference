# Workflow guidance

Keep backend dependencies installed in CI. A workflow may not convert a failed
required integration test into a skip. Any new required output or notebook gate
must be reflected in the smoke and validation commands.
