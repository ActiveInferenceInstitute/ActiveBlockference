# Test guidance

Tests are deterministic, local, and network-free. Each public behavior has a
focused test, malformed artefacts are negative controls, and required radCAD
and cadCAD integration tests run when the locked development environment is
installed. Use `tmp_path` for file I/O.
