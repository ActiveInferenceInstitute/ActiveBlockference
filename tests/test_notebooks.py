"""Parse and execute every repository notebook."""

from pathlib import Path

import pytest

from blockference.notebook_inventory import execute_notebooks

pytestmark = pytest.mark.slow


def test_notebooks_parse_and_execute(tmp_path):
    root = Path(__file__).parents[1]
    execute_notebooks(root, workspace=tmp_path)
