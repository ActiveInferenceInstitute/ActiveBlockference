"""Tests for the explicit pymdp adapter surface."""

import importlib.util

from blockference.agent import BlockferenceAgent


def test_adapter_is_the_only_package_agent_symbol():
    import blockference.agent as module

    assert module.__all__ == ["BlockferenceAgent"]
    assert not hasattr(module, "Agent")
    assert BlockferenceAgent.__name__ == "BlockferenceAgent"


def test_optional_adapter_fails_clearly_without_extra():
    if importlib.util.find_spec("pymdp") is None:
        import pytest

        with pytest.raises(ImportError, match=r"active-blockference\[pymdp\]"):
            BlockferenceAgent(None, None)
    else:
        assert BlockferenceAgent.__mro__[1].__name__ == "Agent"
