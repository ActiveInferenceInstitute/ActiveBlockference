"""Tests for the explicit pymdp adapter surface."""

from blockference.agent import BlockferenceAgent


def test_adapter_is_the_only_package_agent_symbol():
    import blockference.agent as module

    assert module.__all__ == ["BlockferenceAgent"]
    assert not hasattr(module, "Agent")
    assert BlockferenceAgent.__name__ == "BlockferenceAgent"
