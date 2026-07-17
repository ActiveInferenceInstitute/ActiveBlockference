"""Active Inference adapter for the upstream pymdp agent."""

from __future__ import annotations

from pymdp.agent import Agent as PymdpAgent


class BlockferenceAgent(PymdpAgent):
    """Thin extension hook over :class:`pymdp.agent.Agent`."""

    def __init__(self, A, B, **kwargs):
        super().__init__(A, B, **kwargs)


__all__ = ["BlockferenceAgent"]
