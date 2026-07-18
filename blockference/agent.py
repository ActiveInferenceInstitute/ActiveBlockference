"""Active Inference adapter for the upstream pymdp agent."""

from __future__ import annotations

from typing import Any

try:
    from pymdp.agent import Agent as PymdpAgent
except ImportError as exc:  # pragma: no cover - exercised in a slim install
    _PYMDP_IMPORT_ERROR = exc

    class BlockferenceAgent:
        """Informative placeholder when the optional pymdp extra is absent."""

        def __init__(self, A: Any, B: Any, **kwargs: Any) -> None:
            del A, B, kwargs
            raise ImportError(
                "BlockferenceAgent requires the optional pymdp dependency; "
                "install active-blockference[pymdp]"
            ) from _PYMDP_IMPORT_ERROR

else:

    class BlockferenceAgent(PymdpAgent):  # type: ignore[no-redef]
        """Thin extension hook over :class:`pymdp.agent.Agent`."""

        def __init__(self, A: Any, B: Any, **kwargs: Any) -> None:
            super().__init__(A, B, **kwargs)


__all__ = ["BlockferenceAgent"]
