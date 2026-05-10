"""Active Inference Agent class — pymdp 1.x compatibility layer.

ActiveBlockference's grid pipeline does **not** require a pymdp ``Agent``;
the per-step inference loop in :mod:`blockference.gridference` and
:mod:`blockference.utils.policy` runs the math directly so we can keep
results as ``numpy.ndarray`` (cadCAD/radCAD do not work well with JAX
tracers).

We still expose :class:`BlockferenceAgent` as a thin extension hook over
:class:`pymdp.agent.Agent` (v1.0.x), so callers who want pymdp's
high-level batch / JAX inference machinery can subclass it without
forking.

Migration notes (pymdp 0.0.x → 1.0.x)
--------------------------------------

The 1.0 ``Agent`` constructor is meaningfully different:

* ``A`` and ``B`` are now **lists** of JAX arrays (one per modality /
  factor), not plain NumPy arrays.
* Many keyword arguments were renamed (e.g. ``inference_algo='VANILLA'``
  is no longer valid; use ``'fpi'``, ``'vmp'``, etc.).
* Returned quantities are JAX arrays — convert with ``np.asarray(...)``
  before passing them to cadCAD state-update functions.

If you only need the math primitives (softmax, EFE, sampling) you do
**not** need pymdp at all — use :mod:`blockference.maths` and
:mod:`blockference.utils.utils`.
"""

from __future__ import annotations

from pymdp.agent import Agent as PymdpAgent


class BlockferenceAgent(PymdpAgent):
    """Thin extension hook over :class:`pymdp.agent.Agent` (v1.0.x).

    All keyword arguments are forwarded verbatim to the upstream
    constructor. See the pymdp 1.0 docs for the full signature, including
    the per-modality ``A`` / ``B`` list contract.
    """

    def __init__(self, A, B, **kwargs):
        super().__init__(A, B, **kwargs)


# Back-compat alias — keep both names exported so older imports do not break.
Agent = BlockferenceAgent

__all__ = ["BlockferenceAgent", "Agent"]
