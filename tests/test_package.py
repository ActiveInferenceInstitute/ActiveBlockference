"""Sanity tests for the package surface."""

import blockference


def test_version_is_string():
    assert isinstance(blockference.__version__, str)
    assert blockference.__version__.count(".") >= 1


def test_public_surface_present():
    for name in (
        "ActiveGridference",
        "BlockferenceAgent",
        "actinf_graph",
        "actinf_planning_single",
        "make_grid",
    ):
        assert hasattr(blockference, name), f"missing public symbol: {name}"


def test_subpackage_imports():
    import blockference.envs  # noqa: F401
    import blockference.simulations  # noqa: F401
    import blockference.utils  # noqa: F401
