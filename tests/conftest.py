"""Shared pytest fixtures."""

import itertools
import random

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _seed_rng():
    """Seed numpy + random before each test for reproducibility."""
    np.random.seed(42)
    random.seed(42)


@pytest.fixture
def grid_3x3():
    """Standard 3x3 grid coordinate list used across tests."""
    return list(itertools.product(range(3), repeat=2))


@pytest.fixture
def grid_4x4():
    return list(itertools.product(range(4), repeat=2))
