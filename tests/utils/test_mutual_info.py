"""Tests for blockference.utils.mutual_info."""

import numpy as np
import pandas as pd

from blockference.utils.mutual_info import calculate_mi


def test_calculate_mi_returns_series_with_index():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "informative": np.arange(200) + rng.normal(scale=0.01, size=200),
            "noise": rng.normal(size=200),
        }
    )
    y = X["informative"].copy()
    scores = calculate_mi(X, y)
    repeated = calculate_mi(X, y)
    assert isinstance(scores, pd.Series)
    assert scores.equals(repeated)
    assert set(scores.index) == {"informative", "noise"}
    # the informative column must score higher than pure noise
    assert scores["informative"] > scores["noise"]
