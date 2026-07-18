"""Mutual-information helpers for analysing simulation traces.

The previous version of this file shipped with broken module-level argparse
code that crashed on import. This rewrite exposes :func:`calculate_mi` as a
plain function and gates all CLI side-effects behind ``if __name__ ==
"__main__"``.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

__all__ = ["calculate_mi"]


def calculate_mi(X: pd.DataFrame, y: pd.Series, *, random_state: int | None = 0) -> pd.Series:
    """Compute mutual information between every column of ``X`` and ``y``.

    Object/categorical columns are factorised before scoring.
    """
    if not isinstance(X, pd.DataFrame) or not isinstance(y, (pd.Series, np.ndarray, list, tuple)):
        raise TypeError("X must be a DataFrame and y must be a one-dimensional sequence")
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows")
    if X.empty or X.shape[1] == 0:
        raise ValueError("X must contain at least one feature column")
    X = X.copy()
    for colname in X.select_dtypes("object").columns:
        X[colname], _ = X[colname].factorize()
    discrete_features = [ptype.kind in ("i", "u", "b") for ptype in X.dtypes]

    mi_scores = mutual_info_regression(
        X.to_numpy(), np.asarray(y), discrete_features=discrete_features, random_state=random_state
    )
    return pd.Series(mi_scores, name="MI Scores", index=X.columns).sort_values(ascending=False)


def _read_table(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute mutual information.")
    parser.add_argument("X", help="CSV file containing the feature matrix.")
    parser.add_argument(
        "--target",
        required=True,
        help="Column name in X to use as the target y.",
    )
    parser.add_argument("--viz", action="store_true", help="Render an MI bar chart with seaborn.")
    args = parser.parse_args(argv)

    df = _read_table(args.X)
    if args.target not in df.columns:
        parser.error(f"target column {args.target!r} not present in {args.X}")
    y = df.pop(args.target)
    scores = calculate_mi(df, y)
    print(scores.to_string())

    if args.viz:
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.barplot(x=scores.values, y=scores.index)
        plt.title("Mutual information")
        plt.tight_layout()
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
