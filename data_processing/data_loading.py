"""This code is from the following paper: https://arxiv.org/abs/2602.07674"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Union
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from typing import Union
from collections import defaultdict
import re

def load_polish_notebook() -> Tuple[np.ndarray, np.ndarray]:
    nparr = pd.read_csv("datasets/polish-companies_clean_uncut.csv").values
    np.random.shuffle(nparr)
    X_all, y_all = nparr[:, 1:65].astype(np.float32), nparr[:, 65]
    classes, min_cnt = np.unique(y_all, return_counts=True)
    min_cnt = int(min_cnt.min())
    X, y = [], []
    for c in classes:
        idx = np.random.choice(np.where(y_all == c)[0], size=min_cnt, replace=False)
        X.append(X_all[idx]); y.append(y_all[idx])
    X, y = np.vstack(X), np.concatenate(y)
    rng = np.random.permutation(len(y))
    X, y = X[rng], y[rng]
    return X, y

def load_compas_notebook() -> Tuple[np.ndarray, np.ndarray]:
    nparr = pd.read_csv("datasets/compas.csv").values
    np.random.shuffle(nparr)
    n = nparr.shape[1]
    X_all, y_all = nparr[:, 1:n].astype(np.float32), nparr[:, 0]
    classes, min_cnt = np.unique(y_all, return_counts=True)
    min_cnt = int(min_cnt.min())
    X, y = [], []
    for c in classes:
        idx = np.random.choice(np.where(y_all == c)[0], size=min_cnt, replace=False)
        X.append(X_all[idx]); y.append(y_all[idx])
    X, y = np.vstack(X), np.concatenate(y)
    rng = np.random.permutation(len(y))
    X, y = X[rng], y[rng]
    return X, y

def load_germanc_notebook(
    csv_path: Union[str, Path] = "datasets/germanc.csv",
    random_state: int = 42,
):
    df = (
        pd.read_csv(csv_path)
        .drop(columns=[c for c in ["Unnamed: 0"] if c in pd.read_csv(csv_path).columns])
        .apply(pd.to_numeric, errors="coerce")
        .dropna(axis=0, how="any")
    )

    dummy_groups = defaultdict(list)
    pattern = re.compile(r"(.+)_\w+$")

    for col in df.columns:
        m = pattern.match(col)
        if m:
            dummy_groups[m.group(1)].append(col)

    # cols_to_drop = []
    # for group_cols in dummy_groups.values():
    #     if len(group_cols) > 1 and (df[group_cols].sum(axis=1) == 1).all():
    #         cols_to_drop.append(group_cols[0])  # drop the first dummy

    y_all = df["y"].astype(int)
    X_all = df.drop(columns=["y"])

    rng = np.random.RandomState(random_state)
    min_cnt = y_all.value_counts().min()
    balanced_idx = np.concatenate(
        [rng.choice(np.where(y_all == cls)[0], size=min_cnt, replace=False)
         for cls in y_all.unique()]
    )
    rng.shuffle(balanced_idx)

    return (
        X_all.iloc[balanced_idx].to_numpy(dtype=np.float32),
        y_all.iloc[balanced_idx].to_numpy(dtype=int),
    )
