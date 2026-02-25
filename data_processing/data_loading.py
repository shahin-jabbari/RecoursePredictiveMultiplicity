import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Union
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from typing import Union
from collections import defaultdict
import re

def load_polish() -> Tuple[np.ndarray, np.ndarray]:
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

def load_compas() -> Tuple[np.ndarray, np.ndarray]:
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
