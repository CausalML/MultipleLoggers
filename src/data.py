from typing import Dict
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def load_datasets(
    data: str, ratio: float, test_size: float = 0.7, is_iid: bool = False,
):
    """Load and preprocess raw multiclass classification data."""
    data_path = Path(f"../data/{data}")
    if data == "optdigits":
        data_ = np.r_[
            np.loadtxt(data_path / f"{data}.tra", delimiter=","),
            np.loadtxt(data_path / f"{data}.tes", delimiter=","),
        ]
    elif data == "pageblock":
        data_ = np.genfromtxt(
            data_path / "page-blocks.data", delimiter="", dtype="str"
        ).astype(float)
        data_[:, -1] = data_[:, -1] - 1
    elif data == "pendigits":
        data_ = np.r_[
            np.loadtxt(data_path / f"{data}.tra", delimiter=","),
            np.loadtxt(data_path / f"{data}.tes", delimiter=","),
        ]
    elif data == "sat":
        data_ = np.r_[
            np.loadtxt(data_path / f"{data}.trn", delimiter=" "),
            np.loadtxt(data_path / f"{data}.tst", delimiter=" "),
        ]
        data_[:, -1] = np.where(data_[:, -1] == 7, 5, data_[:, -1] - 1)
    data_tr, data_ev = train_test_split(data_, test_size=test_size, random_state=12345)
    n_train, n_eval = data_tr.shape[0], data_ev.shape[0]
    n_dim = np.int(data_tr.shape[1] / 2)
    y_tr, y_ev = data_tr[:, -1].astype(int), data_ev[:, -1].astype(int)
    n_class = np.unique(y_tr).shape[0]
    y_full_ev = np.zeros((n_eval, n_class))
    y_full_ev[np.arange(n_eval), y_ev] = 1
    X_tr, X_ev = data_tr[:, :-1], data_ev[:, :-1]
    X_tr1, X_tr2 = data_tr[:, :n_dim], data_tr[:, n_dim:]
    X_ev1, X_ev2 = data_ev[:, :n_dim], data_ev[:, n_dim:]

    # multiple logger index generation
    ratio1 = ratio / (1 + ratio)
    n_eval1 = np.int(n_eval * ratio1)
    if is_iid:
        idx1 = np.random.binomial(1, ratio1, size=n_eval).astype(bool)
    else:
        idx1 = np.ones(n_eval, dtype=bool)
        idx1[n_eval1:] = False

    return dict(
        n_train=n_train,
        n_eval=n_eval,
        n_dim=n_dim,
        n_class=n_class,
        n_behavior_policies=2,
        X_tr=X_tr,
        X_tr1=X_tr1,
        X_tr2=X_tr2,
        X_ev=X_ev,
        X_ev1=X_ev1,
        X_ev2=X_ev2,
        y_tr=y_tr,
        y_ev=y_ev,
        y_full_ev=y_full_ev,
        idx1=idx1,
        ratio1=(n_eval1 / n_eval),
    )


def generate_bandit_feedback(data_dict: Dict, pi_b1: np.ndarray, pi_b2: np.ndarray):
    """Generate logged bandit feedback data."""
    n_eval = data_dict["n_eval"]
    idx1, ratio1 = data_dict["idx1"], data_dict["ratio1"]
    idx1_expanded = np.expand_dims(idx1, 1)
    pi_b = pi_b1 * idx1_expanded + pi_b2 * (1 - idx1_expanded)
    pi_b_star = pi_b1 * ratio1 + pi_b2 * (1.0 - ratio1)
    action_set = np.arange(data_dict["n_class"])
    actions = np.zeros(data_dict["n_eval"], dtype=int)
    for i, pvals in enumerate(pi_b):
        actions[i] = np.random.choice(action_set, p=pvals)
    rewards = data_dict["y_full_ev"][np.arange(n_eval), actions]
    return dict(
        n_eval=data_dict["n_eval"],
        n_class=data_dict["n_class"],
        X_ev=data_dict["X_ev"],
        pi_b=pi_b,
        pi_b_star=pi_b_star,
        actions=actions,
        idx1=idx1,
        rewards=rewards,
    )
