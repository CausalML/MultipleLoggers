from typing import Dict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def calc_ipw(
    rewards: np.ndarray, actions: np.ndarray, pi_b: np.ndarray, pi_e: np.ndarray,
) -> float:
    n_data = actions.shape[0]
    iw = pi_b[np.arange(n_data), actions] / pi_e[np.arange(n_data), actions]
    return (rewards * iw).mean()


def calc_sigma(
    rewards: np.ndarray, actions: np.ndarray, pi_b: np.ndarray, pi_e: np.ndarray,
):
    n_data = actions.shape[0]
    iw = pi_b[np.arange(n_data), actions] / pi_e[np.arange(n_data), actions]
    return np.var(rewards * iw)


def calc_weighted_ipw(
    rewards: np.ndarray,
    actions: np.ndarray,
    idx1: np.ndarray,
    pi_b: np.ndarray,
    pi_e: np.ndarray,
) -> float:
    n_data1, n_data2 = idx1.sum(), (~idx1).sum()
    sigma1 = calc_sigma(
        rewards=rewards[idx1], actions=actions[idx1], pi_b=pi_b[idx1], pi_e=pi_e[idx1],
    )
    sigma2 = calc_sigma(
        rewards=rewards[~idx1],
        actions=actions[~idx1],
        pi_b=pi_b[~idx1],
        pi_e=pi_e[~idx1],
    )
    lam1 = (sigma1 * ((n_data1 / sigma1) + (n_data2 / sigma2))) ** (-1)
    lam2 = (sigma2 * ((n_data1 / sigma1) + (n_data2 / sigma2))) ** (-1)
    iw1 = pi_b[idx1, actions[idx1]] / pi_e[idx1, actions[idx1]]
    iw2 = pi_b[~idx1, actions[~idx1]] / pi_e[~idx1, actions[~idx1]]
    estimated_rewards = lam1 * (iw1 * rewards[idx1]).sum()
    estimated_rewards += lam2 * (iw2 * rewards[~idx1]).sum()
    return estimated_rewards


def calc_dr(
    rewards: np.ndarray,
    actions: np.ndarray,
    estimated_q_func: np.ndarray,
    pi_b: np.ndarray,
    pi_e: np.ndarray,
) -> float:
    n_data = actions.shape[0]
    baseline = np.average(estimated_q_func, weights=pi_e)
    iw = pi_b[np.arange(n_data), actions] / pi_e[np.arange(n_data), actions]
    shifted_rewards = rewards - estimated_q_func[np.arange(n_data), actions]
    return baseline + (iw * shifted_rewards).mean()


def estimate_q_func(
    bandit_feedback: Dict, k_fold: int = 2, mrdr: bool = False,
) -> np.ndarray:
    estimated_q_func = np.zeros((bandit_feedback["n_eval"], bandit_feedback["n_class"]))
    X = bandit_feedback["X_ev"]
    y = bandit_feedback["rewards"]
    a = bandit_feedback["actions"]
    skf = StratifiedKFold(n_splits=k_fold)
    skf.get_n_splits(X, y)
    for train_ind, test_ind in skf.split(X, y):
        X_tr, X_ev = X[train_ind], X[test_ind]
        y_tr, a_tr = y[train_ind], a[train_ind]
        for action_ in np.arange(bandit_feedback["n_class"]):
            action_match = a_tr == action_
            sample_weight = np.ones(action_match.sum())
            if mrdr:
                pass  # TODO: implement mrdr weighting
            clf = LogisticRegression(
                random_state=0,
                max_iter=1000,
                solver="lbfgs",
                multi_class="multinomial",
            ).fit(
                X=X_tr[action_match], y=y_tr[action_match], sample_weight=sample_weight
            )
            estimated_q_func[test_ind, action_] = clf.predict_proba(X=X_ev)[:, 1]
    return estimated_q_func


def calc_ground_truth(y_true: np.ndarray, pi: np.ndarray) -> float:
    """Calculate the ground-truth policy value of an eval policy"""
    return pi[np.arange(y_true.shape[0]), y_true].mean()
