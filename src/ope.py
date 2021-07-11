from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import ops

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression


def calc_ground_truth(y_true: np.ndarray, pi: np.ndarray) -> float:
    """Calculate the ground-truth policy value of an eval policy"""
    return pi[np.arange(y_true.shape[0]), y_true].mean()


def calc_ipw(
    rewards: np.ndarray,
    actions: np.ndarray,
    pi_b: np.ndarray,
    pi_e: np.ndarray,
) -> float:
    n_data = actions.shape[0]
    iw = pi_e[np.arange(n_data), actions] / pi_b[np.arange(n_data), actions]
    return (rewards * iw).mean()


def calc_var(
    rewards: np.ndarray,
    actions: np.ndarray,
    pi_b: np.ndarray,
    pi_e: np.ndarray,
    estimated_q_func: np.ndarray,
):
    n_data = actions.shape[0]
    v = np.average(estimated_q_func, weights=pi_e, axis=1)
    shifted_rewards = rewards - estimated_q_func[np.arange(n_data), actions]
    iw = pi_e[np.arange(n_data), actions] / pi_b[np.arange(n_data), actions]
    return np.var(shifted_rewards * iw + v)


def calc_weighted(
    rewards: np.ndarray,
    actions: np.ndarray,
    idx1: np.ndarray,
    pi_b: np.ndarray,
    pi_e: np.ndarray,
    estimated_q_func: np.ndarray = None,
    n_fold: int = 2,
) -> float:
    estimated_rewards_list = list()
    if estimated_q_func is None:
        estimated_q_func = np.zeros((actions.shape[0], np.int(actions.max() + 1)))
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=12345)
    for train_idx, test_idx in kf.split(rewards):
        rewards_tr, rewards_ev = rewards[train_idx], rewards[test_idx]
        actions_tr, actions_ev = actions[train_idx], actions[test_idx]
        idx1_tr, idx1_ev = idx1[train_idx], idx1[test_idx]
        pi_b_tr, pi_b_ev = pi_b[train_idx], pi_b[test_idx]
        pi_e_tr, pi_e_ev = pi_e[train_idx], pi_e[test_idx]
        estimated_q_func_tr = estimated_q_func[train_idx]
        estimated_q_func_ev = estimated_q_func[test_idx]
        # estimate lambda with one of the fold
        n_data1, n_data2 = idx1_tr.sum(), (~idx1_tr).sum()
        var1 = calc_var(
            rewards=rewards_tr[idx1_tr],
            actions=actions_tr[idx1_tr],
            pi_b=pi_b_tr[idx1_tr],
            pi_e=pi_e_tr[idx1_tr],
            estimated_q_func=estimated_q_func_tr[idx1_tr],
        )
        var2 = calc_var(
            rewards=rewards_tr[~idx1_tr],
            actions=actions_tr[~idx1_tr],
            pi_b=pi_b_tr[~idx1_tr],
            pi_e=pi_e_tr[~idx1_tr],
            estimated_q_func=estimated_q_func_tr[~idx1_tr],
        )
        denominator = (n_data1 / var1) + (n_data2 / var2)
        lam1 = (n_data1 / var1) / denominator
        lam2 = (n_data2 / var2) / denominator
        # estimate the policy value with the other fold
        iw1 = (
            pi_e_ev[idx1_ev, actions_ev[idx1_ev]]
            / pi_b_ev[idx1_ev, actions_ev[idx1_ev]]
        )
        iw2 = (
            pi_e_ev[~idx1_ev, actions_ev[~idx1_ev]]
            / pi_b_ev[~idx1_ev, actions_ev[~idx1_ev]]
        )
        v1 = np.average(estimated_q_func_ev[idx1_ev], weights=pi_e_ev[idx1_ev], axis=1)
        v2 = np.average(
            estimated_q_func_ev[~idx1_ev], weights=pi_e_ev[~idx1_ev], axis=1
        )
        shifted_rewards1 = (
            rewards_ev[idx1_ev] - estimated_q_func_ev[idx1_ev, actions_ev[idx1_ev]]
        )
        shifted_rewards2 = (
            rewards_ev[~idx1_ev] - estimated_q_func_ev[~idx1_ev, actions_ev[~idx1_ev]]
        )
        estimated_rewards = lam1 * (iw1 * shifted_rewards1 + v1).mean()
        estimated_rewards += lam2 * (iw2 * shifted_rewards2 + v2).mean()
        estimated_rewards_list.append(estimated_rewards)
    return np.mean(estimated_rewards_list)


def calc_dr(
    rewards: np.ndarray,
    actions: np.ndarray,
    estimated_q_func: np.ndarray,
    pi_b: np.ndarray,
    pi_e: np.ndarray,
) -> float:
    n_data = actions.shape[0]
    v = np.average(estimated_q_func, weights=pi_e, axis=1)
    iw = pi_e[np.arange(n_data), actions] / pi_b[np.arange(n_data), actions]
    shifted_rewards = rewards - estimated_q_func[np.arange(n_data), actions]
    return (iw * shifted_rewards + v).mean()


def estimate_q_func(
    bandit_feedback,
    pi_e: np.ndarray,
    fitting_method: str = "naive",
    k_fold: int = 2,
) -> np.ndarray:
    X = bandit_feedback["X_ev"]
    y = bandit_feedback["rewards"]
    pi_b_star = bandit_feedback["pi_b_star"]
    idx1 = bandit_feedback["idx1"].astype(int)
    a = pd.get_dummies(bandit_feedback["actions"]).values
    skf = StratifiedKFold(n_splits=k_fold)
    skf.get_n_splits(X, y)
    estimated_q_func = np.zeros((bandit_feedback["n_eval"], bandit_feedback["n_class"]))
    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_ev = X[train_idx], X[test_idx]
        y_tr, a_tr = y[train_idx], a[train_idx].astype(float)
        pi_e_tr = pi_e[train_idx]
        pi_b_star_tr = pi_b_star[train_idx]
        idx1_tr = idx1[train_idx]
        ops.reset_default_graph()
        clf = QFuncEstimator(
            num_features=X_tr.shape[1],
            num_classes=bandit_feedback["n_class"],
            fitting_method=fitting_method,
        )
        clf.train(
            X=X_tr,
            a=a_tr,
            y=y_tr,
            pi_e=pi_e_tr,
            pi_b_star=pi_b_star_tr,
            idx1=idx1_tr,
        )
        for a_idx in np.arange(bandit_feedback["n_class"]):
            estimated_q_func_for_a = clf.predict(X=X_ev, a_idx=a_idx)[:, a_idx]
            estimated_q_func[test_idx, a_idx] = estimated_q_func_for_a
        clf.s.close()
    return estimated_q_func


@dataclass
class QFuncEstimator:
    num_features: int
    num_classes: int
    eta: float = 0.01
    std: float = 0.01
    lam: float = 0.001
    batch_size: int = 256
    epochs: int = 20
    fitting_method: str = "stratified"

    def __post_init__(self) -> None:
        """Initialize Class."""
        tf.set_random_seed(0)
        self.s = tf.Session()
        self.create_placeholders()
        self.build_graph()
        self.create_losses()
        self.add_optimizer()

    def create_placeholders(self) -> None:
        """Create the placeholders to be used."""
        self.input_X = tf.placeholder(
            "float32", shape=(None, self.num_features), name="input_X"
        )
        self.input_A = tf.placeholder(
            "float32", shape=(None, self.num_classes), name="input_A"
        )
        self.input_R = tf.placeholder("float32", shape=(None,), name="input_R")
        self.input_pi_e = tf.placeholder(
            "float32", shape=(None, self.num_classes), name="input_pi_e"
        )
        self.input_pi_b_star = tf.placeholder(
            "float32", shape=(None, self.num_classes), name="input_pi_b_star"
        )
        self.input_idx1 = tf.placeholder("float32", shape=(None,), name="input_idx1")

    def build_graph(self) -> None:
        """Build the main tensorflow graph with embedding layers."""
        self.weights = tf.Variable(
            tf.random_normal(
                [self.num_features + self.num_classes, self.num_classes],
                stddev=self.std,
            )
        )
        self.bias = tf.Variable(tf.random_normal([self.num_classes], stddev=self.std))

        with tf.variable_scope("prediction"):
            input_X = tf.concat([self.input_X, self.input_A], axis=1)
            self.preds = tf.sigmoid(tf.matmul(input_X, self.weights) + self.bias)

    def create_losses(self) -> None:
        """Create the losses."""
        with tf.name_scope("loss"):
            shifted_rewards = self.input_R - tf.reduce_sum(
                self.preds * self.input_A, axis=1
            )
            if self.fitting_method == "normal":
                self.loss = tf.reduce_mean(tf.square(shifted_rewards))
            else:
                ratio1 = tf.reduce_mean(self.input_idx1)
                input_idx2 = tf.ones_like(self.input_idx1) - self.input_idx1
                ratio2 = tf.reduce_mean(input_idx2)
                pi_e = tf.reduce_sum(self.input_pi_e * self.input_A, 1)
                pi_b_star = tf.reduce_sum(self.input_pi_b_star * self.input_A, 1)
                v = tf.reduce_sum(self.input_pi_e * self.preds, 1)
                phi = (pi_e / pi_b_star) * shifted_rewards + v
                phi1 = self.input_idx1 * phi
                phi2 = input_idx2 * phi
                if self.fitting_method == "stratified":
                    self.loss = ratio1 * tf.reduce_mean(tf.square(phi1))
                    self.loss += ratio2 * tf.reduce_mean(tf.square(phi2))
                    self.loss -= ratio1 * tf.square(tf.reduce_mean(phi1))
                    self.loss -= ratio2 * tf.square(tf.reduce_mean(phi2))
                elif self.fitting_method == "naive":
                    self.loss = tf.reduce_mean(tf.square(phi))
                    self.loss -= tf.square(tf.reduce_mean(phi))

            self.var_list = [self.weights, self.bias]
            l2_reg = [tf.nn.l2_loss(v) for v in self.var_list]
            self.loss += self.lam * tf.add_n(l2_reg)

    def add_optimizer(self) -> None:
        """Add the required optimizer to the graph."""
        with tf.name_scope("optimizer"):
            self.apply_grads = tf.train.MomentumOptimizer(
                learning_rate=self.eta, momentum=0.8
            ).minimize(self.loss, var_list=self.var_list)

    def train(
        self,
        X: np.ndarray,
        a: np.ndarray,
        y: np.ndarray,
        pi_e: np.ndarray,
        pi_b_star: np.ndarray,
        idx1: np.ndarray,
    ) -> None:
        self.s.run(tf.global_variables_initializer())
        for _ in np.arange(self.epochs):
            arr = np.arange(X.shape[0])
            np.random.shuffle(arr)
            for idx in np.arange(0, X.shape[0], self.batch_size):
                arr_ = arr[idx : idx + self.batch_size]
                self.s.run(
                    self.apply_grads,
                    feed_dict={
                        self.input_X: X[arr_],
                        self.input_A: a[arr_],
                        self.input_R: y[arr_],
                        self.input_pi_e: pi_e[arr_],
                        self.input_pi_b_star: pi_b_star[arr_],
                        self.input_idx1: idx1[arr_],
                    },
                )

    def predict(self, X: np.ndarray, a_idx: int):
        a_ = np.zeros((X.shape[0], self.num_classes))
        a_[:, a_idx] = 1
        return self.s.run(self.preds, feed_dict={self.input_X: X, self.input_A: a_})


def estimate_pi_b(bandit_feedback, k_fold: int = 2) -> None:
    X = bandit_feedback["X_ev"]
    idx1 = bandit_feedback["idx1"]
    a = bandit_feedback["actions"]
    skf = StratifiedKFold(n_splits=k_fold, shuffle=True)
    skf.get_n_splits(X, a)
    estimated_pi_b1 = np.zeros((bandit_feedback["n_eval"], bandit_feedback["n_class"]))
    estimated_pi_b2 = np.zeros((bandit_feedback["n_eval"], bandit_feedback["n_class"]))
    estimated_pi_b_star = np.zeros(
        (bandit_feedback["n_eval"], bandit_feedback["n_class"])
    )
    for train_idx, test_idx in skf.split(X, a):
        X_tr, X_ev = X[train_idx], X[test_idx]
        idx1_tr, a_tr = idx1[train_idx], a[train_idx]
        clf = LogisticRegression(random_state=12345)
        clf.fit(X=X_tr[idx1_tr], y=a_tr[idx1_tr])
        estimated_pi_b1[test_idx, :] = clf.predict_proba(X_ev)
        clf = LogisticRegression(random_state=12345)
        clf.fit(X=X_tr[~idx1_tr], y=a_tr[~idx1_tr])
        estimated_pi_b2[test_idx, :] = clf.predict_proba(X_ev)
        clf = LogisticRegression(random_state=12345)
        clf.fit(X=X_tr, y=a_tr)
        estimated_pi_b_star[test_idx, :] = clf.predict_proba(X_ev)
    idx1 = np.expand_dims(idx1.astype(int), 1)
    bandit_feedback["pi_b"] = np.clip(
        idx1 * estimated_pi_b1 + (1 - idx1) * estimated_pi_b2, 1e-6, 1.0
    )
    bandit_feedback["pi_b_star"] = np.clip(estimated_pi_b_star, 1e-6, 1.0)
    return bandit_feedback
