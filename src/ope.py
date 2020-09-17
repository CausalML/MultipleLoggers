from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold


def calc_ground_truth(y_true: np.ndarray, pi: np.ndarray) -> float:
    """Calculate the ground-truth policy value of an eval policy"""
    return pi[np.arange(y_true.shape[0]), y_true].mean()


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


def estimate_q_func(bandit_feedback: Dict, k_fold: int = 2,) -> np.ndarray:
    estimated_q_func = np.zeros((bandit_feedback["n_eval"], bandit_feedback["n_class"]))
    X = bandit_feedback["X_ev"]
    y = bandit_feedback["rewards"]
    a = bandit_feedback["actions"]
    skf = StratifiedKFold(n_splits=k_fold)
    skf.get_n_splits(X, y)
    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_ev = X[train_idx], X[test_idx]
        y_tr, a_tr = y[train_idx], a[train_idx]
        for action_ in np.arange(bandit_feedback["n_class"]):
            action_match = a_tr == action_
            clf = MLPClassifier(hidden_layer_sizes=(100, 100), random_state=0,).fit(
                X=X_tr[action_match], y=y_tr[action_match]
            )
            estimated_q_func[test_idx, action_] = clf.predict_proba(X=X_ev)[:, 1]
    return estimated_q_func


def estimate_q_func_with_mrdr(
    bandit_feedback, pi_e: np.ndarray, fitting_method: str = "naive", k_fold: int = 2,
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
        clf = QFuncWithMRDR(
            num_features=X_tr.shape[1],
            num_classes=bandit_feedback["n_class"],
            fitting_method=fitting_method,
        )
        clf.train(
            X=X_tr, a=a_tr, y=y_tr, pi_e=pi_e_tr, pi_b_star=pi_b_star_tr, idx1=idx1_tr,
        )
        estimated_q_func[test_idx, :] = clf.predict(X=X_ev)
        clf.s.close()
    return estimated_q_func


@dataclass
class QFuncWithMRDR:
    num_features: int
    num_classes: int
    num_layers_0: int = 100
    num_layers_1: int = 100
    eta: float = 0.001
    lam: float = 0.0001
    std: float = 0.01
    batch_size: int = 256
    epochs: int = 1000
    fitting_method: str = "stratified"

    def __post_init__(self) -> None:
        """Initialize Class."""
        tf.set_random_seed(0)
        # Build the graphs
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
        # Weights initialized by random normal function
        self.weights_0 = tf.Variable(
            tf.random_normal([self.num_features, self.num_layers_0], stddev=self.std)
        )
        self.bias_0 = tf.Variable(
            tf.random_normal([self.num_layers_0], stddev=self.std)
        )
        self.weights_1 = tf.Variable(
            tf.random_normal([self.num_layers_0, self.num_layers_1], stddev=self.std)
        )
        self.bias_1 = tf.Variable(
            tf.random_normal([self.num_layers_1], stddev=self.std)
        )
        self.weights_2 = tf.Variable(
            tf.random_normal([self.num_layers_1, self.num_classes], stddev=self.std)
        )
        self.bias_2 = tf.Variable(tf.random_normal([self.num_classes], stddev=self.std))

        with tf.variable_scope("prediction"):
            hidden_output_0 = tf.nn.relu(
                tf.matmul(self.input_X, self.weights_0) + self.bias_0
            )
            hidden_output_1 = tf.nn.relu(
                tf.matmul(hidden_output_0, self.weights_1) + self.bias_1
            )
            self.preds = tf.sigmoid(
                tf.matmul(hidden_output_1, self.weights_2) + self.bias_2
            )

    def create_losses(self) -> None:
        """Create the losses."""
        with tf.name_scope("loss"):
            if self.fitting_method == "normal":
                preds_flatten = tf.reduce_sum(self.preds * self.input_A, axis=1)
                self.loss = tf.sigmoid(
                    tf.reduce_mean(tf.square(self.input_R - preds_flatten))
                )

            else:
                ratio1 = tf.reduce_mean(self.input_idx1)
                input_idx2 = tf.ones_like(self.input_idx1) - self.input_idx1
                ratio2 = tf.reduce_mean(input_idx2)
                pi_e = tf.reduce_sum(self.input_pi_e * self.input_A, 1)
                pi_b_star = tf.reduce_sum(self.input_pi_b_star * self.input_A, 1)
                preds_flatten = tf.reduce_sum(self.preds * self.input_A, axis=1)
                shifted_rewards = self.input_R - preds_flatten
                baseline = tf.reduce_mean(
                    tf.reduce_sum(self.input_pi_e * self.preds, axis=1)
                )
                phi = (pi_e / pi_b_star) * shifted_rewards + baseline

                if self.fitting_method == "stratified":
                    self.loss = ratio1 * tf.reduce_mean(
                        tf.square(self.input_idx1 * phi)
                    )
                    self.loss += ratio2 * tf.reduce_mean(tf.square(input_idx2 * phi))
                    self.loss -= ratio1 * tf.square(
                        tf.reduce_mean(self.input_idx1 * phi)
                    )
                    self.loss -= ratio2 * tf.square(tf.reduce_mean(input_idx2 * phi))

                elif self.fitting_method == "naive":
                    self.loss = tf.reduce_mean(tf.square(phi))

            self.var_list = [
                self.weights_0,
                self.weights_1,
                self.weights_2,
                self.bias_0,
                self.bias_1,
                self.bias_2,
            ]
            l2_reg = [tf.nn.l2_loss(v) for v in self.var_list]
            self.loss += self.lam * tf.add_n(l2_reg)

    def add_optimizer(self) -> None:
        """Add the required optimiser to the graph."""
        with tf.name_scope("optimizer"):
            learning_rate = tf.train.exponential_decay(
                self.eta, 0, 5, 0.85, staircase=True
            )
            self.apply_grads = tf.train.AdamOptimizer(
                learning_rate=learning_rate,  # momentum=0.9,
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
        self.s = tf.InteractiveSession()
        self.s.run(tf.global_variables_initializer())
        for _ in np.arange(self.epochs):
            arr = np.arange(X.shape[0])
            np.random.shuffle(arr)
            for index in np.arange(0, X.shape[0], self.batch_size):
                self.s.run(
                    self.apply_grads,
                    feed_dict={
                        self.input_X: X[arr[index : index + self.batch_size]],
                        self.input_A: a[arr[index : index + self.batch_size]],
                        self.input_R: y[arr[index : index + self.batch_size]],
                        self.input_pi_e: pi_e[arr[index : index + self.batch_size]],
                        self.input_pi_b_star: pi_b_star[
                            arr[index : index + self.batch_size]
                        ],
                        self.input_idx1: idx1[arr[index : index + self.batch_size]],
                    },
                )

    def predict(self, X: np.ndarray):
        return self.s.run(self.preds, feed_dict={self.input_X: X})
