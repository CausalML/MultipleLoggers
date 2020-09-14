from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression


def train_policies(data_dict: Dict, policy_params: Dict) -> List[np.ndarray]:
    """Train evaluation and behavior policies."""
    policy_list = list()
    for pol in policy_params.keys():
        # make label predictions
        base = policy_params[pol]["base_model"]
        X_tr, y_tr = data_dict[f"X_tr{base}"], data_dict[f"y_tr"]
        clf = LogisticRegression(
            random_state=0, max_iter=1000, solver="lbfgs", multi_class="multinomial"
        ).fit(X=X_tr, y=y_tr)
        preds = clf.predict(X=data_dict[f"X_ev{base}"]).astype(int)
        # transform predictions into distribution over actions
        alpha = policy_params[pol]["alpha"]
        beta = policy_params[pol]["beta"]
        pi = np.zeros((data_dict["n_eval"], data_dict["n_class"]))
        u = np.random.uniform(-0.5, 0.5)
        pi[:, :] = (1 - alpha - beta * u) / (data_dict["n_class"] - 1)
        pi[np.arange(data_dict["n_eval"]), preds] = alpha + beta * u
        policy_list.append(pi)
    return policy_list
