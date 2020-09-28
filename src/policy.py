from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression


def train_policies(data_dict: Dict, policy_params: Dict) -> List[np.ndarray]:
    """Train evaluation and behavior policies."""
    policy_list = list()
    for pol in policy_params.keys():
        # make label predictions
        base = policy_params[pol]["base_model"]
        X_tr, y_tr = data_dict[f"X_tr"], data_dict[f"y_tr"]
        clf = LogisticRegression(
            random_state=0, solver="lbfgs", multi_class="multinomial",
        ).fit(X=X_tr, y=y_tr)
        preds = clf.predict(X=data_dict[f"X_ev"]).astype(int)
        # transform predictions into distribution over actions
        alpha = policy_params[pol]["alpha"]
        pi = np.zeros((data_dict["n_eval"], data_dict["n_class"]))
        pi[:, :] = (1.0 - alpha) / data_dict["n_class"]
        pi[np.arange(data_dict["n_eval"]), preds] = (
            alpha + (1.0 - alpha) / data_dict["n_class"]
        )
        policy_list.append(pi)
    return policy_list
