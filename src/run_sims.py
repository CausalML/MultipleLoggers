import argparse
import time
import yaml
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings(action="ignore", category=ConvergenceWarning)

from data import load_datasets, generate_bandit_feedback
from ope import (
    calc_ipw,
    calc_weighted,
    calc_dr,
    estimate_q_func,
    estimate_pi_b,
    calc_ground_truth,
)
from policy import train_policies


def calc_rel_rmse(policy_value_true: float, policy_value_estimated: float) -> float:
    return np.sqrt(
        (((policy_value_true - policy_value_estimated) / policy_value_true) ** 2).mean()
    )


with open("./conf/policy_params.yaml", "rb") as f:
    policy_params = yaml.safe_load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_sims", "-n", type=int, required=True)
    parser.add_argument("--test_size", "-t", type=float, default=0.7, required=True)
    parser.add_argument("--data", "-d", type=str, required=True)
    parser.add_argument("--is_estimate_pi_b", "-i", action="store_true")
    args = parser.parse_args()
    print(args)

    # configurations
    num_sims = args.num_sims
    data = args.data
    test_size = args.test_size
    is_estimate_pi_b = args.is_estimate_pi_b
    np.random.seed(12345)
    ratio_list = [0.1, 0.2, 0.5, 1, 2, 4, 10]
    estimator_names = [
        "ground_truth",
        "IS-Avg",
        "IS",
        "IS-PW(f)",
        "DR-Avg",
        "DR-PW",
        "DR",
        "MRDR",
        "SMRDR",
    ]
    log_path = (
        Path("../log") / data / f"test_size={test_size}" / "estimated_pi_b"
        if is_estimate_pi_b
        else Path("../log") / data / f"test_size={test_size}" / "true_pi_b"
    )
    log_path.mkdir(parents=True, exist_ok=True)
    raw_results_path = log_path / "raw_results"
    raw_results_path.mkdir(parents=True, exist_ok=True)

    rel_rmse_results = {
        name: {r: np.zeros(num_sims) for r in ratio_list} for name in estimator_names
    }
    for ratio in ratio_list:
        start = time.time()
        ope_results = {name: np.zeros(num_sims) for name in estimator_names}
        for sim_id in np.arange(num_sims):
            # load and split data
            data_dict = load_datasets(
                data=data, test_size=test_size, ratio=ratio, random_state=sim_id
            )
            # train eval and two behavior policies
            pi_e, pi_b1, pi_b2 = train_policies(
                data_dict=data_dict,
                policy_params=policy_params,
                random_state=sim_id,
            )
            # generate bandit feedback
            bandit_feedback_ = generate_bandit_feedback(
                data_dict=data_dict, pi_b1=pi_b1, pi_b2=pi_b2
            )
            # estimate pi_b1, pi_b2, and pi_b_star with 2-fold cross-fitting
            if is_estimate_pi_b:
                bandit_feedback = estimate_pi_b(bandit_feedback=bandit_feedback_)
            else:
                bandit_feedback = bandit_feedback_
            # estimate q-function with 2-fold cross-fitting
            estimated_q_func = estimate_q_func(
                bandit_feedback=bandit_feedback,
                pi_e=pi_e,
                fitting_method="normal",
            )
            estimated_q_func_with_mrdr_wrong = estimate_q_func(
                bandit_feedback=bandit_feedback,
                pi_e=pi_e,
                fitting_method="naive",
            )
            estimated_q_func_with_mrdr = estimate_q_func(
                bandit_feedback=bandit_feedback,
                pi_e=pi_e,
                fitting_method="stratified",
            )
            # off-policy evaluation
            ope_results["ground_truth"][sim_id] = calc_ground_truth(
                y_true=data_dict["y_ev"], pi=pi_e
            )
            ope_results["IS-Avg"][sim_id] = calc_ipw(
                rewards=bandit_feedback["rewards"],
                actions=bandit_feedback["actions"],
                pi_b=bandit_feedback["pi_b"],
                pi_e=pi_e,
            )
            ope_results["IS"][sim_id] = calc_ipw(
                rewards=bandit_feedback["rewards"],
                actions=bandit_feedback["actions"],
                pi_b=bandit_feedback["pi_b_star"],
                pi_e=pi_e,
            )
            ope_results["IS-PW(f)"][sim_id] = calc_weighted(
                rewards=bandit_feedback["rewards"],
                actions=bandit_feedback["actions"],
                idx1=bandit_feedback["idx1"],
                pi_b=bandit_feedback["pi_b"],
                pi_e=pi_e,
            )
            ope_results["DR-Avg"][sim_id] = calc_dr(
                rewards=bandit_feedback["rewards"],
                actions=bandit_feedback["actions"],
                estimated_q_func=estimated_q_func,
                pi_b=bandit_feedback["pi_b"],
                pi_e=pi_e,
            )
            ope_results["DR-PW"][sim_id] = calc_weighted(
                rewards=bandit_feedback["rewards"],
                actions=bandit_feedback["actions"],
                idx1=bandit_feedback["idx1"],
                pi_b=bandit_feedback["pi_b"],
                pi_e=pi_e,
                estimated_q_func=estimated_q_func,
            )
            ope_results["DR"][sim_id] = calc_dr(
                rewards=bandit_feedback["rewards"],
                actions=bandit_feedback["actions"],
                estimated_q_func=estimated_q_func,
                pi_b=bandit_feedback["pi_b_star"],
                pi_e=pi_e,
            )
            ope_results["MRDR"][sim_id] = calc_dr(
                rewards=bandit_feedback["rewards"],
                actions=bandit_feedback["actions"],
                estimated_q_func=estimated_q_func_with_mrdr_wrong,
                pi_b=bandit_feedback["pi_b_star"],
                pi_e=pi_e,
            )
            ope_results["SMRDR"][sim_id] = calc_dr(
                rewards=bandit_feedback["rewards"],
                actions=bandit_feedback["actions"],
                estimated_q_func=estimated_q_func_with_mrdr,
                pi_b=bandit_feedback["pi_b_star"],
                pi_e=pi_e,
            )
            if ((sim_id + 1) % 20) == 0:
                print(
                    f"ratio={ratio}-{sim_id+1}th: {np.round((time.time() - start) / 60, 2)}min"
                )
        # save raw off-policy evaluation results.
        with open(raw_results_path / f"ratio={ratio}.pkl", mode="wb") as f:
            pickle.dump(ope_results, f)
        for estimator in estimator_names:
            rel_rmse_results[estimator][ratio] = calc_rel_rmse(
                policy_value_true=ope_results["ground_truth"],
                policy_value_estimated=ope_results[estimator],
            )
        print(f"finish ratio={ratio}: {np.round((time.time() - start) / 60, 2)}min")
        print("=" * 50)

    # save results of the evaluation of OPE
    rel_rmse_results_df = pd.DataFrame(rel_rmse_results).drop("ground_truth", 1)
    rel_rmse_results_df.T.round(5).to_csv(log_path / f"rel_rmse.csv")
