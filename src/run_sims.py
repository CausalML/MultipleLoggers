import argparse
import time
import yaml
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings(action="ignore", category=ConvergenceWarning)

from data import load_datasets, generate_bandit_feedback
from ope import (
    calc_ipw,
    calc_weighted_ipw,
    calc_dr,
    estimate_q_func,
    estimate_q_func_with_mrdr,
    calc_ground_truth,
)
from policy import train_policies


def calc_rmse(policy_value_true, policy_value_estimated) -> float:
    return np.sqrt(
        (((policy_value_true - policy_value_estimated) / policy_value_true) ** 2).mean()
    )


with open("./conf/policy_params.yaml", "rb") as f:
    policy_params = yaml.safe_load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_sims", "-n", type=int, default=200,
    )
    parser.add_argument("--data", "-d", type=str, required=True)
    args = parser.parse_args()
    print(args)

    # configurations
    num_sims = args.num_sims
    data = args.data
    np.random.seed(12345)
    ratio_list = [0.25]
    estimator_names = [
        "ground_truth",
        "IPS",
        "BAL",
        "WEI",
        "DR-naive",
        "DR",
        "MRDR-naive",
        "MRDR-stratified",
    ]
    log_path = Path("../log") / data
    log_path.mkdir(parents=True, exist_ok=True)

    ope_results = {name: np.zeros(num_sims) for name in estimator_names}
    rel_rmse_results = {
        name: {r: np.zeros(num_sims) for r in ratio_list} for name in estimator_names
    }
    for ratio in ratio_list:
        data_dict = load_datasets(data=data, ratio=ratio, is_iid=False)
        start = time.time()
        for sim_id in np.arange(num_sims):
            # train eval and two behavior policies
            pi_e, pi_b1, pi_b2 = train_policies(
                data_dict=data_dict, policy_params=policy_params
            )
            # generate bandit feedback
            bandit_feedback = generate_bandit_feedback(
                data_dict=data_dict, pi_b1=pi_b1, pi_b2=pi_b2
            )
            # estimate q-function with cross-fitting
            estimated_q_func = estimate_q_func(bandit_feedback=bandit_feedback)
            estimated_q_func_with_mrdr_naive = estimate_q_func_with_mrdr(
                bandit_feedback=bandit_feedback, pi_e=pi_e, fitting_method="naive",
            )
            estimated_q_func_with_mrdr_stratified = estimate_q_func_with_mrdr(
                bandit_feedback=bandit_feedback, pi_e=pi_e, fitting_method="stratified",
            )
            # off-policy evaluation
            ope_results["ground_truth"][sim_id] = calc_ground_truth(
                y_true=data_dict["y_ev"], pi=pi_e
            )
            ope_results["IPS"][sim_id] = calc_ipw(
                rewards=bandit_feedback["rewards"],
                actions=bandit_feedback["actions"],
                pi_b=bandit_feedback["pi_b"],
                pi_e=pi_e,
            )
            ope_results["BAL"][sim_id] = calc_ipw(
                rewards=bandit_feedback["rewards"],
                actions=bandit_feedback["actions"],
                pi_b=bandit_feedback["pi_b_star"],
                pi_e=pi_e,
            )
            ope_results["WEI"][sim_id] = calc_weighted_ipw(
                rewards=bandit_feedback["rewards"],
                actions=bandit_feedback["actions"],
                idx1=bandit_feedback["idx1"],
                pi_b=bandit_feedback["pi_b"],
                pi_e=pi_e,
            )
            ope_results["DR-naive"][sim_id] = calc_dr(
                rewards=bandit_feedback["rewards"],
                actions=bandit_feedback["actions"],
                estimated_q_func=estimated_q_func,
                pi_b=bandit_feedback["pi_b"],
                pi_e=pi_e,
            )
            ope_results["DR"][sim_id] = calc_dr(
                rewards=bandit_feedback["rewards"],
                actions=bandit_feedback["actions"],
                estimated_q_func=estimated_q_func,
                pi_b=bandit_feedback["pi_b_star"],
                pi_e=pi_e,
            )
            ope_results["MRDR-naive"][sim_id] = calc_dr(
                rewards=bandit_feedback["rewards"],
                actions=bandit_feedback["actions"],
                estimated_q_func=estimated_q_func_with_mrdr_naive,
                pi_b=bandit_feedback["pi_b_star"],
                pi_e=pi_e,
            )
            ope_results["MRDR-stratified"][sim_id] = calc_dr(
                rewards=bandit_feedback["rewards"],
                actions=bandit_feedback["actions"],
                estimated_q_func=estimated_q_func_with_mrdr_stratified,
                pi_b=bandit_feedback["pi_b_star"],
                pi_e=pi_e,
            )
            if ((sim_id + 1) % 20) == 0:
                print(
                    f"ratio={ratio}-{sim_id+1}th: {np.round((time.time() - start) / 60, 2)}min"
                )
        for estimator in estimator_names:
            rel_rmse_results[estimator][ratio] = calc_rmse(
                policy_value_true=ope_results["ground_truth"],
                policy_value_estimated=ope_results[estimator],
            )
        print(f"finish ratio={ratio}: {np.round((time.time() - start) / 60, 2)}min")
        print("=" * 50)

    # save results of the evaluation of OPE
    rel_rmse_results_df = pd.DataFrame(rel_rmse_results).drop("ground_truth", 1)
    rel_rmse_results_df.T.round(5).to_csv(log_path / f"rel_rmse.csv")
    # visualize results (IPS-BAL-WEI-DR)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(
        hue="event",
        style="event",
        markers=True,
        markersize=15,
        linewidth=5.0,
        data=rel_rmse_results_df[["IPS", "BAL", "WEI", "DR"]],
    )
    plt.rcParams["font.family"] = "sans-serif"
    plt.xlabel(r"stratum size ratio ($r = n_1 / n_2$)", fontsize=20)
    plt.ylabel(r"$relative-RMSE(\hat{J})$", fontsize=20)
    plt.yticks(fontsize=15)
    plt.ylim(0.0, 1.0)
    plt.xticks(fontsize=15)
    plt.xscale("log")
    # ax.set_xticks(np.arange(0, 4.1))
    plt.legend(bbox_to_anchor=(1.001, 0.999), loc="upper left", fontsize=20)
    fig.savefig(log_path / f"rel_rmse.png", bbox_inches="tight", pad_inches=0.05)
    # visualize results (among DRs)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(
        hue="event",
        style="event",
        markers=True,
        markersize=15,
        linewidth=5.0,
        data=rel_rmse_results_df[["DR-naive", "DR", "MRDR-naive", "MRDR-stratified"]],
    )
    plt.rcParams["font.family"] = "sans-serif"
    plt.xlabel(r"stratum size ratio ($r = n_1 / n_2$)", fontsize=20)
    plt.ylabel(r"$relative-RMSE(\hat{J})$", fontsize=20)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.xscale("log")
    # ax.set_xticks(np.arange(0, 4.1))
    plt.legend(bbox_to_anchor=(1.001, 0.999), loc="upper left", fontsize=20)
    fig.savefig(log_path / f"rel_rmse_drs.png", bbox_inches="tight", pad_inches=0.05)
