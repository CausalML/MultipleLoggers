# Optimal Off-Policy Evaluation from Multiple Logging Policies

## Overview
This repository contains the code for replicating the experiments from the paper
[**"Optimal Off-Policy Evaluation from Multiple Logging Policies"**](https://arxiv.org/abs/2010.11002)

If you find this code useful in your research then please cite:
```
@article{kallus2020optimal,
  title={Optimal Off-Policy Evaluation from Multiple Logging Policies},
  author={Kallus, Nathan and Saito, Yuta and Uehara, Masatoshi},
  journal={arXiv preprint arXiv:2010.11002},
  year={2020}
}
```

## Dependencies
- python==3.7.3
- numpy==1.18.1
- pandas==0.25.1
- scikit-learn==0.23.1
- tensorflow==1.15.4
- pyyaml==5.1

### Running the code

Please download the corresponding datasets from the [uci repository](https://archive.ics.uci.edu/ml/datasets.php) and put them in the `./data/` directory.
Then, run the following commands in the `./src/` directory:

```
for data in optdigits pendigits sat letter
do
    python run_sims.py --num_sims 200 --data $data --is_estimate_pi_b
done
```

Nota that the configurations used in the experiments can be found in `./conf/policy_params.yaml`.
The summarized results can be found in the `../log/{data}` directory for each data after running the simulations.
