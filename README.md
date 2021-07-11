# Optimal Off-Policy Evaluation from Multiple Logging Policies

## Overview
This repository contains the code for replicating the experiments of the paper
**"Optimal Off-Policy Evaluation from Multiple Logging Policies" (ICML2021)**

If you find this code useful in your research then please cite:
```
@inproceedings{kallus2021optimal,
  title={Optimal Off-Policy Evaluation from Multiple Logging Policies},
  author={Kallus, Nathan and Saito, Yuta and Uehara, Masatoshi},
  booktitle = {Proceedings of the 38th International Conference on Machine Learning},
  pages={5247-5256},
  year={2021},
  volume = {139},
  publisher={PMLR},
}
```

## Dependencies
- python==3.7.3
- numpy==1.18.1
- pandas==0.25.1
- scikit-learn==0.23.1
- tensorflow==1.15.4
- pyyaml==5.1
- seaborn==0.10.1
- matplotlib==3.2.2

### Running the code

To run the simulations with the multi-class classification datasets, run the following commands in the `./src/` directory:

```
for data in optdigits pendigits sat letter
do
    screen python run_sims.py --num_sims 200 --data $data -i
done
```

Nota that the configurations used in the experiments can be found in `./conf/policy_params.yaml`.
Once the simulations have finished running, the summarized results can be found in the `../log/{data}` directory for each data.
