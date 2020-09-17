## Efficient Evaluation Using Logged Bandit Feedback from Multiple Loggers

## Overview
This repository contains the code for replicating the experiments from the paper **"Efficient Evaluation Using Logged Bandit Feedback from Multiple Loggers"**

If you find this code useful in your research then please cite:
```
#TODO: add bibtex
```

## Dependencies
- python==3.7.3
- numpy==1.18.1
- pandas==0.25.1
- scikit-learn==0.23.1
- tensorflow==1.15.2
- pyyaml==5.1
- seaborn==0.10.1
- matplotlib==3.2.2

### Running the code

To run simulations with the multi-class classification datasets conducted in Section 5, run the following commands in the `./src/` directory:

- stratified case (OPE from multiple loggers)
```
for data in pendigits
do
    python run_sims.py --num_sims 10 --data $data
done
```

Nota that the configurations used in the experiments can be found in `./conf/policy_params.yaml`.
Once the simulations have finished running, the summarized results can be found in the `../log/{data}` directory for each dataset.
