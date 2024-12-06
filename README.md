# BEACON: A Bayesian Optimization Inspired Strategy for Efficient Novelty Search
This repo contains the codes for [BEACON: A Bayesian Optimization Inspired
Strategy for Efficient Novelty Search](https://arxiv.org/abs/2406.03616)

# Usage
We provide the code scripts for executing different SOTA novelty search algorithm including BEACON, MaxVar, and NS-EA.

BEACON
------------------------------
Run the following commands to execute BEACON under different problem setting:

**Continuous feature space (e.g. synthetic problem conducted in this paper)**
   
For continuous feature space and single outcome problem:
```sh
python Continuous_SingleOutcome_BEACON.py
```

For continuous feature space and multi outcome problem:
```sh
python Continuous_Multioutcome_BEACON.py
```

**Discrete feature space (e.g. Material and drug case study conducted in this paper)**
   
For discrete feature space and single outcome problem:
```sh
python Discrete_SingleOtcome_BEACON.py
```

For discrete feature space and multi outcome problem:
```sh
python Discrete_MultiOutcome_BEACON.py
```
