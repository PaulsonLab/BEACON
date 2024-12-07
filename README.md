# BEACON: A Bayesian Optimization Inspired Strategy for Efficient Novelty Search
This repo contains the codes for [BEACON: A Bayesian Optimization Inspired
Strategy for Efficient Novelty Search](https://arxiv.org/abs/2406.03616)

# Installation
```sh
pip install -r requirements.txt
```

# Usage
We provide the code scripts for executing BEACON on different problem setting. Noted that all code script requires the usage of [ThompsonSampling.py](https://github.com/PaulsonLab/BEACON/blob/1ede361eb98824b459da9df3a17839ab8753d02b/ThompsonSampling.py) file to perform efficient Thompson sampling strategy proposed in [this work](https://arxiv.org/abs/2002.09309).

Running Experiments
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

# Extension to high-dimensional problem
We propose an extension algorithm TR-BEACON for addressing high-dimensional novelty search problem.
Please refer to the [paper](https://openreview.net/pdf?id=9Xo6ONB8E3) and [github repo](https://github.com/PaulsonLab/TR-BEACON).
