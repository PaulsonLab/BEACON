# BEACON: A Bayesian Optimization Inspired Strategy for Efficient Novelty Search
This repo contains the codes for [BEACON: A Bayesian Optimization Inspired
Strategy for Efficient Novelty Search](https://arxiv.org/abs/2406.03616)

# Installation
```sh
pip install -r requirements.txt
```

# Usage
We provide the code scripts for executing BEACON on different problem setting. Noted that BEACON requires the usage of [ThompsonSampling.py](https://github.com/PaulsonLab/BEACON/blob/main/src/ThompsonSampling.py) file to perform efficient Thompson sampling strategy proposed in [this work](https://arxiv.org/abs/2002.09309).

# Extension to high-dimensional problem
We propose an extension algorithm TR-BEACON for addressing high-dimensional novelty search problem.
Please refer to the [paper](https://openreview.net/pdf?id=9Xo6ONB8E3) and [github repo](https://github.com/PaulsonLab/TR-BEACON).
