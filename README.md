# BEACON: A Bayesian Optimization Inspired Strategy for Efficient Novelty Search
This repo contains the codes for [BEACON: A Bayesian Optimization Inspired
Strategy for Efficient Novelty Search](https://arxiv.org/abs/2406.03616)

# Installation
```sh
pip install -r requirements.txt
```

# Usage
We provide the code scripts for executing BEACON on different problem setting. Each folder corresponds to specific problem setting (continuous/discrete search space and single/multi outcome space). All benchmark problems studied in the main paper section is included (Ackley/Rosenbrock/StyblinskiTang, Material Discovery, Moleculer Discovery, Maze, and MNIST). 

Noted that BEACON requires the usage of [ThompsonSampling.py](https://github.com/PaulsonLab/BEACON/blob/main/src/ThompsonSampling.py) file to perform efficient Thompson sampling strategy proposed in [this work](https://arxiv.org/abs/2002.09309).

# Extension to high-dimensional problem
We propose an extension algorithm TR-BEACON for addressing high-dimensional novelty search problem.
Please refer to the [paper](https://openreview.net/pdf?id=9Xo6ONB8E3) and [github repo](https://github.com/PaulsonLab/TR-BEACON).

## Citation
If you use this code in your research, please cite the following paper:

```
@article{tang2024beacon,
  title={Beacon: A bayesian optimization strategy for novelty search in expensive black-box systems},
  author={Tang, Wei-Ting and Chakrabarty, Ankush and Paulson, Joel A},
  journal={arXiv preprint arXiv:2406.03616},
  year={2024}
}
```
