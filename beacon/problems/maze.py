"""Deceptive point-maze navigation (paper Sec. 5.4).

A linear policy with 8 parameters in ``[-1, 1]`` steers a point mass in
Gymnasium-Robotics' ``PointMaze_Large-v3`` for at most 300 steps. The
behaviour (outcome) is the final ``(x, y)`` position; BEACON never sees the
reward. Performance is reported as the best reward found so far,
``1 - ||final - goal|| / ||start - goal||`` (1 when the goal is reached).

Requires the optional dependencies ``gymnasium`` and ``gymnasium-robotics``.
"""

from __future__ import annotations

import numpy as np
import torch

from beacon.device import DTYPE
from beacon.metrics import BestValue, GridPartition
from beacon.problems.base import ContinuousProblem


def _make_env():
    try:
        import gymnasium as gym
        import gymnasium_robotics
    except ImportError as exc:
        raise ImportError("The maze benchmark needs `pip install gymnasium gymnasium-robotics`.") from exc
    gym.register_envs(gymnasium_robotics)
    return gym.make("PointMaze_Large-v3", continuing_task=False)


GOAL_CELL = np.array([5, 2])
RESET_CELL = np.array([7, 4])
ENV_SEED = 10
MAX_STEPS = 300


def _policy(param: np.ndarray, state: np.ndarray):
    p1 = -1 + 2 / (1 + np.exp(-(param[0:4] @ state[0:4])))
    p2 = -1 + 2 / (1 + np.exp(-(param[4:8] @ state[0:4])))
    return np.array([p1, p2], dtype=np.float64)


class MazeSimulator:
    """Runs the maze rollout for a batch of policy parameter vectors."""

    def __init__(self):
        self.env = _make_env()
        observation, _ = self._reset()
        self.goal = np.asarray(observation["desired_goal"], dtype=np.float64)
        self.initial_distance = float(np.linalg.norm(observation["achieved_goal"] - observation["desired_goal"]))

    def _reset(self):
        return self.env.reset(seed=ENV_SEED, options={"goal_cell": GOAL_CELL, "reset_cell": RESET_CELL})

    def rollout(self, param: np.ndarray) -> np.ndarray:
        observation, _ = self._reset()
        for _ in range(MAX_STEPS):
            observation, _, terminated, truncated, _ = self.env.step(_policy(param, observation["observation"]))
            if terminated or truncated:
                if terminated:
                    return self.goal.copy()
                break
        return np.asarray(observation["achieved_goal"], dtype=np.float64)

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        params = X.detach().cpu().numpy()
        final = np.stack([self.rollout(p) for p in params])
        return torch.as_tensor(final, dtype=DTYPE, device=X.device)

    def reward(self, Y: torch.Tensor) -> torch.Tensor:
        goal = torch.as_tensor(self.goal, dtype=Y.dtype, device=Y.device)
        return 1.0 - torch.linalg.norm(Y - goal, dim=-1) / self.initial_distance


class MazeProblem(ContinuousProblem):
    """Maze problem; the initial design keeps only points with reward < 0.9."""

    def initial_data(self, n_init: int, seed: int, device=None):
        X = self.initial_design(n_init, seed).to(device)
        Y, _ = self.evaluate(X)
        keep = self.simulator.reward(Y) < 0.9  # do not start from (near-)solutions
        return X[keep], Y[keep], Y[keep]


def make_maze(**_) -> MazeProblem:
    simulator = MazeSimulator()
    # Guidance regions for UG-BEACON: 10 x 10 grid over the maze's position range.
    partition = GridPartition([-6.0, -4.5], [6.0, 4.5], 10)
    problem = MazeProblem(
        name="maze",
        objective=simulator,
        input_dim=8,
        output_dim=2,
        lower=-1.0,
        upper=1.0,
        metric=BestValue(simulator.reward),
        partition=partition,
        defaults=dict(n_init=10, iterations=200, replicates=20, k=10, kernel="rbf", likelihood="gamma", archive="observed"),
        description="Point-maze navigation with a linear policy (8 params); metric: best reward.",
        metric_name="best_reward",
    )
    problem.simulator = simulator
    return problem
