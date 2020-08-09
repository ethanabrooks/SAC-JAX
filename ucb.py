from pathlib import Path

import numpy as np
import pandas as pd
from gym.utils import seeding

"""
Created on Tue Sep 11 17:00:29 2018

@author: Mohammad Doosti Lakhani
"""


class UCB:
    def __init__(self, seed=0):
        self.random, _ = seeding.np_random(seed)

    def argmax(self, x: np.ndarray):
        return int(self.random.choice(np.arange(x.size)[x == x.max()]))

    def train_loop(self, dataset: np.ndarray, selections=None, rewards=None):
        # Implementing Upper Bound Confidence
        m, n = dataset.shape

        if selections is None:
            selections = np.zeros(
                (m + 1, n), dtype=np.float16
            )  # number of selection of ad i
        if rewards is None:
            rewards = np.zeros((m + 1, n), dtype=np.float16)  # sum of reward of ad i

        # first sample each
        selections[:n] = np.tril(np.ones((n, n)))
        # noinspection PyTypeChecker
        initial_rewards = np.diag(dataset)
        # noinspection PyTypeChecker
        np.fill_diagonal(rewards, initial_rewards)
        rewards[:n] = np.cumsum(rewards[:n], axis=0)
        c = 1
        for i, r in enumerate(initial_rewards):
            c = yield i, r

        # implementation in vectorized form
        for i, (N0, R0, N1, R1, data), in enumerate(
            zip(
                selections[n - 1 :],
                rewards[n - 1 :],
                selections[n:],
                rewards[n:],
                dataset,
            ),
        ):
            r = R0 / N0
            if c is None:
                c = 1
            delta = np.sqrt(3 / 2 * np.log(i + 1) / N0)
            upper_bound = r + c * delta
            choice = self.argmax(upper_bound)
            N1[:] = N0
            N1[choice] += 1
            R1[:] = R0
            reward = data[choice]
            R1[choice] += reward
            c = yield choice, reward


def main():
    dataset = pd.read_csv("Ads_CTR_Optimisation.csv")
    selections, rewards = zip(*UCB().train_loop(dataset))
    for r in rewards:
        print(r)

    with Path("results.npz").open("wb") as f:
        np.savez(f, selections=selections, rewards=rewards)


if __name__ == "__main__":
    main()
