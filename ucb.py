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

    def train_loop(self, dataset: np.ndarray):
        # Implementing Upper Bound Confidence
        T, n, d = dataset.shape

        choices = np.ones((n, d), dtype=np.float16)  # number of selection of ad i
        rewards = np.zeros((n, d))
        arange = np.arange(n)

        # first sample each
        c = None
        for i in range(d):
            rewards[:, i] = dataset[i, :, i]
            c = yield i * np.ones(n), dataset[i, :, i]

        # implementation in vectorized form
        for i, data, in enumerate(dataset[d:], start=d):
            r = rewards / choices
            if c > 1:
                rewards = np.zeros((n, d))
                arange = np.arange(n)
                choice = np.random.choice(d, size=n)
            else:
                delta = (np.log(i + 1) / choices) ** (1 / 2)  # TODO
                upper_bound = r + c * delta
                choice = np.argmax(upper_bound, axis=-1)
            choices[arange, choice] += 1
            reward = data[arange, choice]
            rewards[arange, choice] += reward
            c = yield choice, reward


def main():
    # dataset = pd.read_csv("Ads_CTR_Optimisation.csv")
    choices = 5
    batches = 2
    size = (batches, choices)
    means = np.random.normal(size=size, scale=1)
    # stds = abs(np.random.normal(size=size, scale=0.01))
    T = 10

    loc = np.tile(means, (T, 1, 1))
    # scale = np.tile(stds, (T, 1, 1))

    # dataset = np.random.binomial(choices, p)
    dataset = np.random.normal(loc)
    choices, rewards = [np.stack(x) for x in zip(*UCB().train_loop(dataset))]
    import ipdb

    ipdb.set_trace()

    with Path("results.npz").open("wb") as f:
        np.savez(f, selections=choices, rewards=rewards)


if __name__ == "__main__":
    main()
