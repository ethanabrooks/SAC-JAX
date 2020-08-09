from pathlib import Path

import numpy as np
import pandas as pd

"""
Created on Tue Sep 11 17:00:29 2018

@author: Mohammad Doosti Lakhani
"""


def argmax(x: np.ndarray):
    return int(np.random.choice(np.arange(x.size)[x == x.max(initial=-np.inf)]))


def gen(dataset: pd.DataFrame):
    # Importing dataset

    # Implementing Upper Bound Confidence
    m, n = dataset.shape

    selection = np.zeros((m + 1, n), dtype=np.float16)  # number of selection of ad i
    selections = np.zeros((m + 1, n), dtype=np.float16)  # number of selection of ad i
    rewards = np.zeros((m + 1, n), dtype=np.float16)  # sum of reward of ad i

    # first sample each
    selections[:n] = np.tril(np.ones((n, n)))
    # noinspection PyTypeChecker
    initial_rewards = np.diag(dataset.values)
    # noinspection PyTypeChecker
    np.fill_diagonal(rewards, initial_rewards)
    rewards[:n] = np.cumsum(rewards[:n], axis=0)
    yield from enumerate(initial_rewards)

    # implementation in vectorized form
    for i, (N0, R0, N1, R1, data), in enumerate(
        zip(
            selections[n - 1 :],
            rewards[n - 1 :],
            selections[n:],
            rewards[n:],
            dataset.values,
        ),
    ):
        r = R0 / N0
        delta = np.sqrt(3 / 2 * np.log(i + 1) / N0)
        upper_bound = r + delta
        choice = argmax(upper_bound)
        N1[:] = N0
        N1[choice] += 1
        R1[:] = R0
        reward = data[choice]
        R1[choice] += reward
        yield choice, reward

    # with Path("results.npz").open("wb") as f:
    #     np.savez(f, selections=selections, rewards=rewards)

    # Visualizing selections
    # plt.hist(ad_selected)
    # plt.show()


def main():
    dataset = pd.read_csv("Ads_CTR_Optimisation.csv")
    selections, rewards = zip(*gen(dataset))
    for r in rewards:
        print(r)

    with Path("results.npz").open("wb") as f:
        np.savez(f, selections=selections, rewards=rewards)


if __name__ == "__main__":
    main()
