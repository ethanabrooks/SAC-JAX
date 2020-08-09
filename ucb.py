import numpy as np
import pandas as pd

"""
Created on Tue Sep 11 17:00:29 2018

@author: Mohammad Doosti Lakhani
"""


def argmax(x: np.ndarray):
    return int(np.random.choice(np.arange(x.size)[x == x.max(initial=-np.inf)]))


def main():
    # Importing dataset
    dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

    # Implementing Upper Bound Confidence
    m, n = dataset.shape

    selections = np.zeros((m + 1, n), dtype=np.float16)  # number of selection of ad i
    rewards = np.zeros((m + 1, n), dtype=np.float16)  # sum of reward of ad i

    # first sample each
    selections[:n] = np.tril(np.ones((n, n)))
    # # noinspection PyTypeChecker
    np.fill_diagonal(rewards, np.diag(dataset.values))
    rewards[:n] = np.cumsum(rewards[:n], axis=0)

    # implementation in vectorized form
    for i, (N0, R0, N1, R1), in enumerate(
        zip(selections[n - 1 :], rewards[n - 1 :], selections[n:], rewards[n:])
    ):
        r = R0 / N0
        delta = np.sqrt(3 / 2 * np.log(i + 1) / N0)
        upper_bound = r + delta
        max_index = argmax(upper_bound)
        N1[:] = N0
        N1[max_index] += 1
        reward = dataset.values[i, max_index]
        R1[:] = R0
        R1[max_index] += reward
        print(r[max_index])

    # Visualizing selections
    # plt.hist(ad_selected)
    # plt.show()


if __name__ == "__main__":
    main()
