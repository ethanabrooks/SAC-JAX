import pandas as pd
import numpy as np

"""
Created on Tue Sep 11 17:00:29 2018

@author: Mohammad Doosti Lakhani
"""

# Importing dataset
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

# Implementing Upper Bound Confidence
m, n = dataset.shape

ad_selected = np.zeros(m)  # selected ads by users

N = np.zeros(n, dtype=np.float16)  # number of selection of ad i
R = np.zeros(n, dtype=np.float16)  # sum of reward of ad i

total_reward = 0

# implementation in vectorized form
for i in range(0, m):
    r = R / N
    delta = np.sqrt(3 / 2 * np.log(i + 1) / N)
    upper_bound = r + delta
    max_index = np.argmax(upper_bound)
    ad_selected[i] = max_index
    N[max_index] += 1
    reward = dataset.values[i, max_index]
    R[max_index] += reward
    print(r.mean())
    total_reward += reward

# Visualizing selections
# plt.hist(ad_selected)
# plt.show()
