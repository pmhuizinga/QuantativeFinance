import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

"""
Question 1c
Inverse optimisation: generate > 2000 random allocations sets n × 1 (not optimal), and plot
the cloud of points of µΠ vertically on σΠ horizontally. How you standardise each set to satisfy
w01 = 1 is up to you. If not coding but using spreadsheets then it is allowed to have ≈ 200 random
allocations sets of low discrepancy, ie this smaller Nsim must reveal a good picture.
"""

# Variables
u = np.array([0.02, 0.07, 0.15, 0.20])
s = np.array([0.05, 0.12, 0.17, 0.25])

nr_of_assets = len(u)
nr_of_random_allocations = 2000
S = np.diag(s)
Sigma = np.dot(np.dot(S, R), S)

# create array with random numbers and rescale to 100%
random_weights = np.random.rand(nr_of_random_allocations * nr_of_assets).reshape(nr_of_random_allocations, nr_of_assets)
random_weights = np.array([x / sum(x) for x in random_weights])

returns = np.dot(random_weights, u)
standard_deviation = np.array([(np.sqrt(np.dot(x.T, np.dot(x, Sigma)))) for x in random_weights])

plt.scatter(standard_deviation, returns, s=30, edgecolors='black')
plt.xlabel('risk')
plt.ylabel('return')
plt.show()
#plt.savefig('mod2_question1c.png')

