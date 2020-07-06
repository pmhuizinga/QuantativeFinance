import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns;

sns.set()
"""
Question 2c
Plot the true efficient frontier in the presence of risk-free earning asset for rf = 100bps; 175bps and
specifically identify its shape..
"""
# # # check
# u = np.array([0.05, 0.07, 0.15, 0.27])
# s = np.array([0.07, 0.12, 0.30, 0.60])
# R = np.array([[1, 0.8, 0.5, 0.4],
#               [0.8, 1, 0.7, 0.5],
#               [0.5, 0.7, 1, 0.8],
#               [0.4, 0.5, 0.8, 1]])

u = np.array([0.02, 0.07, 0.15, 0.20])
s = np.array([0.05, 0.12, 0.17, 0.25])
R = np.array([[1, 0.3, 0.3, 0.3],
              [0.3, 1, 0.6, 0.6],
              [0.3, 0.6, 1, 0.6],
              [0.3, 0.6, 0.6, 1]])

ones = np.ones(len(s))
S = np.diag(s)
Sigma = np.dot(np.dot(S, R), S)
InverseSigma = np.linalg.inv(Sigma)
rf_list = [0.01, 0.0175]
rf = 0.01

A = np.dot(np.dot(ones, InverseSigma), ones)
B = np.dot(np.dot(u, InverseSigma), ones)
C = np.dot(np.dot(u, InverseSigma), u)
D = (A * C) - (B ** 2)

# minimum variance portfolio
min_var_port_mu = B / A
min_var_port_w = np.dot(InverseSigma, ones) / A
min_var_port_std = np.sqrt(np.dot(np.dot(min_var_port_w.T, Sigma), min_var_port_w))

# efficient frontier
list_of_mu = np.arange(-0.2, 0.5, 0.0001)
list_of_std = np.sqrt(((A * list_of_mu ** 2) - 2 * B * list_of_mu + C) / D)

# true efficient frontier (all returns >= minimum variance portfolio return)
a = zip(list_of_std, list_of_mu)
b = list([(x, y) for x, y in a if y >= min_var_port_mu])
true_list_of_std, true_list_of_mu = zip(*b)


# tangency portfolio
def tangency_portfolio(A, B, C, rf, ones):
    tp_mu = (C - (B * rf)) / (B - (A * rf))
    tp_std = np.sqrt((C - 2 * B * rf + A * rf ** 2) / (B - A * rf) ** 2)
    tp_w = (np.dot(InverseSigma, (u - rf * ones))) / (B - A * rf)
    return tp_mu, tp_std, tp_w

# plot
for rf in rf_list:
    tp_mu, tp_std, tp_w = tangency_portfolio(A, B, C, rf, ones)
    plt.scatter(list_of_std, list_of_mu, label='efficient frontier', s=5, c='grey')
    plt.scatter(true_list_of_std, true_list_of_mu, label='real efficient frontier', s=5, c='darkblue')
    plt.scatter(min_var_port_std, min_var_port_mu, s=50, color='red', label='minimum variance portfolio, rtn: {}%'.format(round(min_var_port_mu*100,3)))
    plt.scatter(tp_std, tp_mu, s=50, color='green', label='tangency portfolio')
    plt.title('Tangency portfolio with risk free rate {}%'.format(round(rf * 100, 2)))
    plt.xlabel('Risk')
    plt.ylabel('Return')
    plt.legend()
    plt.savefig('ef_rf_{}.png'.format(rf))
    plt.show()
