import numpy as np
import pandas as pd

"""
Question 2b
For the range of tangency portfolios given by rf = 50bps; 100bps; 150bps; 175bps optimal compute
allocations (ready formula) and σΠ. Present results in a table.
"""
# check
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
rf_list = [0.0, 0.005, 0.01, 0.015, 0.0175]

A = np.dot(np.dot(ones, InverseSigma), ones)
B = np.dot(np.dot(u, InverseSigma), ones)
C = np.dot(np.dot(u, InverseSigma), u)


def tangency_portfolio(A, B, C, rf, InverseSigma, ones):
    tp_rtn = (C - (B * rf)) / (B - (A * rf))
    tp_std = np.sqrt((C - 2 * B * rf + A * rf ** 2) / (B - A * rf) ** 2)
    tp_weight = (np.dot(InverseSigma, (u - rf * ones))) / (B - A * rf)
    return tp_rtn, tp_std, tp_weight


columns = ['Risk free rate', 'return', 'variance', 'weight A', 'weight B', 'weight C', 'weight D']
data = []
for r in rf_list:
    tp = tangency_portfolio(A, B, C, r, InverseSigma, ones)
    print('-' * 60)
    print('Rf:', r)
    print('return:', tp[0])
    print('variance:', tp[1])
    print('weights:', tp[2])
    a = [[r, tp[0], tp[1], tp[2][0], tp[2][1], tp[2][2], tp[2][3]]]
    data = data + a

table = pd.DataFrame(data=data, columns=columns)
print(table)
table.to_csv('table_2b.csv', index=False)
