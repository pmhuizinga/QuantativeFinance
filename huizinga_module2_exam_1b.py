import numpy as np
import seaborn as sns; sns.set()

"""
Question 1b
Compute the allocations w∗ and portfolio risk σΠ = pw0Σw, for m = 4:5%. Stress the correlation
matrix: multiply all correlations by ×1:25 and ×1:5, and compute the respective optimal allocations
and portfolio risk (the same m = 4:5%)
"""
# Variables
u = np.array([0.02, 0.07, 0.15, 0.20])
s = np.array([0.05, 0.12, 0.17, 0.25])
R = np.array([[1, 0.3, 0.3, 0.3],
              [0.3, 1, 0.6, 0.6],
              [0.3, 0.6, 1, 0.6],
              [0.3, 0.6, 0.6, 1]])

m = .045
correlation_shock = [1, 1.25, 1.5]

def allocations(u, s, R, m, correlation_factor=1):

    R_s = correlation_stress(R, correlation_factor)

    ones = np.ones(len(s))
    S = np.diag(s)
    Sigma = np.dot(np.dot(S, R_s), S)
    InverseSigma = np.linalg.inv(Sigma)

    A = np.dot(np.dot(ones, InverseSigma), ones)
    B = np.dot(np.dot(u, InverseSigma), ones)
    C = np.dot(np.dot(u, InverseSigma), u)

    l = (A * m - B) / (A * C - B ** 2)
    g = (C - B * m) / (A * C - B ** 2)
    z = (l * u + g * ones)

    w = np.dot(InverseSigma, z)
    u_port = np.sqrt(np.dot(np.dot(w.T, Sigma), w))

    return w, u_port


def correlation_stress(r, correlation_factor=1):

    r_stressed = r * correlation_factor
    # set diagonal to 1
    for i in range(r_stressed.shape[0]):
        r_stressed[i][i] = 1

    return r_stressed

print('-' * 80)
print('asset returns:', u)
print('asset stdev:  ', s)
print('-' * 80)
for cs in correlation_shock:
    w, up = allocations(u, s, R, m, cs)
    print('correlation_factor:', cs)
    print('portfolio weights:', w)
    print('portfolio risk:', up)
    print('-' * 80)

