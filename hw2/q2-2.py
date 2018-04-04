import numpy as np
import matplotlib
import math
matplotlib.use('Agg')

import matplotlib.pyplot as plt

np.random.seed(1337)
N = 10000

# generate samples
Z = np.random.normal(0, 1, (2, N))
plt.subplot(131)
plt.scatter(Z[0], Z[1], 0.1)
plt.xlim(-6, 6)
plt.ylim(-6, 6)

# SVD
Q = np.array([[1, 1], [1, -1]]) / (2**0.5)
D = np.array([[1.9, 0], [0, 0.1]])

# make X
S = Q.dot(D**0.5)
X = S.dot(Z)
plt.subplot(132)
plt.scatter(X[0], X[1], 0.1)
plt.xlim(-6, 6)
plt.ylim(-6, 6)

# using Numpy's implementation
Y = np.random.multivariate_normal(np.zeros(2), [[1.0, 0.9], [0.9, 1.0]], N).transpose()
plt.subplot(133)
plt.scatter(Y[0], Y[1], 0.1)
plt.xlim(-6, 6)
plt.ylim(-6, 6)

plt.savefig('q2-2.png')
