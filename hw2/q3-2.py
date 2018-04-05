import numpy as np
import matplotlib
import math
matplotlib.use('Agg')

import matplotlib.pyplot as plt

np.random.seed(1337)

# constants
n = 1000
d = 100
lmb = 0.1
step_size = 0.01

# initial values
x = np.vstack([np.random.normal(0.1, 1, (n//2, d)),
    np.random.normal(-0.1, 1, (n//2, d))])
y = np.hstack([np.ones(n//2), -1.*np.ones(n//2)])
w0 = np.random.normal(0, 1, d)

# helper functions
def grad(w):
    dv = np.zeros(d)
    for i in range(n):
        if y[i] * (w.dot(x[i])) < 1:
            dv += -y[i] * x[i]
    return (dv/n) + (lmb*w)

def f(w):
    v = 0
    for i in range(n):
        v += max(0, 1 - w.dot(x[i]))
    return (v/n) + lmb / 2 + np.linalg.norm(w)

def accuracy(w):
    ret = 0
    for i in range(n):
        cur = w.dot(x[i])
        if cur > 0 and y[i] > 0:
            ret += 1
        elif cur < 0 and y[i] < 0:
            ret += 1
    return ret / n

# plot data
fv = []
ac = []

for _ in range(100):
    grad_w = grad(w0)

    w0 += -step_size * grad_w
    fv.append(f(w0))
    ac.append(accuracy(w0))

plt.plot(fv)
plt.savefig('q3-2-fv.png')

plt.clf()

plt.plot(ac)
plt.savefig('q3-2-ac.png')
