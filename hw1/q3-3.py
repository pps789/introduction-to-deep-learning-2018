import numpy as np
import matplotlib
import math
matplotlib.use('Agg')

import matplotlib.pyplot as plt

def inv_F(y):
    return -math.log(1-y)

def F(x):
    return 1-math.exp(-x)

N = 1000000

# samples!
data = np.random.uniform(0,1,N)
samples = list(map(inv_F, data))
H_sample, X_sample = np.histogram(samples, bins = 500, normed = True)
dx_sample = X_sample[1] - X_sample[0]
C_sample = np.cumsum(H_sample) * dx_sample
plt.bar(X_sample[1:], C_sample, label='samples', width=np.diff(X_sample), linewidth=0)

# get exact values!
X_exacts = np.arange(0, X_sample[-1], 0.001)
Y_exacts = list(map(F, X_exacts))
plt.plot(X_exacts, Y_exacts, label='exacts', color='red')

plt.legend(['samples', 'exacts'])

plt.savefig('q3-3.png')
