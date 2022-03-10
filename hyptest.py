import sys
import math
import random
import numpy as np
import scipy as sc
from sklearn import linear_model
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from algos import test_diagonal, test_Q, test_MDP, rescale_variance

########################### EXPERIMENTS ##############################

NUM_TRIALS = 100

# >>> parameters <<< #
n = 200    # samples
d = 500    # dimension
k = 30     # sparsity
theta = 10  # SNR
rescaled = True
slrType = 'lasso'

print(f'n: {n} d: {d} k: {k} t: {theta}')

# draw a random direction from a k-dim isotropic Gaussian
k_unif = np.random.multivariate_normal([0] * k, np.identity(k), 1)
#print k_unif  / np.linalg.norm(k_unif)

pad = np.array([0] * (d-k)).reshape((1,d-k))
spike = np.hstack((k_unif, pad))
spike = spike / np.linalg.norm(spike)

mean = np.array([0] * d)  # zero mean
cov_0 = np.identity(d) # null
cov_1 = np.identity(d) + theta * spike * spike.T # spiked covariance

if rescaled:
	rescale_variance(cov_1, theta, spike, k)

#for i in range(k):
#	cov_1[i][i] = 1

D0 = []    # D statistics for H0 samples
D1 = []    # D statistics for H1 samples
Q0 = []    # Q       "      "
Q1 = []    # Q       "      "
M0 = []
M1 = []
A0 = []
A1 = []
B0 = []
B1 = []

for t in range(NUM_TRIALS):
	print('iter', t)
	samples0 = np.random.multivariate_normal(mean, cov_0, n)
	samples1 = np.random.multivariate_normal(mean, cov_1, n)
	#D0.append(test_diagonal(samples0, n, d, k, theta, hyptest=True))
	#D1.append(test_diagonal(samples1, n, d, k, theta, hyptest=True))
	Q0.append(test_Q(samples0, n, d, k, theta, spike, slr=slrType, hyptest=True))
	Q1.append(test_Q(samples1, n, d, k, theta, spike, slr=slrType, hyptest=True))
	A0.append(test_Q(samples0, n, d, k, theta, spike, slr=slrType, hyptest=True, alpha=0.))
	A1.append(test_Q(samples1, n, d, k, theta, spike, slr=slrType, hyptest=True, alpha=0.))
	B0.append(test_Q(samples0, n, d, k, theta, spike, slr=slrType, hyptest=True, beta=0.))
	B1.append(test_Q(samples1, n, d, k, theta, spike, slr=slrType, hyptest=True, beta=0.))
	#M0.append(test_MDP(samples0, n, d, k, theta))
	#M1.append(test_MDP(samples1, n, d, k, theta))

#print 'Q0 avg', sum(Q0)/len(Q0)
#print 'Q1 avg', sum(Q1)/len(Q1)
#print A0, A1

cnt = 0
#'DT-0', 'DT-1', 'Q-0', 'Q-1',
labels = ['Q-0', 'Q-1', 'A-0', 'A-1', 'B-0', 'B-1']
for DATA in [Q0, Q1, A0, A1, B0, B1]:
	density = gaussian_kde(DATA)
	MIN = min(DATA) - 0.05
	MAX = max(DATA) + 0.05
	xs = np.linspace(MIN, MAX, 200)
	density.covariance_factor = lambda : .25
	density._compute_covariance()
	plt.plot(xs,density(xs), label=labels[cnt])
	cnt += 1

plt.legend()
plt.xlabel('value of statistic')
plt.ylabel('empirical probability density')
plt.savefig('graphs/hyptest.{}.n{}.d{}.k{}.t{}.{}.{}.png'.format(slrType, n, d, k, theta, ('rescaled' if rescaled else ''), str(np.random.rand())))
plt.show()