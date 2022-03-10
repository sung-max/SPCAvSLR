import sys
import math
import random
import numpy as np
import scipy as sc
from sklearn import linear_model
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pylab

from algos import test_diagonal, test_Q, test_CT, test_T, test_TPower, test_GPower, rescale_variance

'''
	Compare performance of DT, Q, CT at while varying k.

'''

NUM_TRIALS = 5

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> PARAMETERS AND SETTINGS <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< #
theta = 4.0       # SNR
rescaled = False  # whether cov matrix has been rescaled to correlation matrix
slrType = 'lasso'
sign_vs_unif = True

print(f'theta: {theta} rescaled: {rescaled} sign_vs_unif: {sign_vs_unif}')

ax = plt.subplot(111)
ax.set_ylim([-0.3, 1.3])

# use same marker, color for the same algorithm for consistency
markers = {'DT': 'D', 'Q': 'o', 'CT': 'x', 'TP': 's', 'GP': '4'}
colors = {'DT': 'g', 'Q': 'r', 'CT': 'b', 'TP': 'c', 'GP': 'm'}

for d in [625]:#, 400, 500, 600]:
	Qs = []      # list of Q's while varying k (see below for definition of Q)
	Ds = []      # '' '' Diagonal thresholding
	Cs = []      # '' ''
	Ts = []
	TPs = []
	GPs = []
	n = d

	xs = np.linspace(0.30, 3.0, 10)       # x := k / sqrt(n)

	for x in xs:
		k = int(x * math.sqrt(n))

		Q = 0.    # fraction of support recovered for Q, summed over trials
		D = 0.    # '' ''  Diagonal thresholding
		C = 0.    # '' ''  Covariance thresholding
		T = 0.    # '' ''  Correlation test
		TP = 0.   # '' ''  Truncated Power
		GP = 0.    # '' ''  Generalized Power
		print(f'n: {n} d: {d} k: {k}')

		for t in range(NUM_TRIALS):
			print('generating data...')
			rand_dir = None  # random k-dim direction
			# Option A: random signs
			if sign_vs_unif:
				rand_dir = np.array([(1. if random.random() > 0.5 else -1.) for _ in range(k)]).reshape((1,k))
			# Option B: unif random from sphere
			else:
				rand_dir = np.random.multivariate_normal([0] * k, np.identity(k), 1)

			# Option 1: plant spike at {0,...,k-1}
            # TODO: use a flag rather than commenting...
			support = None
			pad = np.array([0] * (d-k)).reshape((1,d-k))
			spike = np.hstack((rand_dir, pad))   # pad it to make it d-dim
			spike /= np.linalg.norm(spike)

			'''
			# Option 2: plant spike at random subset of size k
			support = np.random.choice(range(d), size=k, replace=False)
			spike = np.array([0.] * d).reshape((1,d))
			for i in range(k):
				spike[0][support[i]] = k_unif[0][i]
			spike /= np.linalg.norm(spike)
			'''

			mean = np.array([0] * d)  # zero mean
			cov_1 = np.identity(d) + theta * spike * spike.T # spiked covariance
			if rescaled:
				rescale_variance(cov_1, theta, spike, k)
			samples1 = np.random.multivariate_normal(mean, cov_1, n)

			print('testing D...')
			D += test_diagonal(samples1, n, d, k, theta, supp=support)
			#Q += test_Q(samples1, n, d, k, theta, spike, supp=support, slr=slrType)
			#C += test_CT(samples1, n, d, k, theta, supp=support, tau=4., mu=1.)
			print('testing T...')
			TP += test_TPower(samples1, n, d, k, theta, supp=support)
			#GP += test_GPower(samples1, n, d, k, theta, supp=support)
			#T += test_T(samples1, n, d, k, theta, supp=support)
			print(D, Q, C, TP, GP)

		print(D / NUM_TRIALS, Q / NUM_TRIALS, C / NUM_TRIALS, TP / NUM_TRIALS, GP / NUM_TRIALS)
		Ds.append(D / NUM_TRIALS)
		Qs.append(Q / NUM_TRIALS)
		Cs.append(C / NUM_TRIALS)
		TPs.append(TP / NUM_TRIALS)
		GPs.append(GP / NUM_TRIALS)

	ax.plot(xs, Ds, markers['DT'], c=colors['DT'], label='DT', linestyle='-')
	#ax.plot(xs, Qs, markers['Q'], c=colors['Q'], label='Q', linestyle='-')
	#ax.plot(xs, Cs, markers['CT'], c=colors['CT'], label='CT', linestyle='-')
	ax.plot(xs, TPs, markers['TP'], c=colors['TP'], label='TPower', linestyle='-')
	#ax.plot(xs, GPs, markers['GP'], c=colors['GP'], label='GPower', linestyle='-')

plt.legend(loc='best')
plt.xlabel('$k/\sqrt{n}$')
plt.ylabel('Fraction of support recovered')
plt.savefig('graphs/{}-d{}-t{}-{}-{}-{}.png'.format(slrType, d, theta,
	('rescaled' if rescaled else 'nr'), ('sign' if sign_vs_unif else 'unif'),
	str(np.random.rand())[2:6]))
plt.show()
