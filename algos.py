import math
import sys
import random
import numpy as np
import scipy as sc
from sklearn import linear_model

############################ helper methods  #############################

def equal(a,b):
	return abs(a-b) <= 1e-8

# soft threshold entries of A
def soft_threshold(A, z):
	for i in range(A.shape[0]):
		for j in range(A.shape[1]):
			if A[i][j] > z:
				A[i][j] -= z
			elif A[i][j] < -z:
				A[i][j] += z
			else:
				A[i][j] = 0.

# hard threshold vector v
def hard_threshold(v, z):
	for i in range(v.shape[0]):
		if abs(v[i]) < z:
			v[i] = 0.

def truncate(v, k):
	thres = sorted([abs(v[i]) for i in range(v.shape[0])], reverse=True)[k-1]
	hard_threshold(v, thres)

def fraction_recovered(top_k, k, supp=None):
	num_recovered = 0
	if supp is None:
		for i in range(k):
			if i in top_k:
				num_recovered += 1
	else:
		for i in supp:
			if i in top_k:
				num_recovered += 1
	return num_recovered * 1. / k

def rescale_variance(cov, theta, spike, k):
	# rescale top k variables so that its variances become 1. Can we make DT worse?!
	for i in range(k):
		for j in range(k):
			if True:
				cov[i][j] /= math.sqrt((1. + theta*spike[0][i]**2.)*(1. + theta*spike[0][j]**2.))

	#print cov[0:k,0:k]


######################### SLR METHODS #############################

def truncatedLasso(y, X, d, n, k):
	#alpha = math.sqrt(math.log(d) / n) * 2.       # theoretical recommendation
	alpha = 0.1
	lasso = linear_model.Lasso(alpha=alpha, fit_intercept=False, normalize=False,
        precompute=False, copy_X=True, max_iter=100, tol=0.05,
        warm_start=False)#, selection='random')

	lasso.fit(X, y)

	nz_coeffs = []
	for i in range(d-1):
		if not equal(lasso.coef_[i], 0.):
			nz_coeffs.append((lasso.coef_[i], i))

	#print nz_coeffs

	# truncation: zero out the coeffs that are not the kth largest
	for (_,i) in sorted(nz_coeffs)[0: max(0, len(nz_coeffs) - k)]:
		lasso.coef_[i] = 0.

	#print 'diff', before - lasso.predict(X)

	return lasso.predict(X)   # X \betahat


# 'Q' in FoBa paper. Squared two norm error.
def R(y, X, beta):
	#print 'yo', y.shape, X.shape, beta.shape
	return np.linalg.norm(y - X.dot(beta)) ** 2. / y.shape[0]

# find ordinary least squares minimizer for regressing y on X_F (restricted to columns in F)
def OLS(y, X, F):
	X_F = None
	for i in F:
		if X_F is None:
			X_F = X[:,i].reshape((X.shape[0],1))
		else:
			X_F = np.hstack((X_F, X[:,i].reshape((X.shape[0],1))))

	# beta is |F|-dim here
	beta = np.linalg.lstsq(X_F, y)[0]

	# augment back to d-dim
	d = X.shape[1]
	beta_full = np.array([0.] * d)
	j = 0
	for i in F:
		beta_full[i] = beta[j]
		j += 1

	return beta_full

'''
	Foward-Backward of Tong Zhang
'''
def FoBa(y, X, eps=1e-1):
	nu = 0.5     # tunes how aggressively backward compared to forward
	d = X.shape[1]
	n = X.shape[0]

	k = 0        # size of basis
	F = set()    # current basis
	beta_old = np.array([0.] * d)
	while True:
		# ================== forward  ===================
		residual = y - X.dot(beta_old)

		i_best = None
		beta_new = None
		MIN = sys.float_info.max
		perm = np.random.permutation(d)
		for i in perm:
			X_i = X[:,i]
			# error after removing X_i direction from the old residual
			alpha = residual.dot(X_i) / (np.linalg.norm(X_i) ** 2.)
			err = np.linalg.norm(residual - alpha * X_i) ** 2. / n
			if err < MIN:
				MIN = err
				i_best = i

		# add the new feature that leads to most incremental reduction in error,
		# then recompute respect to new basis.
		F.add(i_best)
		#print 'added', i_best
		beta_new = OLS(y, X, F)

		delta = R(y, X, beta_old) - R(y, X, beta_new)
		if delta <= eps:
			break

		k += 1
		beta_old = np.copy(beta_new)

		# ==================== backward  ===================
		while True:
			MIN = sys.float_info.max
			j_best = None
			#print 'len', len(F)
			for j in F:
				beta_new = np.copy(beta_old)
				beta_new[j] = 0.
				R_new = R(y, X, beta_new)
				if R_new < MIN:
					MIN = R_new
					j_best = j

			d_neg = MIN - R(y, X, beta_old)
			d_pos = delta

			#print d_neg, nu * d_pos
			if d_neg > nu * d_pos:
				break

			k -= 1
			F.remove(j_best)
			print 'removed', j_best
			beta_old = OLS(y, X, F)

	#print F
	return beta_new

########################## TEST STATISTICS ############################

'''
	Diagonal thresholding, original credits to Johnstone and Lu?
'''
def test_diagonal(samples, n, d, k, theta, hyptest=False, supp=None):
	D = 0.
	Ds = []
	tau = 1 + theta/k
	for i in range(d):
		sigma_hat_i = (np.linalg.norm(samples[:,i]) ** 2.)/n
		D = max(D, sigma_hat_i)
		#if sigma_hat_i > tau:
		Ds.append((sigma_hat_i,i))

	#print 'top k', sorted(Ds, reverse=True)[0:k]
	#top_k = set([i for (D_i, i) in Ds])

	if hyptest:
		return D
	else:
		top_k = set([i for (D_i, i) in sorted(Ds, reverse=True)[0:k]])
		return fraction_recovered(top_k, k, supp=supp)

	'''
	if sorted([i for (D_i, i) in sorted(Ds, reverse=True)[0:k]])[-1] == k-1:
		return 1
	else:
		return 0
	'''

'''
	maximum correlation test (simplified version of Q)
'''
def test_T(X, n, d, k, theta, hyptest=False, supp=None):
	T = 0.
	Ts = []

	cov = X.T.dot(X)
	for i in range(d):
		#sigma_hat_i = (np.linalg.norm(samples[:,i]) ** 2.)/n
		T_i = 0.
		'''
		for j in range(d):
			if j != i:
				T_i = max(T_i, abs(samples[:,i].dot(samples[:,j])))
		'''

		y = X[:,i]
		#X = np.hstack((samples[:,0:i], samples[:,i+1:]))

		# regress on best choice of one other column
		for j in range(d):
			if j != i:
				#alpha = cov[i][j] / cov[j][j]
				T_i = max(T_i, cov[i][j] ** 2. / n)

		T = max(T, T_i)
		Ts.append((T_i,i))

	if hyptest:
		return T
	else:
		top_k = set([i for (T_i, i) in sorted(Ts, reverse=True)[0:k]])
		return fraction_recovered(top_k, k, supp=supp)

'''
	Minimal Dual Perturbation of Berthet, Rigollet
'''
def test_MDP(samples, n, d, k, theta):
	sample_cov = samples.T.dot(samples) / n

	# This is the objective: \lambda_max(st_z(\Sigma_hat)) + kz
	def dual_perturb(z):
		#print 'z', z
		cov = np.copy(sample_cov)
		#print 'before st'
		soft_threshold(cov, z)
		#print 'before e. val'
		eigenValues, eigenVectors = np.linalg.eig(cov)

		idx = eigenValues.argsort()[::-1]
		eigenValues = eigenValues[idx]
		eigenVectors = eigenVectors[:,idx]

		DP = max(eigenValues) + k * z
		#print 'WARNING', DP, sum([(1 if x > 1e-2 else 0) for x in eigenVectors[0]])
		return DP

	'''
 	z0 = math.log(d,2)/n + math.sqrt(math.log(d,2)/n)
	result = sc.optimize.fmin_bfgs(dual_perturb, z0, full_output=True, retall=True, maxiter=10)
	fopt = result[1]
	'''

	MAX = max([max (a) for a in sample_cov])
	SPLIT = 100
	MDP = sys.float_info.max
	for i in range(SPLIT):
		z = MAX * i / SPLIT
		DP = dual_perturb(z)
		MDP = min(MDP, DP)
		if DP > MDP + 1.0:
			break

	return MDP
	#return fopt

'''
	Our Q!
'''
def test_Q(samples, n, d, k, theta, spike, slr='lasso', hyptest=False, supp=None, alpha=1.0, beta=1.0):
	#tau = 10. * k * math.log(d) / n  * 0.005

	Q = -99999.
	count = 0
	Qs = []

	for i in range(d):
		#i = random.randint(0, d-1)
		#print i
		y = samples[:,i]
		X = np.hstack((samples[:,0:i], samples[:,i+1:]))

		# Use sLR "blackbox" of choice here
		if slr == 'FoBa':
			#print i
			predicted = X.dot(FoBa(y, X))
		elif slr == 'opt':
			u_rest = np.hstack((spike[0,0:i], spike[0,i+1:])).T
			u_i = spike[0,i]
			predicted = X.dot(u_rest) * (theta * u_i)/(1 + (1 - u_i**2.)*theta)
		else:
			predicted = truncatedLasso(y, X, d, n, k)

		#print predicted
		Q_i = (alpha*(np.linalg.norm(y) ** 2.) - beta*(np.linalg.norm(y - predicted) ** 2.))/n
		#if Q_i == 0:
		#	print i, np.linalg.norm(y - predicted) ** 2.

		Q = max(Q, Q_i)
		Qs.append((Q_i,i))

	if hyptest:
		#print Q
		return Q
	else:
		top_k = set([i for (Q_i, i) in sorted(Qs, reverse=True)[0:k]])
		#print [i for (Q_i, i) in sorted(Qs, reverse=True)[0:k]]
		return fraction_recovered(top_k, k, supp=supp)

	'''
	if sorted([i for (Q_i, i) in sorted(Qs, reverse=True)[0:k]])[-1] == k-1:
		return 1
	else:
		return 0
	'''


'''
	For experimenting with alternate meta-statistics using Q
'''
def test_Q_spread(samples, n, d, k, theta, spike, slr='lasso', hyptest=False, supp=None):
	#tau = 10. * k * math.log(d) / n  * 0.005

	Qmax = 0.
	Qmin = 100000.
	count = 0
	Qs = []

	for i in range(d):
		#i = random.randint(0, d-1)
		#print i
		y = samples[:,i]
		X = np.hstack((samples[:,0:i], samples[:,i+1:]))

		# Use sLR "blackbox" of choice here
		if slr == 'FoBa':
			#print i
			predicted = X.dot(FoBa(y, X))
		elif slr == 'opt':
			u_rest = np.hstack((spike[0,0:i], spike[0,i+1:])).T
			u_i = spike[0,i]
			predicted = X.dot(u_rest) * (theta * u_i)/(1 + (1 - u_i**2.)*theta)
		else:
			predicted = truncatedLasso(y, X, d, n, k)

		Q_i = ((np.linalg.norm(y) ** 2.) - (np.linalg.norm(y - predicted) ** 2.))/n

		Qmax = max(Qmax, Q_i)
		Qmin = min(Qmin, Q_i)
		Qs.append((Q_i,i))

	return [Q_i for (Q_i, i) in Qs]

	if hyptest:
		return sum([Q_i for (Q_i, i) in sorted(Qs, reverse=True)[0:k]])
		#return Qmax - Qmin
	else:
		top_k = set([i for (Q_i, i) in sorted(Qs, reverse=True)[0:k]])
		#print [i for (Q_i, i) in sorted(Qs, reverse=True)[0:k]]
		return fraction_recovered(top_k, k, supp=supp)


'''
	Truncated Power method of Yuan, Zhang
'''
def test_TPower(samples, n, d, k, theta, supp=None):
	cov = samples.T.dot(samples)
	DELTA = 0.01

	eigenValues, eigenVectors = np.linalg.eig(cov)
	idx = eigenValues.argsort()[::-1]     # sorted largest to smallest
	eigenValues = eigenValues[idx]
	eigenVectors = eigenVectors[:,idx]
	x0 = eigenVectors[:,0]                # initialize with top eigenvector

	entries = [(abs(x0[i]), i) for i in range(d)]
	top_k = sorted([i for (_, i) in sorted(entries, reverse=True)[0:k]])
	#print x0[0:40]
	print 'init ', fraction_recovered(top_k, k, supp=supp)

	x_old = x0
	x_new = None

	while True:
		x_new = cov.dot(x_old)
		truncate(x_new, k)
		x_new /= np.linalg.norm(x_new)

		entries = [(abs(x_new[i]), i) for i in range(d)]
		top_k = sorted([i for (_, i) in sorted(entries, reverse=True)[0:k]])
		#print x0[0:40]
		print 'current ', fraction_recovered(top_k, k, supp=supp)

		if np.linalg.norm(x_new - x_old) < DELTA:
			break
		x_old = x_new

	# pick out support of non-zero entries
	#print x_new[0:20]
	supp_hat = [i for i in range(d) if abs(x_new[i]) > 0.]
	#print supp_hat

	return fraction_recovered(supp_hat, k, supp=supp)



'''
	Generalized Power method of Journee, Nesterov, Richtarik, Sepulchre (Algorithm 2)
'''
def test_GPower(samples, n, d, k, theta, supp=None):
	A = samples
	i_star = max([(np.linalg.norm(A[:,i]), i) for i in range(d)])[1]   # argmax_i |a_i|
	x0 = np.copy(A[:,i_star])
	x0 /= np.linalg.norm(x0)
	print x0.shape

	def nonneg(x):
		return 0 if x < 0 else x
	def sgn(x):
		return 1 if x > 0 else -1


	# Gamma can be continuously adjusted to tradeoff sparisty vs. explained variance.
	# Since the paper does not seem to offer how to choose gamma,
	# binary search for appropriate value of gamma to achieve desired sparsity level
	gammaLow = 0.
	gammaHigh = np.linalg.norm(A[:,i_star])
	#print gammaHigh
	#print sorted([(np.linalg.norm(A[:,i]), i) for i in range(d)], reverse=True)[0:10]
	x = None
	sparsity = -1
	DELTA = 0.0001

	while True:
		gamma = (gammaLow + gammaHigh)/2.
		#print gammaLow, gammaHigh

		x = x0
		x_old = x
		while True:
			x = nonneg(abs(A[:,0].dot(x_old)) - gamma) * sgn(A[:,0].dot(x_old)) * A[:,0]
			#print nonneg(abs(A[:,28].dot(x_old)) - gamma) * sgn(A[:,28].dot(x_old)) * A[:,28]

			for i in range(1,d):
				x += nonneg(abs(A[:,i].dot(x_old)) - gamma) * sgn(A[:,i].dot(x_old)) * A[:,i]
			x /= np.linalg.norm(x)

			#print np.linalg.norm(x - x_old)
			if np.linalg.norm(x - x_old) < DELTA:
				break
			x_old = x

		sparsity = sum([(1 if A[:,i].T.dot(x) > gamma else 0) for i in range(d)])
		print sparsity
		if sparsity == k:
			break
		elif sparsity > k:
			gammaLow = gamma
		else:
			gammaHigh = gamma

	supp_hat = [i for i in range(d) if A[:,i].dot(x) > gamma]
	print supp_hat
	return fraction_recovered(supp_hat, k, supp=supp)


'''
	Covariance thresholding of Desphande, Montenari
'''
def test_CT(samples, n, d, k, theta, tau=0.1, mu=1.0, supp=None):
	n = n/2
	# split samples
	samples1 = samples[0:n]
	samples2 = samples[n:]
	cov1 = samples1.T.dot(samples1) / n - np.identity(d)  # remove diagonal contribution
	cov2 = samples2.T.dot(samples2) / n - np.identity(d)
	'''
	for i in range(d):
		cov1[i][i] = 0.
		cov2[i][i] = 0.
	'''

	# tau = 3 * sigma^2 suggested  (see p. 7 of [DM14])
	soft_threshold(cov1, tau/math.sqrt(n))

	eigenValues, eigenVectors = np.linalg.eig(cov1)
	idx = eigenValues.argsort()[::-1]     # sorted largest to smallest
	eigenValues = eigenValues[idx]
	eigenVectors = eigenVectors[:,idx]

	#print 'evalues', eigenValues[0], eigenValues[-1]

	top_evector = eigenVectors[:,0]
	hard_threshold(top_evector, mu/(2.*math.sqrt(k)))

	# Find entries with large correlation with largest entries in the top e. vector.
	# Use the fresh second set of samples.
	signal = cov2.dot(top_evector)

	entries = [(abs(signal[i]), i) for i in range(d)]
	top_k = sorted([i for (_, i) in sorted(entries, reverse=True)[0:k]])
	#print 'top k', top_k

	return fraction_recovered(top_k, k, supp=supp)

	'''
	# check if all {0,...,k-1} were correctly recovered.
	if top_k[-1] == k-1:
		return 1
	else:
		return 0
	'''