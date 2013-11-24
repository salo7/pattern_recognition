#!/usr/bin/env python

import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math
import scipy
from scipy.stats import dweibull, weibull_min

def load_data():
	data = np.loadtxt('myspace.csv',dtype=np.object,delimiter=',')
	
	X = data[:,1].astype(np.int)
	
	return X[X>0]

def plot_weibull(data, shape, scale):
	dist = dweibull(shape, 0, scale)
	#plt.plot(x, dist.pdf(x), ls='-', c='black',label=r'$k=%.1f,\ \lambda=%i$' % (shape, scale))	
	plt.plot(data, np.arange(1,data.size+1), 'ro', label='data')
	plt.show()
	
def fit_weibull(data, k_init, a_init, n_iter):
	
	data = data.astype(np.float)
	
	p = np.array([[k_init], [a_init]], dtype=np.float)
	
	hessian = np.ndarray((2,2))
	grad = np.ndarray((2,1))
	
	N = data.size
	print 'data size: ', N
	
	for i in xrange(n_iter):
		
		
		k = p[0]
		a = p[1]
		
		print "Iteration", i, k[0], a[0]
		
		tmp1 = np.power(data/a, k)	# {\frac{d_i}{\alpha}}^\kappa
		tmp2 = np.log(data/a)		# \log{\frac{d_i}{\alpha}}
		
		grad[0] = -(N/k - N*math.log(a) + np.sum(np.log(data)) - np.sum(tmp1*tmp2)) # removed /N
		grad[1] = -((k/a)*(np.sum(tmp1) - N))
		
		# CHANGE - minus in the first derivative was not in the original formula
		hessian[0][0] = -(N/pow(k,2))-np.sum(tmp1*np.power(tmp2,2))
		hessian[0][1] = (1/a)*np.sum(tmp1) + (k/a)*np.sum(tmp1*tmp2) - (N/a) #CHANGE - (N/p[1]) was absent 
		hessian[1][0] = hessian[0][1]
		hessian[1][1] = (k/pow(a,2))*(N-(k+1)*np.sum(tmp1)) #CHANGE  N-(K+1) ->  N+(K-1)  
		
		p += (np.dot(linalg.pinv(hessian), grad))
	
	print '\n',"Final", k[0], a[0]	
	return p
		

if __name__ == "__main__":
	# load data
	h = load_data()
	x = np.arange(1,h.size+1)

	d = np.repeat(x,h) 
	
	p = fit_weibull(d, 1, 1, 20)
	
	fig, ax = plt.subplots()
	#plot_weibull(d, p[0], p[1])
	p0, p1, p2= weibull_min.fit(d, floc=0)
	print p0,p1,p2
	#dist = dweibull( p[0], loc=0, scale=p[1])
	dist= weibull_min.pdf(x, p0, p1, p2)
	ax.plot(x, dist * d.size, ls='-', c='red',label='weibull')#r'$k=%.1f,\ \lambda=%i$' % (p[0], p[1]))	
	ax.plot(x, h,  ls='-', c='grey', label='data')
	plt.show()
