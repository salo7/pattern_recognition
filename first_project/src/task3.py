#!/usr/bin/env python

import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math

def load_data():
	data = np.loadtxt('myspace.csv',dtype=np.object,delimiter=',')
	
	X = data[:,1].astype(np.int)
	
	return X[X>0]
	
def fit_weibull(data, k_init, a_init, n_iter):
	
	data = data.astype(np.float)
	
	p = np.array([[k_init], [a_init]], dtype=np.float)
	
	hessian = np.ndarray((2,2))
	grad = np.ndarray((2,1))
	
	N = data.size
	
	for i in xrange(n_iter):
		
		print "Iteration", i
		
		tmp1 = np.power(data/p[1], p[0])	# {\frac{d_i}{\alpha}}^\kappa
		tmp2 = np.log(data/p[1])		# \log{\frac{d_i}{\alpha}}
		
		grad[0] = -(N/p[0] - N*math.log(p[1]) + np.sum(np.log(data))/N - np.sum(tmp1*tmp2))
		grad[1] = -((p[0]/p[1])*(np.sum(tmp1) - N))
		
		hessian[0][0] = (N/pow(p[0],2))-np.sum(tmp1*np.power(tmp2,2))
		hessian[0][1] = (1/p[1])*np.sum(tmp1) + (p[0]/p[1])*np.sum(tmp1*tmp2) - (N/p[1])
		hessian[1][0] = hessian[0][1]
		hessian[1][1] = (p[0]/pow(p[1],2))*(N-(p[0]+1)*np.sum(tmp1))
		
		p -= (np.dot(linalg.pinv(hessian), grad))
		
	return p
		

if __name__ == "__main__":
	# load data
	h = load_data()
	x = np.arange(1,h.size+1)

	d = np.repeat(x,h) 
	
	p = fit_weibull(d, 1, 1, 20)
	
	print p
