#!/usr/bin/env python

################################################################################
### Pattern recognition 1st project
### Team members:
###		Panagiotis Salonitis 2516667
###		Nikhil Patra
###		Parsa Vali
###		Shiva Shokouhi
###		Thomas Werner
###
### task3.py: 
###
###
################################################################################

import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import math
from scipy.stats import weibull_min

def load_data():
	# load the csv data into an ndarray
	data = np.loadtxt('myspace.csv',dtype=np.object,delimiter=',')
	
	# select the second column of data and convert to int
	X = data[:,1].astype(np.int)
	
	# keep only the positive values
	return X[X>0]

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
		
		# CHANGE from the original log likelihood formula given
		grad[0] = -(N/k - N*math.log(a) + np.sum(np.log(data)) - np.sum(tmp1*tmp2)) # removed /N
		grad[1] = -((k/a)*(np.sum(tmp1) - N))
		
		# CHANGE - minus in the first derivative was not in the original formula
		hessian[0][0] = -(N/pow(k,2))-np.sum(tmp1*np.power(tmp2,2))
		hessian[0][1] = (1/a)*np.sum(tmp1) + (k/a)*np.sum(tmp1*tmp2) - (N/a)
		hessian[1][0] = hessian[0][1]
		hessian[1][1] = (k/pow(a,2))*(N-(k+1)*np.sum(tmp1)) 
		
		p += (np.dot(linalg.pinv(hessian), grad))
	
	print '\n',"Final", k[0], a[0]	
	return p
		

if __name__ == "__main__":
	# load data
	h = load_data()
	x = np.arange(1,h.size+1)

	d = np.repeat(x,h) 
	
	p = fit_weibull(d, 1, 1, 20)
	
	fig, subplots = plt.subplots()

	p0, p1, p2= weibull_min.fit(d, floc=0)
	print p0,p1,p2
	
	dist= weibull_min.pdf(x, p0, p1, p2)
	subplots.plot(x, dist * d.size, ls='-', c='red',label='Weibull fit')#r'$k=%.1f,\ \lambda=%i$' % (p[0], p[1]))	
	subplots.plot(x, h,  ls='-', c='grey', label='Google data')	
	
	# legend properties
	leg = subplots.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
       ncol=2, mode="expand", borderaxespad=0., shadow=False, fancybox=False, numpoints=1)
	leg.get_frame().set_alpha(1)
	
	plt.show()
#    plt.savefig('task3', facecolor='w', edgecolor='w',
#                    papertype=None, format='pdf', transparent=False,
#                    bbox_inches='tight', pad_inches=0.1)
