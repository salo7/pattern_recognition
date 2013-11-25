#!/usr/bin/env python

################################################################################
### Pattern recognition 1st project
### Team members:
###		Panagiotis Salonitis 2516667
###		Nikhil Patra
###		Parsa Vali 2554517
###		Shiva Shokouhi 2615879
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

# loads and normalizes the data
def load_data():
	# load the csv data into an ndarray
	data = np.loadtxt('myspace.csv',dtype=np.object,delimiter=',')
	
	# select the second column of data and convert to int since we need it
	# as input for the repeat function
	X = data[:,1].astype(np.int)
	
	# keep only the positive values
	return X[X>0]

# following Newton's formula gives an approximation of it's parameters
# k_init: initial value of Weibull's shape parameter
# a_init: initial value of Weibull's scale parameter
# n_iter: number of iterations over Newton's method
def approximate_weibull_parameters(data, k_init, a_init, n_iter):
	# back to float for our computations	
	data = data.astype(np.float)	
	p = np.array([[k_init], [a_init]], dtype=np.float)
	
	hessian = np.ndarray((2,2))
	grad = np.ndarray((2,1))
	
	N = data.size
	print 'data size: ', N
	
	# increased number of iterations in Newton's formula gives as a better estimation of M.L.E.
	for i in xrange(n_iter):
		# variables added for readability (overhead is less important in this case)		
		k = p[0]
		a = p[1]
		
		print "Iteration", i, k[0], a[0]
		
		# temporal variables for repetitive computations
		tmp1 = np.power(data/a, k)	# (d_i/a)^k
		tmp2 = np.log(data/a)		# log(d_i/a)
		
		# Gradient matrix of Newton's formula
		
		# different from the original log likelihood formula given 
		# original would be like the following
		# grad[0] = -(N/k - N*math.log(a) - np.sum(np.log(data))/N - np.sum(tmp1*tmp2))
		
		grad[0] = -(N/k - N*math.log(a) + np.sum(np.log(data)) - np.sum(tmp1*tmp2)) 
		grad[1] = -((k/a)*(np.sum(tmp1) - N))
		
		# Hessian matrix of Newton's formula
		
		# the first derivative is different than the one given
		# original would be like the following
		# hessian[0][0] = (N/pow(k,2))-np.sum(tmp1*np.power(tmp2,2))
		
		hessian[0][0] = -(N/pow(k,2))-np.sum(tmp1*np.power(tmp2,2))
		hessian[0][1] = (1/a)*np.sum(tmp1) + (k/a)*np.sum(tmp1*tmp2) - (N/a)
		hessian[1][0] = hessian[0][1]
		hessian[1][1] = (k/pow(a,2))*(N-(k+1)*np.sum(tmp1)) 
		
		# producto of inverse Hessian matrix with Gradient matrix added to the previous result
		p += (np.dot(linalg.pinv(hessian), grad))
	
	print '\n',"Final", k[0], a[0]	
	return p
		

if __name__ == "__main__":
	h = load_data()
	# array enumerating for each cell it's position + 1
	x = np.arange(1,h.size+1)
	
	# repeat each value h times, where h is it's frequency
	# log likelihood of weibull requires this input
	d = np.repeat(x,h) 
	
	# our function
	p = approximate_weibull_parameters(d, 1, 1, 20)
	
	#  MLEs for shape, location, and scale parameters from the given data
	p0, p1, p2= weibull_min.fit(d, floc=0)
	print p0,p1,p2
	
	# using the  MLEs for shape, location, and scale parameters, compute the
	# probability density function of Weibull's distribution
	dist= weibull_min.pdf(x, p0, p1, p2)
	
	# plot the graphs
	fig, subplots = plt.subplots()
	subplots.plot(x, dist * d.size, ls='-', c='red',label='Weibull fit')
	subplots.plot(x, h,  ls='-', c='grey', label='Google data')	
	
	# legend properties
	leg = subplots.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
       ncol=2, mode="expand", borderaxespad=0., shadow=False, fancybox=False, numpoints=1)
	leg.get_frame().set_alpha(1)
	
	plt.show()
#    plt.savefig('task3', facecolor='w', edgecolor='w',
#                    papertype=None, format='pdf', transparent=False,
#                    bbox_inches='tight', pad_inches=0.1)
