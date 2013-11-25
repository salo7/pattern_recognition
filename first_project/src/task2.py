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
### task2.py: 
###
###
################################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

# for the given set of data plots them and together with a normal distribution characterizing their density
def plotNormalDistribution(data):	
	# compute mean and standard deviation using numpy
	mean = np.mean(data)
	deviation = np.std(data)	
	print "Mean:", mean, "Deviation:", deviation
	
	# list of evenly distributed numbers in our range o X axis
	x = np.linspace(np.amin(data) - 10 ,np.amax(data) + 10, 100 )
    
    # create one subplot for the data points and one for the normal distribution
	fig, subplots = plt.subplots()
	subplots.plot(data, [0] * data.size, 'ro', label='data')
	subplots.plot(x,mlab.normpdf(x,mean,deviation), label='normal')
	
	# legend properties
	leg = subplots.legend(loc='upper left', shadow=True, fancybox=True, numpoints=1)
	leg.get_frame().set_alpha(0.5)
    
	plt.show()
#    plt.savefig('task2', facecolor='w', edgecolor='w',
#                    papertype=None, format='pdf', transparent=False,
#                    bbox_inches='tight', pad_inches=0.1)

if __name__ == "__main__":
	# load data
	data = np.loadtxt('whData.dat',dtype=np.object,comments='#',delimiter=None)

	# read height and weight data into 2D float array
	X = data[:,1].astype(np.float)

	# select only rows for which all entries are positive
	X = X[X>0]
		
	plotNormalDistribution(X)
