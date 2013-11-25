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
### task1.py: 
###
###
################################################################################

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
	# load data
	data = np.loadtxt('whData.dat',dtype=np.object,comments='#',delimiter=None)

	# read height and weight data into 2D float array
	X = data[:,0:2].astype(np.float)

	# select only rows for which all entries are positive
	X = X[np.all(X>0, axis=1),:]

	# transpose
	X = X.T

	fig = plt.figure()
	axs = fig.add_subplot(111)

	# plot the data 
	axs.plot(X[0,:], X[1,:], 'ro', label='data w/o outliers')

	# set x and y limits of the plotting area
	xmin = X[0,:].min()
	xmax = X[0,:].max()
	axs.set_xlim(xmin-10, xmax+10)
	axs.set_ylim(0, X[1,:].max()+10)
	axs.set_xlabel('Weight')
	axs.set_ylabel('Height')
	axs.set_title('Height/Weight of students')

	# set properties of the legend of the plot
	leg = axs.legend(loc='upper left', shadow=True, fancybox=True, numpoints=1)
	leg.get_frame().set_alpha(0.5)

	plt.show()
#	plt.savefig('task1', facecolor='w', edgecolor='w',
#                    papertype=None, format='pdf', transparent=False,
#                    bbox_inches='tight', pad_inches=0.1)
