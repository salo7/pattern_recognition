#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


if __name__ == "__main__":
	# load data
	data = np.loadtxt('whData.dat',dtype=np.object,comments='#',delimiter=None)

	# read height and weight data into 2D float array
	X = data[:,1].astype(np.float)

	# select only rows for which all entries are positive
	X = X[X>0]
	
	mean = np.mean(X)
	std = np.std(X)
	
	print "Mean:", mean, "Std:", std
	
	# ===== histogramm =====
	fig0 = plt.figure(0)
	ax0 = fig0.add_subplot(111)
	
	n_bins = X.max()-X.min()
	n, bins, patches = ax0.hist(X, n_bins, normed=1, facecolor='green', alpha=0.5, label='data')
	
	x_norm = range(140,210)
	y_norm = mlab.normpdf(x_norm, mean, std)
	ax0.plot(x_norm, y_norm, 'r--', label='normal')
	ax0.legend()
	
	# ===== normal ======
	
	fig1 = plt.figure(1)
	ax1 = fig1.add_subplot(111)
	
	ax1.plot(X, np.zeros(X.shape), 'o', label='data')
	
	ax1.plot(x_norm, y_norm, 'r--', label='normal')

	ax1.legend()
	plt.show()
