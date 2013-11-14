import matplotlib.pyplot as plt
import numpy as np


fig = plt.figure()
axs = fig.add_subplot(111)
data = np.loadtxt('whData.dat',dtype=np.object,comments='#',delimiter=None)
X = data[:,0:2].astype(np.float)
y = data[:,2]
X=X.T
A=[]
NEW=[]
for i in xrange(0, 39):
    if(int(X[0][i])>0 and int(X[1][i])>0):
        A.append(X[0][i])
NEW.append(A)
A=[]
for i in xrange(0, 39):
    if(int(X[0][i])>0 and int(X[1][i])>0):
        A.append(X[1][i])
NEW.append(A)

axs.plot(NEW[0][:], NEW[1][:],'ro', label='data')
xmin = 0
xmax = X[0,:].max()
axs.set_xlim(xmin-10, xmax+10)
axs.set_ylim(-2, X[1,:].max()+10)
leg = axs.legend(["Data"],loc=2, shadow=True, fancybox=True, numpoints=1)
leg.get_frame().set_alpha(0.5)
plt.show()