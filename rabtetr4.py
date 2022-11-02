#1
import matplotlib.pyplot as plt
from numpy import *
import numpy as np
from numpy.random import *

delta = 1.0
x=linspace(-5,5,11)
y=x**3+x**2+delta*(rand(11)-0.5)
x+=delta*(rand(11)-0.5)

x.tofile('x_data.txt','\n')
y.tofile('y_data.txt','\n')

x=fromfile('x_data.txt',float,sep='\n')
y=fromfile('y_data.txt',float,sep='\n')

print(x)
print(y)

m=vstack((x**3,x**2,x,ones(11))).T
s=np.linalg.lstsq(m,y,rcond=None)[0]
x_prec=linspace(-5,5,11)
plt.plot(x,y,'D')
plt.plot(x_prec,s[0]*x_prec**3+s[1]*x_prec**2+s[2]*x_prec+s[3])
plt.grid()
plt.savefig('parabola.png')

#2






