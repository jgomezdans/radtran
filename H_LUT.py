#!/usr/bin/python
'''A script to create a LUT for the values of the H function due
to the discontinuous nature of the function found in Myneni V.20 -
22, Shultis 1988 and Knyazikhin 2004.
The LUT will then be used in the ratran.py script to calculate the
Gamma function (Area Scattering Phase Function).
Stages to the script are:
  1. find the rectangular hyperbolic function at which H = 0 in
  the quadrants where theta_l > pi/2 and theta < pi/2 and > line
  where m = 1, c = pi/2 in theta_l = m*theta + c.
  theta_l = a / (b + theta) + c where:
  a = -2.05199487
  b = -2.41147172
  c = 0.72926237
  For the opposite quadrant the values are:
  a = -2.05187851
  b = -0.73015463
  c = 2.41228578

'''

import numpy as np
import matplotlib.pylab as plt
import scipy as sc
import pdb

theta = np.loadtxt('view_sun.csv', delimiter=',')
theta_l = np.loadtxt('leaf.csv', delimiter=',')
H = np.loadtxt('H_values.csv', delimiter=',')
'''
x = []
y = []
zeros = np.where(H<0.001,0,1)
for i in np.arange(0,np.shape(theta)[1]-1):
  for j in np.arange(0,np.shape(theta)[0]-1):
    ytest = theta[i,j] - np.pi/2.
    if H[i,j] < 0.001 and theta_l[i,j] >= ytest and theta_l[i,j] < np.pi/2.:
      x.append(theta[i,j])
      y.append(theta_l[i,j])
func = lambda a, b, c, x: a / (x + b) + c
a = -2.05187851
b = -0.73015463
c = 2.41228578
guess = (a,b,c)
drivers = (y, x)

def obj_func(guess, drivers):
  y = drivers[0]
  ypred = func(guess[0],guess[1],guess[2],drivers[1])
  mismatch = y - ypred
  rmse = (np.sum(mismatch**2)/len(y))**0.5
  return rmse

print a, b, c
#opt_guess = sc.optimize.fmin_powell(obj_func, guess, args=[drivers])
#print opt_guess
xtest = np.linspace(-np.pi,np.pi,250)
ytest = func(guess[0],guess[1],guess[2],xtest)
plt.pcolormesh(theta, theta_l, zeros)
plt.plot(x,y,'xk')
plt.plot(xtest,ytest,'pk')
plt.axis([theta.max(),theta.min(),theta_l.max(),theta_l.min()])
plt.show()
'''
f = lambda a, b, c , x: a/(b+x)+c
for i, x  in enumerate(theta[0]):
  for j, y in enumerate(theta_l.T[0]):
    a = (-2.05199487, -2.05187851)
    b = (-2.41147172, -0.73015463)
    c = (0.72926237, 2.41228578)
    if y < f(a[0],b[0],c[0],x) and y > c[0]:
      H[i,j] = 0.
plt.pcolormesh(theta, theta_l, H)
#plt.plot(theta.T[0],f(a[1],b[1],c[1],theta.T[0]))
plt.axis([theta.max(),theta.min(),theta_l.max(),theta_l.min()])
plt.show()


