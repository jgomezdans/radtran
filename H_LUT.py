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
import pdb

theta = np.loadtxt('view_sun.csv', delimiter=',')
theta_l = np.loadtxt('leaf.csv', delimiter=',')
H = np.loadtxt('H_values.csv', delimiter=',')
hyperbola = lambda a, b, c , x: a/(b+x)+c

def remove_hips(paras, theta, theta_l, H):
  '''A function to remove the hips of the figure.
  The parameters for the hyperbole that defines the
  hips needs to be defined seperately.
  Input: paras, theta, theta_l, H.
  Output: H without hips set to zero.
  '''
  for i, x  in enumerate(theta[0]):
    for j, y in enumerate(theta_l.T[0]):
      a = paras[0]
      b = paras[1]
      c = paras[2]
      if (y > hyperbola(a[0],b[0],c[0],x) and x < np.pi/2.) or\
          (y < hyperbola(a[1],b[1],c[1],x) and x > np.pi/2.):
        H[j,i] = 0.
  return H

def remove_lips(paras, theta, theta_l, H):
  '''A function to remove the lips on either side of the tummy.
  The function will be extended to interpolate values.
  Input: theta, theta_l, H, paras 
  Output: H without lips set at 1.1
  '''
  for i, x  in enumerate(theta[0]):
    for j, y in enumerate(theta_l.T[0]):
      if x < np.pi/2.:
        ylimit = -x + np.pi/2.
      else:
        ylimit = -x + np.pi*3./2.
      a = paras[0]
      b = paras[1]
      c = paras[2]
      if (y < hyperbola(a[0],b[0],c[0],x) and y >= ylimit and\
          x < np.pi/2.) or\
          (y > hyperbola(a[1],b[1],c[1],x) and y <= ylimit and\
          x > np.pi/2.):
        #pdb.set_trace()
        H[j,i] = 1.1 #np.nan
  #pdb.set_trace()
  return H

def interp_lips(theta, theta_l, H):
  '''A function that interpolates values for where the lips
  were. It needs H to have lip values at 1.1.
  Input: theta, theta_l, H
  Output: H interpolated
  '''
  
  return
paras1 = ((-2.05199487, -2.05187851),(-2.41147172, -0.73015463),\
    (0.72926237, 2.41228578)) # right and left paras
H  = remove_hips(paras1, theta, theta_l, H)
paras2 = ((0.36582294, 0.36580791), (-1.76918702, -1.37241401),\
    (1.77475033, 1.36685314)) # top and bottom paras
H = remove_lips(paras2, theta, theta_l, H)

plt.pcolormesh(theta, theta_l, H)
plt.axis([theta.max(),theta.min(),theta_l.max(),theta_l.min()])
plt.show()
