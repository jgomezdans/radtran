#!/usr/bin/python

'''This will turn out to be the one-angle discrete-ordinate exact-
kernel finite-difference implimentation of the RT equation as set
out in Myneni 1988b. The script references the radtran.py module
which holds all the functions required for the calculations.
'''

import numpy as np
import scipy as sc
import matplotlib.pylab as plt
import warnings
from radtran import *
import nose
import pdb

# interactive selection list to be replaced by input file or prompt
Tol = 1.e-6 # max tolerance
Iter = 200 # max iterations
K = 10 # no. of layers
N = 10 # no. of ordinates
Lc = 2. # LAI
refl = 0.2 # leaf reflectance
trans = 0.1 # leaf transmittance
refl_s = 0.3 # soil reflectance
I0 = 1. # flux at TOC

def I_f(angle, L, I, arch):
  '''A function that will operate as the Beer's law exponential 
  formula. It requires the angle in radians, the 
  optical depth or LAI, and the initial intensity or flux. The
  example used is in Myneni (19). It  provides the calculation 
  of the following:
  I_f = |cos(angle)| * I * exp(G(angle)*L/cos(angle))
  Input: angle - the illumination zenith angle, L - LAI, I - 
  the intial intensity or flux, arch - archetype (see 
  radtran.py).
  Output: the intensity or flux at L.
  '''
  mu = np.cos(angle)
  i =  I * np.exp(G(angle,arch)*L/mu)
  return i

def J(view, sun, arch, refl, trans, Ia):
  '''The J or Distributed Source Term according to Myneni 1988b
  (23). This gives the multiple scattering as opposed to First 
  Collision term Q.
  Input: view - the zenith angle of evaluation, sun - the illumination
  zenith angle, arch - archtype, refl - reflectance, trans - 
  transmittance, L - LAI, Ia - array of fluxes or intensities at
  the sun zenith angles.
  Output: The J term.
  '''
  # it is to be expected that sun is a list of all incomming illumination
  # angles.
  if isinstance(sun, np.ndarray) and isinstance(Ia, np.ndarray):
    albedo = refl + trans
    integ1 = np.multiply(Ia,P(view,sun,arch,refl,trans))
    integ = np.multiply(integ1,G(sun,arch)/G(view,arch))
    j = albedo / 2. * np.sum(integ)
  else:
    raise Exception('ArrayInputRequired')
  return j

def Q1(view, sun, arch, refl, trans, L, I):
  '''The Q1 First First Collision Source Term as defined in Myneni
  1988b (24). This is the downwelling part of the Q term. 
  Input: view - the zenith angle of evaluation, sun - the illumination
  zenith angle, arch - archetype, refl - reflectance, trans -
  transmittance, L - LAI, I - flux or intensity.
  Ouput: The Q1 term.
  '''
  if isinstance(sun, np.ndarray):
    albedo = refl + trans
    q = albedo / 4. * P(view,sun,arch,refl,trans)*G(sun,arch)/\
        G(view,arch) * I_f(sun, L, I, arch)
  else: 
    raise Exception('SunArrayRequired')
  return q

def Q2(view, sun1, sun2, arch, refl, trans, Lc, L, I0, rs):
  '''The Q2 Second First Collision Source Term as defined in
  Myneni 1988b (24). This is the upwelling part of the Q term.
  Input: view - the view zenith angle of evalution, sun1 - the 
  direct illumination zenith angle down, sun2 - list of the 
  upwelling illumination zenith angles, arch - archetype, refl - 
  reflectance, trans - transmittance, Lc - LAI at BOC, L - LAI,
  I0 - flux or intensity of direct sun ray at TOC, refl_s - 
  soil reflectance.
  Output: The Q2 term.
  '''
  if isinstance(sun2, np.ndarray):
    albedo = refl + trans
    dL = Lc - L
    integ1 = np.multiply(P(view,sun2,arch,refl,trans),\
        G(sun2,arch)/G(view,arch)) # element-by-element multipl.
    integ = np.multiply(integ1, I_f(sun, -dL, 1., arch)) # ^^
    q = albedo / 4. * -2. * refl_s * np.cos(sun1) * \
        I_f(sun1, Lc, I0, arch) * np.sum(integ)
  else:
    raise Exception('SunArrayRequired')
  return q

def plot_J(n_angles,arch,refl,trans,Ia):
  '''A function that plots the J function.
  It requires the the number of intervals
  of the sun angle from 0. to pi, the archetype and reflection
  and transmission values and the Intensity or Flux.
  Input: view, n_angles, .
  Output: plots the J function values.
  '''
  sun = np.linspace(np.pi,0.,n_angles)
  j = []
  view = sun.copy()
  for v in view:
    j.append(J(v,sun,arch,refl,trans,Ia))
  plt.plot(sun,j,'--r')
  plt.show()


'''
def setup():
 

def test_I_f():
'''


