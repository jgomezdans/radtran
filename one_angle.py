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

class rt_layers():
  '''The class that will encapsulate the data and methods to be used
  for a single unit of radiative transfer through the canopy. To be 
  implimented is the constructor which takes the initial parameters,
  and a destructor to clear the memory of the instance once processing
  has completed.
  Input: Tol - max tolerance, Iter - max iterations, N - no of 
  discrete ordinates over 0 to pi, K - no of nodes, Lc - total LAI, 
  refl - l eaf reflectance, trans - leaf transmittance, refl_s - 
  soil reflectance, I0 - Downward flux at TOC, sun0 - zenith angle
  of solar illumination.
  Ouput: a rt_layers class.
  '''

  def __init__(self, Tol = 1.e-6, Iter = 200, K = 10, N = 100,\
      Lc = 2., refl = 0.2, trans = 0.1, refl_s = 0.3, I0 = 1.,\
      sun0 = np.pi, arch = 's'):
    '''The constructor for the rt_layers class.
    See the class documentation for details of inputs.
    '''
    self.Tol = Tol
    self.Iter = Iter
    self.K = K
    self.N = N
    self.Lc = Lc
    self.refl = refl
    self.trans = trans
    self.refl_s = refl_s
    self.I0 = I0
    self.sun0 = sun0
    self.arch = arch

  def __del__(self):
    '''The post garbage collection method.
    '''
    print 'An instance of rt_layers has been destroyed.\n'
  
  def __repr__(self):
    '''This prints out the input parameters that define the 
    instance of the class.
    '''
    return '''Tol = %.e, Iter = %i, K = %i, N = %i, Lc = %.3f, 
refl = %.3f, trans = %.3f, refl_s = %.3f, I0 = %.4f,
sun = %.3f, arch = %s''' % (self.Tol, self.Iter, self.K, self.N, \
        self.Lc, self.refl, self.trans, self.refl_s, self.I0, \
        self.sun0, self.arch)

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

  def J(self, view, sun, arch, refl, trans, Ia):
    '''The J or Distributed Source Term according to Myneni 1988b
    (23). This gives the multiple scattering as opposed to First 
    Collision term Q.
    Input: view - the zenith angle of evaluation, sun - the illumination
    zenith angle, arch - archtype, refl - reflectance, trans - 
    transmittance, L - LAI, Ia - array of fluxes or intensities at
    the sun zenith angles with Ia[0] at sun = pi.
    Output: The J term.
    '''
    # expected that sun is a list of all incomming illumination
    # angles.
    if isinstance(sun, np.ndarray) and isinstance(Ia, np.ndarray):
      albedo = refl + trans
      integ1 = np.multiply(Ia,P(view,sun,arch,refl,trans))
      integ = np.multiply(integ1,G(sun,arch)/G(view,arch))
      j = albedo / 2. * np.sum(integ)
    else:
      raise Exception('ArrayInputRequired')
    return j

  def Q1(view, sun0, arch, refl, trans, L, I0):
    '''The Q1 First First Collision Source Term as defined in Myneni
    1988b (24). This is the downwelling part of the Q term. 
    Input: view - the zenith angle of evaluation, sun0 - the uncollided
    illumination zenith angle, arch - archetype, refl - reflectance, 
    trans - transmittance, L - LAI, I0 - flux or intensity at TOC.
    Ouput: The Q1 term.
    '''
    if isinstance(sun0, np.ndarray):
      raise Exception('SunScalarRequired')
    else: 
      albedo = refl + trans
      q = albedo / 4. * P(view,sun0,arch,refl,trans)*G(sun0,arch)/\
          G(view,arch) * I_f(sun0, L, I0, arch)
    return q

  def Q2(view, sun0, n_angles, arch, refl, trans, Lc, L, I0, refl_s):
    '''The Q2 Second First Collision Source Term as defined in
    Myneni 1988b (24). This is the upwelling part of the Q term.
    Input: view - the view zenith angle of evalution, sun0 - the 
    direct illumination zenith angle down, n_angles - the number of
    angles between 0 and pi, arch - archetype, refl - 
    reflectance, trans - transmittance, Lc - total LAI, L - LAI,
    I0 - flux or intensity of direct sun ray at TOC, refl_s - 
    soil reflectance.
    Output: The Q2 term.
    '''
    sun_up = np.linspace(np.pi/2.,0.,n_angles/2)
    albedo = refl + trans
    dL = Lc - L
    integ1 = np.multiply(P(view,sun_up,arch,refl,trans),\
        G(sun_up,arch)/G(view,arch)) # element-by-element multipl.
    integ = np.multiply(integ1, I_f(sun_up, -dL, 1., arch)) # ^^
    q = albedo / 4. * -2. * refl_s * np.cos(sun0) * \
        I_f(sun0, Lc, I0, arch) * np.sum(integ)
    return q

  def plot_J(self):
    '''A function that plots the J function.
    It requires the the number of intervals
    of the sun angle from 0. to pi, the archetype and reflection
    and transmission values and the Intensity or Flux.
    Input: view, n_angles, .
    Output: plots the J function values.
    '''
    sun = np.linspace(np.pi,0.,self.N)
    j = []
    view = sun.copy()
    Ia = np.zeros_like(view)
    index = round(np.pi - self.sun0)*self.N/np.pi
    Ia[index] = self.I0
    for v in view:
      j.append(self.J(v,sun,self.arch,self.refl,self.trans,Ia))
    plt.plot(sun,j,'--r')
    plt.show()

  def plot_Q1(sun0,n_angles,arch,refl,trans,L,I0):
    '''A function that plots the Q1 function.
    It requires the single sun angle (not list) for uncollided
    Downwelling illumination, the number of view angles to 
    evaluate between 0 and pi, the archetype, the reflectance,
    transmittance, LAI, and Flux at the TOC.
    Input: sun0 - sun angle, n_angles - number of angles, arch -
    archetype, refl - reflectance, trans - transmittance, L - 
    LAI, I0 - Flux at TOC.
    Output: Q1 term
    '''
    view = np.linspace(0.,np.pi,n_angles)
    q = []
    for v in view:
      q.append(Q1(v, sun0, arch, refl, trans, L, I0))
    plt.plot(view,q,'--r')
    plt.show()

  def plot_Q2(sun0, n_angles, arch, refl, trans, Lc, L, I0, rs):
    '''A function that plots the Q2 function. To be noted is that
    this is the upwelling component of the first collision term Q.
    The graph would then portray greater scattering in the lower
    half towards view angles pi/2 to pi for the typical scenarios.
    Input: sun, n_angles, arch, refl trans, Lc, L, I0, rs.
    Ouput: Q2 term.
    '''
    view = np.linspace(0.,np.pi,n_angles)
    q = []
    for v in view:
      q.append(Q2(v, sun0, n_angles, arch, refl, trans, Lc, L, I0, rs))
    plt.plot(view,q,'--r')
    plt.show()

  def plot_Q(sun0,n_angles,arch,refl,trans,Lc,L,I0,rs):
    '''A function that plots the Q function as sum of Q1 and Q2.

    '''
    view = np.linspace(0.,np.pi,n_angles)
    q = []
    for v in view:
      q.append(Q1(v, sun0, arch, refl, trans, L, I0)+\
          Q2(v, sun0, n_angles, arch, refl, trans, Lc, L, I0, rs))
    plt.plot(view,q,'--r')
    plt.show()

'''
def setup():
 

def test_I_f():
'''


