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
import pickle
import os
from radtran import *
import nose
import pdb

# Gaussian quadratures sets to be used in instances
gauss_f_mu = open('lgvalues-abscissa.dat','rb')
gauss_f_wt = open('lgvalues-weights.dat','rb')
gauss_mu = pickle.load(gauss_f_mu)
gauss_wt = pickle.load(gauss_f_wt)
# sort all dictionary items
for k in gauss_mu.keys():
  ml = gauss_mu[k]
  wl = gauss_wt[k]
  ml, wl = zip(*sorted(zip(ml,wl),reverse=True))
  gauss_mu[k] = ml
  gauss_wt[k] = wl

class rt_layers():
  '''The class that will encapsulate the data and methods to be used
  for a single unit of radiative transfer through the canopy. To be 
  implimented is the constructor which takes the initial parameters,
  and a destructor to clear the memory of the instance once processing
  has completed.
  Input: Tol - max tolerance, Iter - max iterations, N - no of 
  discrete ordinates over 0 to pi, K - no of nodes, Lc - total LAI, 
  refl - leaf reflectance, trans - leaf transmittance, refl_s - 
  soil reflectance, I0 - Downward flux at TOC, sun0 - zenith angle
  of solar illumination.
  Ouput: a rt_layers class.
  '''


  def __init__(self, Tol = 1.e-3, Iter = 200, K = 40, N = 16,\
      Lc = 4., refl = 0.475, trans = 0.475, refl_s = 0.2, I0 = 1.,\
      sun0 = 180., arch = 's'):
    '''The constructor for the rt_layers class.
    See the class documentation for details of inputs.
    '''
    self.Tol = Tol
    self.Iter = Iter
    self.K = K
    if int(N) % 2 == 1:
      N = int(N)+1
      print 'N rounded up to even number:', str(N)
    self.N = N
    self.Lc = Lc
    self.refl = refl
    self.trans = trans
    self.refl_s = refl_s
    self.I0 = I0
    self.sun0 = sun0 * np.pi / 180.
    self.mu_s = np.cos(self.sun0)
    self.arch = arch
    self.albedo = self.refl + self.trans # leaf single scattering albedo
    self.gauss_wt = np.array(gauss_wt[str(N)])
    self.gauss_mu = np.array(gauss_mu[str(N)])

    # intervals
    dk = Lc/K
    self.mid_ks = np.arange(dk/2.,Lc,dk)
    self.n = self.N/2
    
    # node arrays and boundary arrays
    self.views = np.arccos(self.gauss_mu)
    self.sun_up = self.views[:self.n]
    # discrete ordinate equations
    g = G(self.views,self.arch)
    mu = np.cos(self.views) 
    self.a = (1. + g*dk/2./mu)/(1. - g*dk/2./mu)
    self.b = (g*dk/mu)/(1. + g*dk/2./mu)
    self.c = (g*dk/mu)/(1. - g*dk/2./mu)
    # G-function cross-sections
    self.Gx = g
    self.Gs = G(self.sun0,arch)
    self.Inodes = np.zeros((K,3,N)) # K, upper-mid-lower, N
    self.Jnodes = np.zeros((K,N))
    self.Q1nodes = self.Jnodes.copy()
    self.Q2nodes = self.Jnodes.copy()
    self.Px = np.zeros((N,N)) # P cross-section array
    self.Ps = np.zeros(N)
    for (i,v) in enumerate(self.views):
      self.Ps[i] = P(v,self.sun0,self.arch,self.refl,\
        self.trans)
    for (i,v) in enumerate(self.views):
      self.Px[i] = P(v, self.views,self.arch,self.refl,self.trans)
    for (i, k) in enumerate(self.mid_ks):
      for (j, v) in enumerate(self.views):
        self.Q1nodes[i,j] = self.Q1(v,k) 
        self.Q2nodes[i,j] = self.Q2(v,k)
    self.Bounds = np.zeros((2,N))

 
  # function to search angle database for index
  def angle_search(self,v,up=False):
    '''A method that provides the index of an angle in the
    views array. If up is True then provides the index 
    relative to the start of the array for upward angles,
    if False then it's relative to the middle or down
    direction.
    Input: v - angle to search, up - True/False.
    Output: index of angle in views array.
    '''
    if not up:
      index = np.where(np.abs(self.views-v)<=1.0e-6)[0][0]
    else:
      index = np.where(np.abs(self.views[:self.n]-v)<=1.0e-6)[0][0]
    return index

  def sun0(self,sun0):
    '''Method used for entering solar insolation angle which
    takes care of conversion to radians. Try not to assign 
    angles directly to self.sun0 variable but use this method.
    Input: sun0 - solar zenith angle in degrees.
    Output: converts and stores value in self.sun0.
    '''
    self.sun0 = sun0*np.pi/180.

  def I_down(self, k):
    '''The discrete ordinate downward equation.
    '''
    n = self.n
    self.Inodes[k,2,n:] = self.a[n:]*self.Inodes[k,0,n:] - \
        self.c[n:]*(self.Jnodes[k,n:] + self.Q1nodes[k,n:] + \
        self.Q2nodes[k,n:])
    if min(self.Inodes[k,2,n:]) < 0.:
      self.Inodes[k,2,n:] = np.where(self.Inodes[k,2,n:] < 0., \
        0., self.Inodes[k,2,n:]) # negative fixup need to revise....
      print 'Negative downward flux fixup performed at node %d' \
          %(k+1)
    if k < self.K-1:
      self.Inodes[k+1,0,n:] = self.Inodes[k,2,n:]
  
  def I_up(self, k):
    '''The discrete ordinate upward equation.
    '''
    n = self.n
    self.Inodes[k,0,:n] = 1./self.a[:n]*self.Inodes[k,2,:n] + \
        self.b[:n]*(self.Jnodes[k,:n] + self.Q1nodes[k,:n] + \
        self.Q2nodes[k,:n])
    if min(self.Inodes[k,0,:n]) < 0.:
      self.Inodes[k,0,:n] = np.where(self.Inodes[k,0,:n] < 0., \
         0., self.Inodes[k,0,:n]) # negative fixup need to revise...
      print 'Negative upward flux fixup performed at node %d' \
          %(k+1)
    if k != 0:
      self.Inodes[k-1,2,:n] = self.Inodes[k,0,:n]

  def reverse(self):
    '''Reverses the transmissivity at soil boundary.
    '''
    Ir = np.multiply(np.cos(self.views[self.n:]),\
        self.Inodes[self.K-1,2,self.n:])
    self.Inodes[self.K-1,2,:self.n] = - 2. * self.refl_s * \
        np.sum(Ir*self.gauss_wt[self.n:]) 

  def converge(self):
    '''Check for convergence and returns true if converges.
    '''
    misclose_top = np.abs((self.Inodes[0,0] - self.Bounds[0])/\
        self.Inodes[0,0])
    misclose_bot = np.abs((self.Inodes[self.K-1,2] - \
        self.Bounds[1])/self.Inodes[self.K-1,2])
    max_top = max(misclose_top)
    max_bot = max(misclose_bot)
    print 'misclosures top: %.g, and bottom: %.g.' %\
        (max_top, max_bot)
    #pdb.set_trace()
    if max_top  <= self.Tol and max_bot <= self.Tol:
      return True
    else:
      return False
  
  def solve(self):
    '''The solver. Run this as a method of the instance of the
    rt_layers class to solve the RT equations. You first need
    to create an instance of the class though using:
    eg. test = rt_layers() # see rt_layers for more options.
    then test.solve().
    Input: none.
    Output: the fluxes at discrete ordinates and nodes.
    '''
    for i in range(self.Iter):
      # forward sweep into the slab
      for k in range(self.K):
        self.I_down(k)
      # reverse the diffuse transmissivity
      self.reverse()
      # backsweep out of the slab
      for k in range(self.K-1,-1,-1):
        self.I_up(k)
      # check for negativity in flux
      if np.min(self.Inodes) < 0.:
        # print self.Inodes
        print 'negative values in flux'
      # compute I_k+1/2 and J_k+1/2
      for k in range(self.K):
        self.Inodes[k,1] = (self.Inodes[k,0] + self.Inodes[k,2])/2.
        for j, v in enumerate(self.views):
          self.Jnodes[k,j] = self.J(v,self.views,self.Inodes[k,1])
      # acceleration can be implimented here...
      # check for convergence
      print self.Inodes[0,0]
      print 'iteration no: %d completed.' % (i+1)
      if self.converge():
        I_TOC = (self.Inodes[0,0,:self.n] + \
            self.Q2nodes[0,:self.n]) / -self.mu_s
        I_soil = (self.Inodes[self.K-1,2,self.n:] + \
            self.I_f(self.sun0,self.Lc,self.I0)) / -self.mu_s
        self.I_top_bottom = np.append(I_TOC,I_soil)
        print 'solution at iteration %d and saved in class.Inodes.'\
            % (i+1)
        #os.system('play --no-show-progress --null --channels 1 \
        #    synth %s sine %f' % ( 0.5, 500)) # ring the bell
        print 'TOC (up) and soil (down) fluxe array:'
        return self.I_top_bottom
        break
      else:
        # swap boundary for new flux
        self.Bounds[0] = self.Inodes[0,0]
        self.Bounds[1] = self.Inodes[self.K-1,2]
        continue

  def __del__(self):
    '''The post garbage collection method.
    '''
    print 'An instance of rt_layers has been destroyed.\n'
  
  def __repr__(self):
    '''This prints out the input parameters that define the 
    instance of the class.
    '''
    return '''Tol = %.e, Iter = %i, K = %i, N = %i, 
    Lc = %.3f, refl = %.3f, trans = %.3f, 
    refl_s = %.3f, I0 = %.4f,
    sun0 = %.3f, arch = %s''' % (self.Tol, self.Iter, self.K, \
        self.N, self.Lc, self.refl, self.trans, self.refl_s, \
        self.I0, self.sun0*180./np.pi, self.arch)

  def I_f(self, angle, L, I):
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
    i =  I * np.exp(G(angle,self.arch)*L/mu)
    return i

  def J(self, view, sun, Ia):
    '''The J or Distributed Source Term according to Myneni 1988b
    (23). This gives the multiple scattering as opposed to First 
    Collision term Q.
    Input: view - the zenith angle of evaluation, sun - the illumination
    zenith angles array, arch - archtype, refl - reflectance, trans - 
    transmittance, L - LAI, Ia - array of fluxes or intensities at
    the sun zenith angles with Ia[0] at sun = pi.
    Output: The J term.
    '''
    # expected that sun is a list of all incomming illumination
    # angles.
    index_view = self.angle_search(view)
    index_sun = self.angle_search(sun)
    if isinstance(sun, np.ndarray) and isinstance(Ia, np.ndarray):
      integ1 = np.multiply(Ia,self.Px[index_view,index_sun])
      integ = np.multiply(integ1,self.Gx[index_sun]/\
          self.Gx[index_view])
      # element-by-element multiplication
      # numerical integration by gaussian qaudrature
      j = self.albedo / 2. * np.sum(integ*self.gauss_wt)
    else:
      raise Exception('ArrayInputRequired')
    return j

  def Q1(self, view, L):
    '''The Q1 First First Collision Source Term as defined in Myneni
    1988b (24). This is the downwelling part of the Q term. 
    Input: view - the zenith angle of evaluation, sun0 - the uncollided
    illumination zenith angle, arch - archetype, refl - reflectance, 
    trans - transmittance, L - LAI, I0 - flux or intensity at TOC.
    Ouput: The Q1 term.
    '''
    index_view = self.angle_search(view)
    I = self.I_f(self.sun0, L, self.I0)
    q = self.albedo / 4. * self.Ps[index_view] * self.Gs/\
        self.Gx[index_view] * I
    return q

  def Q2(self, view, L):
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
    index_view = self.angle_search(view)
    index_sun = self.angle_search(self.sun_up,up=True)
    dL = self.Lc - L
    integ1 = np.multiply(self.Px[index_view,index_sun],\
        self.Gs/self.Gx[index_view]) 
        # element-by-element multipl.
    integ = np.multiply(integ1, self.I_f(self.sun_up, -dL, 1.)) # ^^
    q = self.albedo / 4. * -2. * self.refl_s * self.mu_s * \
        self.I_f(self.sun0, self.Lc, self.I0) * \
        np.sum(integ*self.gauss_wt[:self.n]) 
        # numerical integration by gaussian quadrature
    return q
    
  def Scalar_flux(self):
    '''A method to return the scalar fluxes of canopy reflection,
    soil absorption, and canopy absorption.
    Input: none.
    Output: canopy refl, soil absorp, canopy absorp.
    '''
    c_refl = np.sum(self.I_top_bottom[:self.n]*\
        self.gauss_wt[:self.n])
    down = np.sum(self.I_top_bottom[self.n:]*\
        self.gauss_wt[self.n:])
    up = np.sum(self.Inodes[self.K-1,2,:self.n]*\
        self.gauss_wt[:self.n])/-self.mu_s
    s_abs = down - up
    c_abs = self.I0 - c_refl - s_abs
    return (c_refl,s_abs,c_abs)

def plot_J(obj):
  '''A function that plots the J function.
  It requires the the number of intervals
  of the sun angle from 0. to pi, the archetype and reflection
  and transmission values and the Intensity or Flux.
  Input: obj - instance of rt_layers object.
  Output: plots the J function values.
  '''
  sun = np.linspace(np.pi,0.,obj.N)
  j = []
  view = sun.copy()
  Ia = np.zeros_like(view)
  index = round(np.pi - obj.sun0)*obj.N/np.pi
  Ia[index] = self.I0
  for v in view:
    j.append(obj.J(v,sun,Ia))
  plt.plot(sun,j,'--r')
  plt.show()

def plot_Q1(obj,L):
  '''A function that plots the Q1 function.
  It requires the single sun angle (not list) for uncollided
  Downwelling illumination, the number of view angles to 
  evaluate between 0 and pi, the archetype, the reflectance,
  transmittance, LAI, and Flux at the TOC.
  Input: obj - instance of rt_layers object, L - LAI.
  Output: Q1 term
  '''
  view = np.linspace(0.,np.pi,obj.N)
  q = []
  for v in view:
    q.append(obj.Q1(v, L))
  plt.plot(view,q,'--r')
  plt.show()

def plot_Q2(obj, L):
  '''A function that plots the Q2 function. To be noted is that
  this is the upwelling component of the first collision term Q.
  The graph would then portray greater scattering in the lower
  half towards view angles pi/2 to pi for the typical scenarios.
  Input: obj - instance of rt_layer object, L.
  Ouput: Q2 term.
  '''
  view = np.linspace(0.,np.pi,obj.N)
  q = []
  for v in view:
    q.append(obj.Q2(v, L))
  plt.plot(view,q,'--r')
  plt.show()

def plot_Q(obj,L):
  '''A function that plots the Q function as sum of Q1 and Q2.

  '''
  view = np.linspace(0.,np.pi,obj.N)
  q = []
  for v in view:
    q.append(obj.Q1(v, L)+\
          obj.Q2(v, L))
    plt.plot(view,q,'--r')
    plt.show()

def plot_brf(obj):
  '''A function to plot the upward and downward scattering at
  TOC and soil. 
  Input: rt_layer instance object.
  Output: plot of scattering.
  '''
  views = np.cos(obj.views)
  brf = obj.I_top_bottom
  fig, ax = plt.subplots(1)
  plt.plot(views,brf,'ro',label='BRF')
  plt.title("BRF at top and bottom node")
  s = obj.__repr__()
  props = dict(boxstyle='round',facecolor='wheat',alpha=0.5)
  plt.text(0.5,0.5,s,horizontalalignment='center',\
      verticalalignment='center',transform=ax.transAxes,\
      bbox=props)
  plt.xlabel(r"$\mu$ (cosine of exit zenith)")
  plt.ylabel(r'Refl.($\mu$+) and Trans.($\mu$-)')
  plt.show()

def plot_prof(obj):
  '''Function that plots a vertical profile of scattering from
  TOC to BOC.
  Input: rt_layer instance object.
  Output: plot of scattering.
  '''
  I = obj.Inodes[:,1,:] \
      / -obj.mu_s
  y = np.linspace(obj.Lc, 0., obj.K+1)
  x = obj.views*180./np.pi
  xm = np.array([])
  for i in np.arange(0,len(x)-1):
    nx = x[i] + (x[i+1]-x[i])/2.
    xm = np.append(xm,nx)
  xm = np.insert(xm,0,0.)
  xm = np.append(xm,180.)
  xx, yy = np.meshgrid(xm, y)
  plt.pcolormesh(xx,yy,I)
  plt.colorbar()
  plt.title('Canopy Fluxes')
  plt.xlabel('Exit Zenith Angle')
  plt.ylabel('Cumulative LAI (0=soil)')
  plt.arrow(135.,3.5,0.,-3.,head_width=5.,head_length=.2,\
      fc='k',ec='k')
  plt.text(140.,2.5,'Downwelling Flux',rotation=90)
  plt.arrow(45.,.5,0.,3.,head_width=5.,head_length=.2,\
      fc='k',ec='k')
  plt.text(35.,2.5,'Upwelling Flux',rotation=270)
  plt.show()

