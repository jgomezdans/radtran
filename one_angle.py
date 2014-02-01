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


  def __init__(self, Tol = 1.e-5, Iter = 200, K = 40, N = 16,\
      Lc = 1., refl = 0.4357, trans = 0.5089, refl_s = 0.35, I0 = 1.,\
      sun0 = 150., arch = 's'):
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
    self.Inodes = np.zeros((K,3,N)) # K, upper-mid-lower, N
    self.Jnodes = np.zeros((K,N))
    self.Q1nodes = self.Jnodes.copy()
    self.Q2nodes = self.Jnodes.copy()
    for (i, k) in enumerate(self.mid_ks):
      for (j, v) in enumerate(self.views): # may need to distinguish
        self.Q1nodes[i,j] = self.Q1(v,k) # between Q1 downward [n:]
        self.Q2nodes[i,j] = self.Q2(v,k) # and Q2 upward [:n].Pinty...:
    self.Bounds = np.zeros((2,N))

    # discrete ordinate equations
    g = G(self.views,arch)
    mu = np.cos(self.views) 
    self.a = (1. + g*dk/2./mu)/(1. - g*dk/2./mu)
    self.b = (g*dk/mu)/(1. + g*dk/2./mu)
    self.c = (g*dk/mu)/(1. - g*dk/2./mu)

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
    #print self.Inodes[k,2]
    #pdb.set_trace()
    if min(self.Inodes[k,2,n:]) < 0.:
      self.Inodes[k,2,n:] = np.where(self.Inodes[k,2,n:] < 0., \
        0., self.Inodes[k,2,n:]) # negative fixup need to revise....
      print 'Negative downward flux fixup performed at node %d' \
          %(k+1)
    #print self.Inodes[k,2]
    #pdb.set_trace()
    if k < self.K-1:
      self.Inodes[k+1,0,n:] = self.Inodes[k,2,n:]
  
  def I_up(self, k):
    '''The discrete ordinate upward equation.
    '''
    n = self.n
    self.Inodes[k,0,:n] = 1./self.a[:n]*self.Inodes[k,2,:n] + \
        self.b[:n]*(self.Jnodes[k,:n] + self.Q1nodes[k,:n] + \
        self.Q2nodes[k,:n])
    #print self.Inodes[k,0]
    #pdb.set_trace()
    if min(self.Inodes[k,0,:n]) < 0.:
      self.Inodes[k,0,:n] = np.where(self.Inodes[k,0,:n] < 0., \
         0., self.Inodes[k,0,:n]) # negative fixup need to revise...
    #print self.Inodes[k,0]
    #pdb.set_trace()
    if k != 0:
      self.Inodes[k-1,2,:n] = self.Inodes[k,0,:n]

  def reverse(self):
    '''Reverses the transmissivity at soil boundary.
    '''
    n = self.n
    Ir = np.multiply(np.cos(self.views[n:]),\
        self.Inodes[self.K-1,2,n:])
    self.Inodes[self.K-1,2,:n] = - 2. * self.refl_s * \
        np.sum(Ir*self.gauss_wt[n:]) 
        # revise: not sure I need gaussian quad wts here
    #print self.Inodes[self.K-1,2]
    #pdb.set_trace()

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
        pdb.set_trace()
      # compute I_k+1/2 and J_k+1/2
      for k in range(self.K):
        self.Inodes[k,1] = (self.Inodes[k,0] + self.Inodes[k,2])/2.
        for j, v in enumerate(self.views):
          self.Jnodes[k,j] = self.J(v,self.views,self.Inodes[k,1])
          # test the Shultis 1988 paper reevaluating J and Q
          #pdb.set_trace()
          #self.Q1nodes[k,j] = self.Q1(v,reval=True,k=k,i=j)
      #print self.Jnodes[0]
      #pdb.set_trace()
      # acceleration can be implimented here...
      # check for convergence
      print self.Inodes[0,0]
      print 'iteration no: %d completed.' % (i+1)
      if self.converge():
        self.Inodes[0,0,:self.n] += self.Q2nodes[0,:self.n]
        print 'solution at iteration %d and saved in class.Inodes.'\
            % (i+1)
        os.system('play --no-show-progress --null --channels 1 \
            synth %s sine %f' % ( 0.5, 500)) # ring the bell
        return self.Inodes[0,0]
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
    return '''Tol = %.e, Iter = %i, K = %i, N = %i, Lc = %.3f, 
refl = %.3f, trans = %.3f, refl_s = %.3f, I0 = %.4f,
sun0 = %.3f, arch = %s''' % (self.Tol, self.Iter, self.K, self.N, \
        self.Lc, self.refl, self.trans, self.refl_s, self.I0, \
        self.sun0, self.arch)

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
    if isinstance(sun, np.ndarray) and isinstance(Ia, np.ndarray):
      integ1 = np.multiply(Ia,P(view,sun,self.arch,self.refl,self.trans))
      integ = np.multiply(integ1,G(sun,self.arch)/G(view,self.arch))
      # element-by-element multiplication
      # numerical integration by gaussian qaudrature
      j = self.albedo / 2. * np.sum(integ*self.gauss_wt)
    else:
      raise Exception('ArrayInputRequired')
    return j

  def Q1(self, view, L=np.nan, reval=False, k=np.nan, i=np.nan):
    '''The Q1 First First Collision Source Term as defined in Myneni
    1988b (24). This is the downwelling part of the Q term. 
    Input: view - the zenith angle of evaluation, sun0 - the uncollided
    illumination zenith angle, arch - archetype, refl - reflectance, 
    trans - transmittance, L - LAI, I0 - flux or intensity at TOC.
    Ouput: The Q1 term.
    '''

    if reval:
      I = self.Inodes[k,1,i]
    else:
      I = self.I_f(self.sun0, L, self.I0)
    q = self.albedo / 4. * P(view,self.sun0,self.arch,self.refl,\
        self.trans)*G(self.sun0,self.arch)/G(view,self.arch) * I
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
    sun_up = self.views[:self.n]
    dL = self.Lc - L
    integ1 = np.multiply(P(view,sun_up,self.arch,self.refl,self.trans),\
        G(sun_up,self.arch)/G(view,self.arch)) 
        # element-by-element multipl.
    integ = np.multiply(integ1, self.I_f(sun_up, -dL, 1.)) # ^^
    q = self.albedo / 4. * -2. * self.refl_s * np.cos(self.sun0) * \
        self.I_f(self.sun0, self.Lc, self.I0) * \
        np.sum(integ*self.gauss_wt[:self.n]) 
        # numerical integration by gaussian quadrature
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
      j.append(self.J(v,sun,Ia))
    plt.plot(sun,j,'--r')
    plt.show()

  def plot_Q1(self,L):
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
    view = np.linspace(0.,np.pi,self.N)
    q = []
    for v in view:
      q.append(self.Q1(v, L))
    plt.plot(view,q,'--r')
    plt.show()

  def plot_Q2(self, L):
    '''A function that plots the Q2 function. To be noted is that
    this is the upwelling component of the first collision term Q.
    The graph would then portray greater scattering in the lower
    half towards view angles pi/2 to pi for the typical scenarios.
    Input: sun, n_angles, arch, refl trans, Lc, L, I0, rs.
    Ouput: Q2 term.
    '''
    view = np.linspace(0.,np.pi,self.N)
    q = []
    for v in view:
      q.append(self.Q2(v, L))
    plt.plot(view,q,'--r')
    plt.show()

  def plot_Q(self,L):
    '''A function that plots the Q function as sum of Q1 and Q2.

    '''
    view = np.linspace(0.,np.pi,self.N)
    q = []
    for v in view:
      q.append(self.Q1(v, L)+\
          self.Q2(v, L))
    plt.plot(view,q,'--r')
    plt.show()

  def plot_brf(self, k=0):
    '''A function to plot the upward and downward scattering at
    a specified node. If no parameters are passes assumes upward
    scattering at TOC and downward scattering at soil layer.
    Input: k - node number starting at TOC = 0.
    Output: plot of scattering.
    '''
    if k==0:
      k2 = self.K-1
    else:
      k2 = k
    views = self.views*180./np.pi
    flux = np.append(self.Inodes[k,0,:self.n],\
        self.Inodes[k2,2,self.n:])/-np.cos(self.sun0)
    brf = self.Q1nodes[0] + self.Q2nodes[0] + self.Q1nodes[0]
        # divide by sun angle
    plt.plot(views,brf,'rx',label='BRF')
    plt.plot(views,flux,'bx',label='Flux')
    plt.title('BRF at %d and Normalized Flux at %d & %d' %\
        (k,k,k2))
    plt.legend()
    plt.xlabel('View Zenith Angle')
    plt.ylabel('Refl. and Trans.')
    plt.show()

