#!/usr/bin/python

'''This will turn out to be the 3D discrete-ordinate exact-
kernel finite-difference implimentation of the RT equation as set
out in Myneni et al 1990. The script references the radtran.py module
which holds all the functions required for the calculations. It 
also requires the prospect_py.so module created using f2py.
'''

import numpy as np
import scipy as sc
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.mlab import griddata
import matplotlib.tri as tri
import warnings
import os
from radtran import *
import nose
from progressbar import *
import pdb
import prospect_py # prospect leaf rt module interface


class rt_layers():
  '''The class that will encapsulate the data and methods to be used
  for a single unit of radiative transfer through the canopy. To be 
  implimented is the constructor which takes the initial parameters,
  and a destructor to clear the memory of the instance once processing
  has completed.
  Input: Tol - max tolerance, Iter - max iterations, N - no of 
  discrete ordinates over 0 to pi, K - no of nodes, Lc - total LAI, 
  refl - leaf reflectance, trans - leaf transmittance, refl_s - 
  soil reflectance, F - total flux incident at TOC, Beta - fraction of 
  direct solar illumination, sun0_zen - zenith angle of solar illumination.
  Ouput: a rt_layers class.
  '''

  def __init__(self, Tol = 1.e-6, Iter = 200, N = 4, lad_file=\
      'scene_3D.dat', refl_s = 0.2, F = np.pi, Beta=1., \
      sun0_zen = 180., sun0_azi = 0., arch = 'u', ln = 1.2, \
      cab = 30., car = 10., cbrown = 0., cw = 0.015, cm = 0.009, \
      lamda = 760, refl = np.nan, trans = np.nan):
    '''The constructor for the rt_layers class.
    See the class documentation for details of inputs.
    '''
    self.Tol = Tol
    self.Iter = Iter
    self.K = K # !!!
    if int(N) % 2 == 1:
      N = int(N)+1
      print 'N rounded up to even number:', str(N)
    self.N = N
    self.Lc = Lc # !!!
    # choose between PROSPECT or refl trans input
    if np.isnan(refl) or np.isnan(trans):
      self.ln = ln
      self.cab = cab
      self.car = car
      self.cbrown = cbrown
      self.cw = cw
      self.cm = cm
      self.lamda = lamda
      refl, trans = prospect_py.prospect_5b(ln, cab, car, cbrown, cw,\
          cm)[lamda-401]
    else:
      self.ln = np.nan
      self.cab = np.nan
      self.car = np.nan
      self.cbrown = np.nan
      self.cw = np.nan
      self.cm = np.nan
      self.lamda = np.nan
    self.refl = refl
    self.trans = trans
    self.refl_s = refl_s
    self.Beta = Beta
    self.sun0_zen = sun0_zen * np.pi / 180.
    self.sun0_azi = sun0_azi * np.pi / 180.
    self.sun0 = np.array([self.sun0_zen, self.sun0_azi])
    self.mu_s = np.cos(self.sun0_zen)
    self.I0 = Beta * F / np.pi # did not include mu_s here
    # which is part of original text. mu_s skews total. more 
    # likely to get proportion of sun at elevation.
    self.F = F
    # it is assumed that illumination at TOC will be provided prior.
    self.Id = (1. - Beta) * F / np.pi
    self.arch = arch
    self.albedo = self.refl + self.trans # leaf single scattering albedo
    f = open('quad_dict.dat')
    quad_dict = pickle.load(f)
    f.close()
    quad = quad_dict[str(N)]
    self.gauss_wt = quad[:,2] / 8.# per octant. 
    # see Lewis 1984 p.172 eq(4-40) 
    self.gauss_mu = np.cos(quad[:,0])
    self.views = np.array(zip(quad[:,0],quad[:,1]))

    # intervals
    dk = Lc/K # !!!
    self.mid_ks = np.arange(dk/2.,Lc,dk) # !!!
    self.n = N*(N + 2) # !!! self.N/2 # needs to be mid of angle pairs
    
    # node arrays and boundary arrays
    self.views_zen = quad[:,0]
    self.views_azi = quad[:,1]
    self.sun_up = self.views[:self.n/2]
    self.sun_down = self.views[self.n/2:]
    self.Ird = -np.sum(np.multiply(self.refl_s * 2. * \
        np.multiply(self.gauss_mu[self.n/2:],\
        self.I_f(self.sun_down, self.Lc, self.Id)),\
        self.gauss_wt[self.n/2:])) # !!! set when reach bottom.
    self.Ir0 = self.refl_s * -self.mu_s * \
        self.I_f(self.sun0, self.Lc, self.I0) # !!! same
    # discrete ordinate equations
    g = G(self.views_zen,self.arch)
    mu = self.gauss_mu 
    # !!! abc changes per lai in cell
    self.a = (1. + g*dk/2./mu)/(1. - g*dk/2./mu) 
    self.b = (g*dk/mu)/(1. + g*dk/2./mu) 
    self.c = (g*dk/mu)/(1. - g*dk/2./mu)
    # build in validation of N to counter negative fluxes.
    if any((g * dk / 2. / mu) > 1.):
      raise Exception('NegativeFluxPossible')
    # G-function cross-sections
    self.Gx = g
    self.Gs = G(self.sun0_zen,arch)
    self.Inodes = np.zeros((K,3,self.n)) # K, upper-mid-lower, N
    self.Jnodes = np.zeros((K,self.n))
    self.Q1nodes = self.Jnodes.copy()
    self.Q2nodes = self.Jnodes.copy()
    self.Q3nodes = self.Jnodes.copy()
    self.Q4nodes = self.Jnodes.copy()
    self.Px = np.zeros((self.n,self.n)) # P cross-section array
    self.Ps = np.zeros(self.n) # P array for solar zenith
    # a litte progress bar for long calcs
    widgets = ['Progress: ', Percentage(), ' ', \
      Bar(marker='0',left='[',right=']'), ' ', ETA()] 
    maxval = np.shape(self.views)[0] * (np.shape(self.mid_ks)[0] + \
        np.shape(self.views)[0] + 1) + 1
    pbar = ProgressBar(widgets=widgets, maxval = maxval)
    count = 0
    print 'Setting-up Phase and Q term arrays....'
    pbar.start()
    for (i,v) in enumerate(self.views):
      count += 1
      pbar.update(count)
      self.Ps[i] = P2(v,self.sun0,self.arch,self.refl,\
        self.trans)
    for (i,v1) in enumerate(self.views):
      for (j,v2) in enumerate(self.views):
        count += 1
        pbar.update(count)
        self.Px[i,j] = P2(v1, v2 ,self.arch, self.refl, self.trans)
    for (i, k) in enumerate(self.mid_ks):
      for (j, v) in enumerate(self.views):
        count += 1
        pbar.update(count)
        # the factors to the right were found through trial and 
        # error. They make calculations work....
        self.Q1nodes[i,j] = self.Q1(v,k) * np.pi / 2. 
        self.Q2nodes[i,j] = self.Q2(v,k) * np.pi / 2. * 3.
        self.Q3nodes[i,j] = self.Q3(v,k) * np.pi / 2.
        self.Q4nodes[i,j] = self.Q4(v,k) * np.pi / 2. * 3.
    pbar.finish()
    self.Bounds = np.zeros((2,self.n))
    os.system('play --no-show-progress --null --channels 1 \
            synth %s sine %f' % ( 0.5, 500)) # ring the bell

 
  # function to search angle database for index
  def angle_search(self,v):
    '''A method that provides the index of an angle in the
    views array. If up is True then provides the index 
    relative to the start of the array for upward angles,
    if False then it's relative to the middle or down
    direction.
    Input: v - angle to search, up - True/False.
    Output: index of angle in views array.
    '''
    for i, view in enumerate(self.views):
      if np.max(np.abs(v - view)) < 1.e-5:
        return i

  def sun0_zen(self,sun0_zen):
    '''Method used for entering solar insolation angle which
    takes care of conversion to radians. Try not to assign 
    angles directly to self.sun0_zen variable but use this method.
    Input: sun0_zen - solar zenith angle in degrees.
    Output: converts and stores value in self.sun0_zen.
    '''
    self.sun0_zen = sun0_zen*np.pi/180.

  def I_down(self, k):
    '''The discrete ordinate downward equation.
    '''
    n = self.n/2
    self.Inodes[k,2,n:] = self.a[n:]*self.Inodes[k,0,n:] - \
        self.c[n:]*(self.Jnodes[k,n:] + self.Q1nodes[k,n:] + \
        self.Q2nodes[k,n:] + self.Q3nodes[k,n:] + \
        self.Q4nodes[k,n:])
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
    n = self.n/2
    self.Inodes[k,0,:n] = 1./self.a[:n]*self.Inodes[k,2,:n] + \
        self.b[:n]*(self.Jnodes[k,:n] + self.Q1nodes[k,:n] + \
        self.Q2nodes[k,:n] + self.Q3nodes[k,:n] + \
        self.Q4nodes[k,:n])
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
    Ir = np.multiply(np.cos(self.views[self.n/2:,0]),\
        self.Inodes[self.K-1,2,self.n/2:])
    self.Inodes[self.K-1,2,:self.n/2] = - 2. * self.refl_s * \
        np.sum(np.multiply(Ir,self.gauss_wt[self.n/2:]))

  def converge(self):
    '''Check for convergence and returns true if converges.
    '''
    misclose_top = np.abs((self.Inodes[0,0] - self.Bounds[0])/\
        self.Inodes[0,0])
    misclose_bot = np.abs((self.Inodes[self.K-1,2] - \
        self.Bounds[1])/self.Inodes[self.K-1,2])
    max_top = np.nanmax(misclose_top)
    max_bot = np.nanmax(misclose_bot)
    print 'misclosures top: %.g, and bottom: %.g.' %\
        (max_top, max_bot)
    if max_top  <= self.Tol and max_bot <= self.Tol:
      return True
    else:
      return False
  
  def solve(self):
    '''The solver. Run this as a method of the instance of the
    rt_layers class to solve the RT equations. You first need
    to create an instance of the class though using:
    eg. test = rt_layers() # see rt_layers ? for more options.
    then run test.solve().
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
          self.Jnodes[k,j] = self.J(v,self.Inodes[k,1])
      # acceleration can be implimented here...
      # check for convergence
      #print self.Inodes[0,0]
      print 'iteration no: %d completed.' % (i+1)
      if self.converge():
        # see Myneni 1989 p 95 for integration of canopy and 
        # soil fluxes below. The fact was found to be linear
        # through trial and error.
        fact = (-1./self.mu_s - 1.) * self.Beta + 1.
        I_TOC = self.Inodes[0,0,:self.n/2] * fact  + \
            self.Q3nodes[0,:self.n/2] / -self.mu_s  + \
            self.Q4nodes[0,:self.n/2]
        I_soil = (self.Inodes[self.K-1,2,self.n/2:] * fact + \
            self.I_f(self.sun0,self.Lc,self.I0) * -self.mu_s + \
            self.I_f(self.views[self.n/2:],self.Lc,self.Id) * \
            -np.cos(self.views[self.n/2:,0]))\
            * (1. - self.refl_s) # needs diff term here.
        self.I_top_bottom = np.append(I_TOC,I_soil)
        print 'solution at iteration %d and saved in class.Inodes.'\
            % (i+1)
        os.system('play --no-show-progress --null --channels 1 \
            synth %s sine %f' % ( 0.5, 500)) # ring the bell
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
    return \
    '''Tol = %.e, Iter = %i, K = %i, N = %i, Beta = %.3f, Lc = %.3f, 
    refl = %.3f, trans = %.3f, refl_s = %.3f, F = %.4f, sun0_zen = %.3f,
    sun0_azi = %.3f, arch = %s, ln = %.2f, cab = %.2f, car = %.2f, 
    cbrown = %.2f, cw = %.3f, cm = %.3f, lamda = %i''' \
        % (self.Tol, self.Iter, self.K, self.N, self.Beta,\
        self.Lc, self.refl, self.trans, self.refl_s, \
        self.F, self.sun0[0]*180./np.pi, self.sun0[1]*180./np.pi,\
        self.arch, self.ln, self.cab, self.car, self.cbrown, \
        self.cw, self.cm, self.lamda)

  def I_f(self, view, L, I):
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
    if np.size(view) > 2:
      angle = view[:,0]
    else:
      angle = view[0]
    mu = np.cos(angle)
    i =  I * np.exp(G(angle,self.arch)*L/mu)
    return i

  def J(self, view, Ia):
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
    if isinstance(Ia, np.ndarray):
      integ1 = np.multiply(Ia,self.Px[index_view])
      integ = np.multiply(integ1,self.Gx/\
          self.Gx[index_view])
      # element-by-element multiplication
      # numerical integration by gaussian qaudrature
      j = self.albedo / 2. * np.sum(integ*self.gauss_wt)
    else:
      raise Exception('ArrayInputRequired')
    return j

  def Q1(self, view, L):
    '''The Q1 First First Collision Source Term as defined in Myneni
    1988d (16). This is the downwelling direct part of the Q term. 
    Input: view - the zenith angle of evaluation, sun0_zen - the uncollided
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
    Myneni 1988d (17). This is the downwelling diffuse part of 
    the Q term.
    Input:
    Output:
    '''
    index_view = self.angle_search(view)
    integ1 = np.multiply(self.Px[index_view,self.n/2:],\
        self.Gx[self.n/2:]/self.Gx[index_view])
    integ = np.multiply(integ1, self.I_f(self.sun_down, L, \
        self.Id))
    q = self.albedo / 4. * np.sum(np.multiply(integ,\
        self.gauss_wt[self.n/2:]))
    return q

  def Q3(self, view, L): 
    '''The Q3 Third First Collision Source Term as defined in
    Myneni 1988d (18). This is the upwelling direct part of 
    the Q term.
    Input: view - the view zenith angle of evalution, sun0_zen - the 
    direct illumination zenith angle down, n_angles - the number of
    angles between 0 and pi, arch - archetype, refl - 
    reflectance, trans - transmittance, Lc - total LAI, L - LAI,
    I0 - flux or intensity of direct sun ray at TOC, refl_s - 
    soil reflectance.
    Output: The Q3 term.
    '''
    index_view = self.angle_search(view)
    dL = self.Lc - L
    integ1 = np.multiply(self.Px[index_view,:self.n/2],\
        self.Gx[:self.n/2]/self.Gx[index_view]) 
        # element-by-element multipl.
    integ = np.multiply(integ1, self.I_f(self.sun_up, -dL,\
        self.Ir0)) # ^^
    q = self.albedo / 4. * np.sum(np.multiply(integ, \
        self.gauss_wt[:self.n/2]))
        # numerical integration by gaussian quadrature
    return q

  def Q4(self, view, L):
    '''placeholder
    '''
    index_view = self.angle_search(view)
    dL = self.Lc - L
    integ1 = np.multiply(self.Px[index_view,:self.n/2],\
        self.Gx[:self.n/2]/self.Gx[index_view])
    integ = np.multiply(integ1, self.I_f(self.sun_up, -dL,\
        self.Ird))
    q = self.albedo / 4. * np.sum(np.multiply(integ, \
        self.gauss_wt[:self.n/2]))
    return q
    
  def Scalar_flux(self):
    '''A method to return the scalar fluxes of canopy reflection,
    soil absorption, and canopy absorption.
    Input: none.
    Output: canopy refl, soil absorp, canopy absorp.
    '''
    c_refl = np.sum(self.I_top_bottom[:self.n/2]*\
        self.gauss_wt[:self.n/2])
    s_abs = np.sum(self.I_top_bottom[self.n/2:]*\
        self.gauss_wt[self.n/2:]) * 2.
    c_abs = self.I0 + self.Id - c_refl - s_abs
    return (c_refl,s_abs,c_abs)

def plot_sphere(obj):
  '''A function that plots the full BRF over the upper and lower
  hemispheres. 
  Input: rt_layers object.
  Output: spherical plot of brf.
  '''
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  z = np.cos(obj.views[:,0])
  x = np.cos(obj.views[:,1]) * np.sqrt(1. - z**2)
  y = np.sin(obj.views[:,1]) * np.sqrt(1. - z**2)
  c = obj.I_top_bottom
  scat = ax.scatter(x, y, z, c=c)
  ax.set_xlabel('X axis')
  ax.set_ylabel('Y axis')
  ax.set_zlabel('Z axis')
  plt.title('BRF over the sphere')
  plt.colorbar(scat, shrink=0.5, aspect=10)
  plt.show()

def plot_contours(obj):
  '''A function that plots the BRF as an azimuthal projection
  with contours over the TOC and soil.
  Input: rt_layers object.
  Output: contour plot of brf.
  '''
  sun = ((np.pi - obj.sun0[0]) * np.cos(obj.sun0[1] + np.pi), \
      (np.pi - obj.sun0[0]) * np.sin(obj.sun0[1] + np.pi))
  theta = obj.views[:,0]
  x = np.cos(obj.views[:,1]) * theta
  y = np.sin(obj.views[:,1]) * theta
  z = obj.I_top_bottom * -obj.mu_s
  if np.max > 1.:
    maxz = np.max(z)
  else:
    maxz = 1.
  minz = 0. #np.min(z)
  space = np.linspace(minz, maxz, 11)
  x = x[:obj.n/2]
  y = y[:obj.n/2]
  zt = z[:obj.n/2]
  zb = z[obj.n/2:]
  fig = plt.figure()
  plt.subplot(121)
  plt.plot(sun[0], sun[1], 'ro')
  triang = tri.Triangulation(x, y)
  plt.gca().set_aspect('equal')
  plt.tricontourf(triang, zt, space, vmax=maxz, vmin=minz)
  plt.title('TOC BRF')
  plt.ylabel('Y')
  plt.xlabel('X')
  plt.subplot(122)
  plt.plot(sun[0], sun[1], 'ro')
  plt.gca().set_aspect('equal')
  plt.tricontourf(triang, zb, space, vmax=maxz, vmin=minz)
  plt.title('Soil Absorption')
  plt.ylabel('Y')
  plt.xlabel('X')
  s = obj.__repr__()
  plt.suptitle(s)
  cbaxes = fig.add_axes([0.11,0.1,0.85,0.05])
  plt.colorbar(orientation='horizontal', ticks=space,\
      cax = cbaxes)
  #plt.tight_layout()
  plt.show()

# The function below have not yet been converted to the two-angle
# case.

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
  index = round(np.pi - obj.sun0_zen)*obj.N/np.pi
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

def plot_Q3(obj, L):
  '''A function that plots the Q3 function. To be noted is that
  this is the upwelling component of the first collision term Q.
  The graph would then portray greater scattering in the lower
  half towards view angles pi/2 to pi for the typical scenarios.
  Input: obj - instance of rt_layer object, L.
  Ouput: Q3 term.
  '''
  view = np.linspace(0.,np.pi,obj.N)
  q = []
  for v in view:
    q.append(obj.Q3(v, L))
  plt.plot(view,q,'--r')
  plt.show()

def plot_Q(obj,L):
  '''A function that plots the Q function as sum of Q1 and Q3.

  '''
  view = np.linspace(0.,np.pi,obj.N)
  q = []
  for v in view:
    q.append(obj.Q1(v, L)+\
          obj.Q3(v, L))
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

