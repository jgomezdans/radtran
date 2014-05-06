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
  Input: Tol - max tolerance, Iter - max iterations, N - quadrature
  levels, LAD file, arch - archetype, ln - leaf layers, cab - 
  chlorophyll a+b, car - carotenoids, cbrown - brown matter, cw -
  water content, cm - dry mass, lambda - frequency in nm, 
  refl - leaf reflectance, trans - leaf transmittance, refl_s - 
  soil reflectance, F - total flux incident at TOC, Beta - fraction of 
  direct solar illumination, sun0_zen - zenith angle of solar illumination.
  Ouput: a rt_layers class.
  '''

  def __init__(self, Tol = 1.e-6, Iter = 400, N = 4, lad_file=\
      'scene_3D.dat', refl_s = 0., F = np.pi, Beta=1., \
      sun0_zen = 180., sun0_azi = 0., arch = 'u', ln = 1.2, \
      cab = 30., car = 10., cbrown = 0., cw = 0.015, cm = 0.009, \
      lamda = 760, refl = 0.5, trans = 0.5):
    '''The constructor for the rt_layers class.
    See the class documentation for details of inputs.
    '''
    self.Tol = Tol
    self.Iter = Iter
    if int(N) % 2 == 1:
      N = int(N)+1
      print 'N rounded up to even number:', str(N)
    self.N = N
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
    # see Lewis 1984 p.172 eq(4-40) and notes on 13/2/14 
    self.gauss_mu = np.cos(quad[:,0])
    self.gauss_eta = np.sqrt(1.-self.gauss_mu**2) * np.sin(quad[:,1])
    self.gauss_xi = np.sqrt(1.-self.gauss_mu**2) * np.cos(quad[:,1])
    self.views = np.array(zip(quad[:,0],quad[:,1]))

    # load LAD file
    self.lad_file = lad_file
    f = open(lad_file)
    dic = pickle.load(f)
    f.close()

    self.mesh = dic['mesh'] # coordinates of midpoints of cells in xyz

    # intervals
    self.dx = self.mesh[0,1] - self.mesh[0,0] # dk in 2 angle
    self.dy = self.mesh[1,1] - self.mesh[1,0]
    self.dz = self.mesh[2,1] - self.mesh[2,0]
    # grid of cell values in xyz divided by cell volume for lad
    self.grid = dic['grid'] / (self.dx * self.dy * self.dz)
    self.mid_xs = self.mesh[0] # mid_ks in 2 angle
    self.mid_ys = self.mesh[1]
    self.mid_zs = self.mesh[2]
    self.k = len(self.mid_xs)
    self.j = len(self.mid_ys)
    self.i = len(self.mid_zs)
    self.n = N*(N + 2) # total directions
    # directions grouped per octant
    self.octi = np.reshape(np.arange(0,self.n), (8,-1))
    # edges grouped per octant o from, t to. see notes 4/5/14
    self.edgo = np.array([[4,5,2],[0,5,2],[0,1,2],[4,1,2],[4,5,6],\
        [0,5,6],[0,1,6],[4,1,6]])
    self.edgt = np.array([[0,1,6],[4,1,6],[4,5,6],[0,5,6],[0,1,2],\
        [4,1,2],[4,5,2],[0,5,2]])
    # end cube flux transfer octants and views. see notes 4/5/14
    self.xa = np.array([self.octi[it] for it in [0,3,4,7]]).flatten()
    self.xb = np.array([self.octi[it] for it in [1,2,5,6]]).flatten()
    self.ya = np.array([self.octi[it] for it in [0,1,4,5]]).flatten()
    self.yb = np.array([self.octi[it] for it in [2,3,6,7]]).flatten()
    # node arrays and boundary arrays
    self.views_zen = quad[:,0]
    self.views_azi = quad[:,1]
    # splits views into octants and sun_up and sun_down
    self.sun_down = self.views[self.n/2:]
    self.sun_up = self.views[:self.n/2]
    self.octv = np.reshape(self.views, (8, -1, 2)) # per octant views
    self.octw = np.reshape(self.gauss_wt, (8, -1)) # per octant wt
    # G-function cross-sections
    self.Gx = G(self.views_zen,self.arch) 
    self.Gs = G(self.sun0_zen,arch)
    # see notes on 29/04/14 for figure of Inode layout
    self.Inodes = np.zeros((self.k, self.j, self.i, 7, self.n)) 
    self.Jnodes = np.zeros((self.k, self.j, self.i, self.n))
    self.Q1nodes = self.Jnodes.copy()
    self.Q2nodes = self.Jnodes.copy()
    self.Q3nodes = self.Jnodes.copy()
    self.Q4nodes = self.Jnodes.copy()
    # use Gamma/pi as function and * by ul per cell to get cross sect
    self.Gampi_x = np.zeros((self.n,self.n)) # part cross-section array
    self.Gampi_s = np.zeros(self.n) # part of array for solar zenith
    # a litte progress bar for long calcs
    widgets = ['Progress: ', Percentage(), ' ', \
      Bar(marker='0',left='[',right=']'), ' ', ETA()] 
    maxval = np.shape(self.views)[0] * (1 + np.shape(self.views)[0] +\
        self.i*self.j*self.k) + 1
    pbar = ProgressBar(widgets=widgets, maxval = maxval)
    count = 0
    print 'Setting-up Phase and Q term arrays....'
    pbar.start()
    for (i,v) in enumerate(self.views):
      count += 1
      pbar.update(count)
      self.Gampi_s[i] = Gamma2(v,self.sun0,self.arch,self.refl,\
        self.trans) / np.pi # sun view part
    for (i,v1) in enumerate(self.views):
      for (j,v2) in enumerate(self.views):
        count += 1
        pbar.update(count)
        self.Gampi_x[i,j] = Gamma2(v1, v2 ,self.arch, self.refl,\
            self.trans) / np.pi # all other views parts
    self.tot_lai_grid = np.sum(self.grid, 2)
    for k in np.arange(0,self.k): # x axis
      for j in np.arange(0,self.j): # y axis
        tot_lai = self.tot_lai_grid[k,j]
        lai = self.grid[k,j]
        cum_lai = np.insert(np.cumsum(self.grid[k,j]), 0, 0.)
        # soil flux needs to be calculated per bottom cell
        Ird = self.refl_s / np.pi * np.sum(np.multiply(\
            np.multiply(-self.gauss_mu[self.n/2:],\
            self.I_f(self.sun_down, tot_lai, self.Id)),\
            self.gauss_wt[self.n/2:])) # diffuse part
        Ir0 = self.refl_s * -self.mu_s / np.pi * \
            self.I_f(self.sun0, tot_lai, self.I0) # direct part
        for i, z in enumerate(self.mid_zs): # z axis
          # test for zero lai pockets, keep scattering at zero
          if np.allclose(0.,lai[i], rtol=1e-08, atol=1e-08):
            count += 1*self.n
            print 'zero lai encountered at cell: (%d, %d, %d)' %(k,j,i)
            pbar.update(count)
            continue
          mid_lai = lai[i]/2. + cum_lai[i]
          pnt_lai = lai[i]
          for l, v in enumerate(self.views):
            count += 1
            pbar.update(count)
            self.Q1nodes[k,j,i,l] = self.Q1(v,l,pnt_lai,mid_lai) 
            self.Q2nodes[k,j,i,l] = self.Q2(v,l,pnt_lai,mid_lai)
            self.Q3nodes[k,j,i,l] = self.Q3(l,pnt_lai,mid_lai,\
                tot_lai, Ir0)
            self.Q4nodes[k,j,i,l] = self.Q4(l,pnt_lai,mid_lai,\
                tot_lai, Ird)
    pbar.finish()
    self.PrevInodes = self.Inodes.copy()
    os.system('play --no-show-progress --null --channels 1 \
            synth %s sine %f' % ( 0.5, 500)) # ring the bell

  def sun0_zen(self,sun0_zen):
    '''Method used for entering solar insolation angle which
    takes care of conversion to radians. Try not to assign 
    angles directly to self.sun0_zen variable but use this method.
    Input: sun0_zen - solar zenith angle in degrees.
    Output: converts and stores value in self.sun0_zen.
    '''
    self.sun0_zen = sun0_zen*np.pi/180.

  def I_down(self, k, j, i):
    '''The discrete ordinate downward equation. The method is based on
    Myneni et al 1990 (70), and 1991 (17), and Lewis 1984 (4-53)
    as well as other neutron papers.
    '''
    # iteration is per octant with edges affecting the octant as edo
    for ind, (edo,edt) in enumerate(zip(self.edgo,self.edgt)):
      xo, yo, zo = edo # the edges from
      xt, yt, zt = edt # the edges to
      vi = self.octi[ind] # the view indexes for this octant
      Zf = 2. * np.abs(self.gauss_mu[vi]) / self.dz
      Yf = 2. * np.abs(self.gauss_eta[vi]) / self.dy
      Xf = 2. * np.abs(self.gauss_xi[vi]) / self.dx
      self.Inodes[k,j,i,3,vi] = (self.Q1nodes[k,j,i,vi] + \
          self.Q2nodes[k,j,i,vi] + self.Q3nodes[k,j,i,vi] + \
          self.Q4nodes[k,j,i,vi] + self.Jnodes[k,j,i,vi] + Zf * \
          self.Inodes[k,j,i,zo,vi] + Yf * self.Inodes[k,j,i,yo,vi] + \
          Xf * self.Inodes[k,j,i,xo,vi]) /\
          (self.Gx[vi] * self.grid[k,j,i] + Zf + Yf + Xf) 
      self.Inodes[k,j,i,xt,vi] = 2. * self.Inodes[k,j,i,3,vi] - \
          self.Inodes[k,j,i,xo,vi]
      self.Inodes[k,j,i,yt,vi] = 2. * self.Inodes[k,j,i,3,vi] - \
          self.Inodes[k,j,i,yo,vi]
      self.Inodes[k,j,i,zt,vi] = 2. * self.Inodes[k,j,i,3,vi] - \
          self.Inodes[k,j,i,zo,vi]
      # negative flux fixup need to revise....
      if min(self.Inodes[k,j,i,edt,vi]) < 0.:
        self.Inodes[k,j,i,edt,vi] = np.where(self.Inodes[k,j,i,edt,vi]\
            < 0., 0., self.Inodes[k,j,i,edt,vi]) 
        print 'Negative downward flux fixup performed at %d,%d,%d,%d' \
          %(k,j,i,ind)
      if k < self.k-1:
        self.Inodes[k+1,j,i,edo,vi] = self.Inodes[k,j,i,edt,vi]
      if j < self.j-1:
        self.Inodes[k,j+1,i,edo,vi] = self.Inodes[k,j,i,edt,vi]
      if i < self.i-1:
        self.Inodes[k,j,i+1,edo,vi] = self.Inodes[k,j,i,edt,vi]

  
  def I_up(self, k, j, i):
    '''The discrete ordinate upward equation.
    '''
    # iteration is per octant with edges affecting the octant as edt
    for ind, (edo,edt) in enumerate(zip(self.edgo,self.edgt)):
      xo, yo, zo = edo # the edges to
      xt, yt, zt = edt # the edges from
      vi = self.octi[ind] # the view indexes for this octant
      Zf = 2. * np.abs(self.gauss_mu[vi]) / self.dz
      Yf = 2. * np.abs(self.gauss_eta[vi]) / self.dy
      Xf = 2. * np.abs(self.gauss_xi[vi]) / self.dx
      self.Inodes[k,j,i,3,vi] = (self.Q1nodes[k,j,i,vi] + \
          self.Q2nodes[k,j,i,vi] + self.Q3nodes[k,j,i,vi] + \
          self.Q4nodes[k,j,i,vi] + self.Jnodes[k,j,i,vi] + Zf * \
          self.Inodes[k,j,i,zt,vi] + Yf * self.Inodes[k,j,i,yt,vi] + \
          Xf * self.Inodes[k,j,i,xt,vi]) /\
          (self.Gx[vi] * self.grid[k,j,i] + Zf + Yf + Xf) 
      self.Inodes[k,j,i,xo,vi] = 2. * self.Inodes[k,j,i,3,vi] - \
          self.Inodes[k,j,i,xt,vi]
      self.Inodes[k,j,i,yo,vi] = 2. * self.Inodes[k,j,i,3,vi] - \
          self.Inodes[k,j,i,yt,vi]
      self.Inodes[k,j,i,zo,vi] = 2. * self.Inodes[k,j,i,3,vi] - \
          self.Inodes[k,j,i,zt,vi]
      # negative flux fixup need to revise....
      if min(self.Inodes[k,j,i,edo,vi]) < 0.:
        self.Inodes[k,j,i,edo,vi] = np.where(self.Inodes[k,j,i,edo,vi]\
            < 0., 0., self.Inodes[k,j,i,edo,vi]) 
        print 'Negative upward flux fixup performed at %d,%d,%d,%d' \
          %(k,j,i,ind)
      if k != 0:
        self.Inodes[k-1,j,i,edt,vi] = self.Inodes[k,j,i,edo,vi]
      if j != 0:
        self.Inodes[k,j-1,i,edt,vi] = self.Inodes[k,j,i,edo,vi]
      if i != 0:
        self.Inodes[k,j,i-1,edt,vi] = self.Inodes[k,j,i,edo,vi]
    
  def reverse(self):
    '''Reverses the transmissivity at soil boundary.
    '''
    for k, x in enumerate(self.mid_xs):
      for j, y in enumerate(self.mid_ys):
        Ir = np.multiply(-np.cos(self.views[self.n/2:,0]),\
            self.Inodes[k,j,self.i-1,2,self.n/2:])
        self.Inodes[k,j,self.i-1,2,:self.n/2] = self.refl_s * \
            np.sum(np.multiply(Ir,self.gauss_wt[self.n/2:]))
        # transfer fluxes for continuous scenes

  def converge(self):
    '''Check for convergence and returns true if converges.
    '''
    misclose_I = np.abs((self.Inodes - self.PrevInodes)/\
        self.Inodes)
    max_I = np.nanmax(misclose_I)
    print 'max misclosure : %.g' % (max_I)
    if max_I  <= self.Tol:
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
    for h in range(self.Iter):
      # forward sweep into the slab
      for i, mz in enumerate(self.mid_zs):
        for j, my in enumerate(self.mid_ys):
          for k, mx in enumerate(self.mid_xs):
            self.I_down(k,j,i)
      # reverse the diffuse? transmissivity and transfer fluxes
      self.reverse()
      # backsweep out of the slab
      for i, mz in enumerate(self.mid_zs[::-1]):
        for j, my in enumerate(self.mid_ys[::-1]):
          for k, mx in enumerate(self.mid_xs[::-1]):
            self.I_up(k,j,i)
      # transfer fluxes from one end to the other of cube
      '''self.Inodes[0,:,:,4,self.xa] = self.Inodes[self.k-1,:,:,0,self.xa]
      self.Inodes[self.k-1,:,:,0,self.xb] = self.Inodes[0,:,:,4,self.xb]
      self.Inodes[:,0,:,5,self.ya] = self.Inodes[:,self.j-1,:,1,self.ya]
      self.Inodes[:,self.j-1,:,1,self.yb] = self.Inodes[:,0,:,5,self.yb]'''
      # check for negativity in flux
      if np.min(self.Inodes) < 0.:
        # print self.Inodes
        print 'negative values in flux'
            # check for convergence
      print 'iteration no: %d completed.' % (h+1)
      if self.converge():
        # see Myneni 1989 p 95 for integration of canopy and 
        # soil fluxes below. The fact was found to be linear
        # through trial and error.
        fact = (-1./self.mu_s - 1.) * self.Beta + 1.
        self.I_TOC = np.zeros((self.k,self.j,self.n/2))
        self.I_soil = self.I_TOC.copy()
        for k, xm in enumerate(self.mid_xs):
          for j, ym in enumerate(self.mid_ys):
            self.I_TOC[k,j] = self.Inodes[k,j,0,6,:self.n/2] * fact  + \
                self.Q3nodes[k,j,0,:self.n/2] / -self.mu_s  + \
                self.Q4nodes[k,j,0,:self.n/2]
            self.I_soil[k,j] = (self.Inodes[k,j,self.i-1,2,self.n/2:]\
                * fact + self.I_f(self.sun0,self.tot_lai_grid[k,j], \
                self.I0) * -self.mu_s + self.I_f(self.views[self.n/2:],\
                self.tot_lai_grid[k,j],self.Id) * 
                -np.cos(self.views[self.n/2:,0]))\
                * (1. - self.refl_s) # needs diff term here.
        print 'solution at iteration %d and saved in class.Inodes.'\
            % (i+1)
        os.system('play --no-show-progress --null --channels 1 \
            synth %s sine %f' % ( 0.5, 500)) # ring the bell
        print 'TOC (up) and soil (down) fluxe array:'
        return (self.I_TOC, self.I_soil)
        break
      else:
        # swap boundary for new flux
        self.PrevInodes = self.Inodes.copy()
        # do we need to average the flux at cell centre here?
        # compute new multiple scattering term J based on I
        for i, mz in enumerate(self.mid_zs):
          for j, my in enumerate(self.mid_ys):
            for k, mx in enumerate(self.mid_xs):
              for l, v in enumerate(self.views):
                self.Jnodes[k,j,i,l] = self.J(v,l,k,j,i)
        # acceleration can be implimented here...
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
    '''Tol = %.e, Iter = %i, lad_file = %s, N = %i, Beta = %.3f, 
    refl = %.3f, trans = %.3f, refl_s = %.3f, F = %.4f, sun0_zen = %.3f,
    sun0_azi = %.3f, arch = %s, ln = %.2f, cab = %.2f, car = %.2f, 
    cbrown = %.2f, cw = %.3f, cm = %.3f, lamda = %.0f''' \
        % (self.Tol, self.Iter, self.lad_file, self.N, self.Beta,\
        self.refl, self.trans, self.refl_s, \
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

  def J(self, view, l, k, j, i):
    '''The J or Distributed Source Term according to Myneni 1990
    (63). This gives the multiple scattering as opposed to First 
    Collision term Q.
    Input: view - the zenith angle of evaluation, l - index of angle,
    k, j, i - the x, y, z cell index.
    Output: The J term for the view direction.
    '''
    # angles.
    integ = self.grid[k,j,i] * np.multiply(self.Inodes[k,j,i,3],\
        self.Gampi_x[l])
    # element-by-element multiplication
    # numerical integration by gaussian qaudrature
    j = np.sum(integ*self.gauss_wt)
    return j

  def Q1(self, view, l, pnt_lai, mid_lai):
    '''The Q1 First First Collision Source Term as defined in Myneni
    1988d (16) and 1990 (26). This is the downwelling direct part of 
    the Q term. 
    Input: view - the zenith angle of evaluation, l - view index,
    pnt_lai - lad at depth, mid_lai - the cumulative lai starting at 
    TOC.
    Ouput: The Q1 term for view direction.
    '''
    I_exp = self.I_f(self.sun0, mid_lai, self.I0)
    q = pnt_lai * self.Gampi_s[l] * I_exp
    return q

  def Q2(self, view, l, pnt_lai, mid_lai):
    '''The Q2 Second First Collision Source Term as defined in
    Myneni 1988d (17) and 1990 (26). This is the downwelling diffuse
    part of the Q term.
    Input: view - the zenith angle of evaluation, l - view index,
    pnt_lai - lad at depth, lai - the cumulative lai starting at TOC.
    Output: The Q2 term for view direction. 
    '''
    integ = np.multiply(self.Gampi_x[l,self.n/2:],\
       self.I_f(self.sun_down, mid_lai, self.Id))
    q = pnt_lai * np.sum(np.multiply(integ,\
        self.gauss_wt[self.n/2:]))
    return q

  def Q3(self, l, pnt_lai, mid_lai, tot_lai, Ir0): 
    '''The Q3 Third First Collision Source Term as defined in
    Myneni 1988d (18) and 1990 (26). This is the upwelling direct 
    part of the Q term. In (26) Q3 and Q4 are combined.
    Input: l - the view zenith angle index, pnt_lai - lad at depth, 
    mid_lai - cummulative lai starting at the soil, tot_lai - 
    total lai, Ir0 - direct flux or intensity at soil. 
    Output: The Q3 term.
    '''
    dL = tot_lai - mid_lai
    integ = np.multiply(self.Gampi_x[l,:self.n/2],\
    self.I_f(self.sun_up, -dL, Ir0)) 
    q = pnt_lai * np.sum(np.multiply(integ, self.gauss_wt[:self.n/2]))
    return q

  def Q4(self, l, pnt_lai, mid_lai, tot_lai, Ird):
    '''The Q4 Fourth First Collision Source Term as defined in
    Myneni 1988d (19) and 1990 (26). This is the upwelling diffuse 
    part of the Q term. In (26) Q3 and Q4 are combined.  
    Input: l - the view zenith angle index, pnt_lai - lad at depth,
    mid_lai - cummulative lai starting at the soil, tot_lai - total lai,
    Ird - diffuse flux or 
    intensity at soil. 
    Output: The Q4 term.
    '''
    dL = tot_lai - mid_lai
    integ = np.multiply(self.Gampi_x[l,:self.n/2],\
    self.I_f(self.sun_up, -dL, Ird))
    q = pnt_lai * np.sum(np.multiply(integ, self.gauss_wt[:self.n/2]))
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

