#!/usr/bin/python

# This will turn out to be the library for all the functions necessary in our implimentation of the Radiative Transfer of radiation through a canopy. Classes may be implimented when deemed beneficial.

# Here are the basic functions used to describe the canopy structure.
import numpy as np
import scipy as sc
from scipy.integrate import quad
import matplotlib.pylab as plt
import warnings
import pdb

def gl(angle, arch='s'):
  '''The leaf normal angle distribution in radians.
  The distributions are based on Myneni III.21 - .23 for 
  planophile, erectophile and plagiophile which itself is
  based on de Wit 1965 and Bunnik 1978. 
  The rest are according to Bunnik p.35 and Ross 1981 p.117. 
  It seems like the the formulas in Liang 2005 p.78 are
  incorrect. They differ by the reciprocal with those in 
  Bunnik 1978 and others.
  Input: angle - leaf normal angle in radians,
    arch - archetype ie. 'p'-planophile, 'e'-erectophile, 
    's'-spherical/random, 'm'-plagiophile, 'x'-extremophile,
    'u'-uniform.
  Output: g value at angle
  '''
  if arch=='p': # planophile
    gl = 2./np.pi*(1. + np.cos(2.*angle))
  elif arch=='e': # erectophile
    gl = 2./np.pi*(1. - np.cos(2.*angle))
  elif arch=='s': # spherical
    gl = np.sin(angle)
  elif arch=='m': # plagiophile
    gl = 2./np.pi*(1. - np.cos(4.*angle))
  elif arch=='x': # extremophile
    gl = 2./np.pi*(1. + np.cos(4.*angle))
  elif arch=='u': # uniform
    if isinstance(angle, np.ndarray):
      gl =  np.ones(np.shape(angle))*2./np.pi
    else:
      gl = 2./np.pi
  else:
    raise Exception('IncorrectArchetype')
  return gl

def psi(angle, view):
  '''The kernel which replaces the azimuth dependence
  of the double integral based on the Myneni III.13.
  Input: angle - the leaf zenith angel in radians,
  view - the view zenith angle in radians.
  Output: The kernel.
  '''
  with warnings.catch_warnings(): # the will only work in single thread app
    warnings.simplefilter("ignore")
    temp = 1./np.tan(angle)/np.tan(view) # inf at angle = 0.
    ctns = np.abs(temp) # value used to check for inf below so ignore warning
    phit = np.arccos(-temp)
  psiv = np.where(ctns>1., np.abs(np.cos(view)*np.cos(angle)),\
      np.cos(angle)*np.cos(view)*(2.*phit/np.pi - 1.) + 2./np.pi*\
      np.sqrt(1. - np.cos(angle)**2)*np.sqrt(1. - np.cos(view)**2)*np.sin(phit))
  return psiv

def G(view, arch='s'):
  '''The Geometry factor for a specific view or solar
  direction based on Myneni III.16.
  Input: view - the view or solar zenith angle in radians, 
    arch - archetype, see gl function for description of each.
  Output: The integral of the Geometry function (G).
  '''
  #pdb.set_trace()
  g = lambda angle, view, arch: gl(angle, arch)\
      *psi(angle,view) # the G function as defined in Myneni III.16.
  if isinstance(view, np.ndarray):
    G = np.zeros_like(view)
    for j,v in enumerate(view):
      G[j] = quad(g, 0., np.pi/2, args=(v, arch))[0]
  else:
    G = quad(g, 0., np.pi/2., args=(view, arch))[0] # integrate leaf angles between 0 to pi/2.
  return G

def K(view, arch='s'):
  '''The Extinction Coefficient for direct beam radiation
  based on Myneni IV.7.
  Input: view - the view or solar zenith angle in radians,
    arch - archetype, see gl function for description of each.
  Output: The Extinction coefficient (K)
  '''
  return -G(view, arch)/np.cos(view)

def P0(view, arch='s', L=5., N=10., Disp='pois'):
  '''The Gap Probability or Zero Term based on Myneni III.33-
  III.35. Simply the fraction of unit horisontal area at 
  depth L that is sunlit. The 3 distributions are as follows:
  Regular (Pos. Binomial), Random (Poisson) and Clumped 
  (Neg. Binomial).
  Input: view - the view or solar zenith angle in radians,
    arch - archetype, see gl function for description of each,
    L - total LAI or depth, N - Number of layers, Disp - 
    Distribution see above ('pb', 'pois', 'nb').
  Output: The gap probability (P0)
  '''
  if Disp == 'pois':
    if isinstance(L, np.ndarray):
      p = np.exp(np.outer(K(view, arch), L))
    else:
      p = np.exp(K(view, arch)*L)
  elif Disp == 'pb':
    if isinstance(L, np.ndarray):
      p = (1. + np.outer(K(view, arch), L/N))**N
    else:
      p = (1. + K(view, arch)*L/N)**N
  elif Disp == 'nb':
    if isinstance(L, np.ndarray):
      p = (1. - np.outer(K(view, arch), L/N))**-N
    else:
      p = (1. - K(view, arch)*L/N)**-N
  else:
    raise Exception('IncorrectDistrType')
  return p

def f(view, angle=0., sun=0., arch='s', refl=0.2, trans=0.1):
  '''The Leaf Scattering Transfer Function based on
  Myneni V.9. and Shultis (16) isotropic leaf
  scattering assumption. This is leaf single-scattering
  albedo in a particular direction per steridian.
  This assumes a bi-lambertian scattering
  model. Modifications to this model will be made using
  a leaf reflectance model such a PROSPECT. At the moment 
  this function is a placeholder for a more elaborate
  model.
  Input: view - the view or solar zenith angle, angle -
    leaf normal zenith angle, sun - the solar zenith angle,
    arch - archetype, see gl function for description of each,
    refl - fraction reflected, trans - fraction transmitted.
  Output: Leaf phase function value.
  '''
  return (refl + trans)/4./np.pi

def Gamma(view, angle=0., sun=0., arch='s', refl=0.2, trans=0.1):
  '''The Area Scattering Phase Function based on Myneni V.18
  and Shultis (17) isotropic scattering assumption. A more 
  elaborate function will be needed see V.18. This is the 
  phase function of the scattering in a particular direction
  based also on the amount of interception in the direction.
  Input: view - view or solar zenith angle, angle - leaf normal
    zenith angle, sun - the solar zenith angle, arch - archetype, 
    see gl function for description, refl - fraction reflected,
    trans - fraction transmitted.
  Output: Area Scattering Phase function value.
  '''
  return G(view, arch)*f(view)*np.pi

#def P(

def plotgl():
  '''A function to plot the LAD distribution for each 
  archetype.
  Output: plots of gl functions
  '''
  types = ['p','e','s','m','x','u']
  colors = ['g','b','+r','xy','--c','p']
  views = np.linspace(0., np.pi/2, 100)
  gf = np.zeros_like(views)
  for i,c in zip(types,colors):
    gf = gl(views, i)
    #pdb.set_trace()
    plt.plot(views*180./np.pi, gf, c, label=i)
  plt.title('Leaf Angle Distribution')
  plt.xlabel('Zenith Angle')
  plt.ylabel('gl')
  plt.legend()
  plt.show()

def plotG():
  '''A function to plot the integrated G functions for 
  every LAD and every view angle. 
  Output: plots of the G functions.
  '''
  types = ['p','e','s','m','x','u']
  colors = ['g','b','+r','xy','--c','p']
  views = np.linspace(0., np.pi/2., 100)
  for i,c in zip(types,colors):
    Gf = G(views, i)
    plt.plot(views*180./np.pi, Gf, c, label=i)
    #pdb.set_trace()
  plt.title('Leaf projection function (G)')
  plt.xlabel('Zenith Angle')
  plt.ylabel('G')
  plt.legend()
  plt.show()

#pdb.set_trace()
#plotgl()
#plotG()

