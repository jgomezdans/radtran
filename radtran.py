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
    's'-spherical/random, 'm'-plagiophile, 'x'-extremophile
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
    temp = 1./np.tan(angle)/np.tan(view)
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
  Output: The integral of the Geometry function.
  '''
  g = lambda angle, view, arch: gl(angle, arch)\
      *psi(angle,view) # the G function as defined in Myneni III.16.
  G = quad(g, 0., np.pi/2., args=(view, arch)) # integrate leaf angles between 0 to pi/2.
  return G

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
    '''for j,v in enumerate(views):
      Gf[j] = G(v, i)[0]'''
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
  Gf = np.zeros_like(views)
  for i,c in zip(types,colors):
    for j,v in enumerate(views):
      Gf[j] = G(v, i)[0]
    plt.plot(views*180./np.pi, Gf, c, label=i)
    #pdb.set_trace()
  plt.title('Leaf projection function (G)')
  plt.xlabel('Zenith Angle')
  plt.ylabel('G')
  plt.legend()
  plt.show()

#pdb.set_trace()
plotgl()
plotG()

