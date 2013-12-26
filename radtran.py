#!/usr/bin/python

# This will turn out to be the library for all the functions necessary in our implimentation of the Radiative Transfer of radiation through a canopy. Classes may be implimented when deemed beneficial.

# Here are the basic functions used to describe the canopy structure.
import numpy as np
import scipy as sc
from scipy.integrate import quadrature
import matplotlib.pylab as plt
import pdb

def gl(angle, arch='s'):
  '''The leaf normal angle distribution in radians. 
  Input: angle - leaf normal angle in radians,
    arch - archetype ie. 'p'-planophile, 'e'-erectophile, 
    's'-spherical, 'm'-plagiophile, 'x'-extremophile
  Output: g value at angle
  '''
  if arch=='p': # planophile
    gl = 3.*(np.cos(angle))**2
  elif arch=='e': # erectophile
    gl = 3./2*(np.sin(angle))**2
  elif arch=='s': # spherical
    if isinstance(angle, np.ndarray):
      gl = np.ones(np.shape(angle))
    else:
      gl = 1.
  elif arch=='m': # plagiophile
    gl = 15./8*(np.sin(2*angle))**2 
  elif arch=='x': # extremophile
    gl = 15./7*(np.cos(2*angle))**2
  else:
    raise Exception('IncorrectArchetype')
  return gl

def psi(angle, view):
  '''The kernel basel on the Myneni III.13.
  Input: angle - the leaf zenith angel in radians,
  view - the view zenith angle in radians.
  Output: The kernel.
  '''
  temp = 1./np.tan(angle)/np.tan(view)
  ctns = np.abs(temp)
  phit = 1./np.cos(-temp)
  #pdb.set_trace()
  psiv = np.where(ctns>1., np.abs(np.cos(view)*np.cos(angle)),\
      np.cos(angle)*np.cos(view)*(2.*phit/np.pi - 1.) + 2./np.pi*\
      np.sqrt(1. - np.cos(angle)**2)*np.sqrt(1. - np.cos(view)**2))
  return psiv

def G(view, arch='s'):
  '''The Geometry factor for a specific direction.
  Input: view - the view zenith angle in radians, 
    arch - archetype, see gl function for description of each.
  Output: The integral of the Geometry function.
  '''
  g = lambda angle, view, arch: 1./(2.*np.pi) * gl(angle, arch)\
      *psi(angle,view)#np.abs(np.cos(angle-view)) # the G function as defined in Liang p.78.
  G = quadrature(g, 0., np.pi/2., args=(view, arch)) # integrate leaf angles between 0 to pi/2.
  return G

def plotgl():
  '''A function to plot the LAD distribution for each 
  archetype.
  Output: plots of gl functions
  '''
  types = ['p','e','s','m','x']
  colors = ['g','b','+r','xy','--c']
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
  types = ['p','e','s','m','x']
  colors = ['g','b','+r','xy','--c']
  views = np.linspace(0., np.pi/2, 100)
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

