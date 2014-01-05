#!/usr/bin/python

# This will turn out to be the library for all the functions necessary in our implimentation of the Radiative Transfer of radiation through a canopy. Classes may be implimented when deemed beneficial.

# Here are the basic functions used to describe the canopy structure.
import numpy as np
import scipy as sc
from scipy.integrate import quad
import matplotlib.pylab as plt
import warnings
import leaf_angle_distributions as lad
import pdb

def gl(angle, arch='s', par=None):
  '''The leaf normal angle distribution in radians.
  The distributions are based on Myneni III.21 - .23 for 
  planophile, erectophile and plagiophile which itself is
  based on de Wit 1965 and Bunnik 1978. 
  The rest are according to Bunnik p.35 and Ross 1981 p.117. 
  It seems like the the formulas in Liang 2005 p.78 are
  incorrect. They differ by the reciprocal with those in 
  Bunnik 1978 and others.
  Added additional distributions provided by J. Gomez-
  Dans. These are: Kuusk (1995) and Campbell (1990)
  distributions. See the lad module for details.
  Input: angle - leaf normal angle in radians,
    arch - archetype ie. 'p'-planophile, 'e'-erectophile, 
    's'-spherical/random, 'm'-plagiophile, 'x'-extremophile,
    'u'-uniform, 'k'-kuusk, 'b'-campbell, par - tuple of
    parameters for the kuusk and campbell distributions.
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
  elif arch=='k': # kuusk. had to multiply by factor below to work.
    if type(par)==tuple:
      gl = lad.kuusk_lad(angle,par[0],par[1])/4.**2*np.pi**2
    else:
      gl = lad.kuusk_lad(angle)/4.**2*np.pi**2
  elif arch=='b': # campbell
    if type(par)==tuple:
      gl = lad.elliptical_lad(angle,par[0],par[1])
    else:
      gl = lad.elliptical_lad(angle)
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
  direction based on Myneni III.16. The projection of 1 unit 
  area of leaves within a unit volume onto the plane perpen-
  dicular to the view or solar direction.
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
      G[j] = quad(g, 0., np.pi/2., args=(v, arch))[0]
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

def H(view, angle):
  '''The H function as described in Shultis 1988 (2.25) and Myneni
  V.20 - V.22 and Knyazikhin 2004. 
  Input: view - view or sun zenith angle, angle - leaf zenith angle.
  Output: H function value.
  '''
  mu = np.cos(view)
  mul = np.cos(angle)
  with warnings.catch_warnings(): # the will only work in single thread app
    warnings.simplefilter("ignore")
    cotcot = 1./(np.tan(view)*np.tan(angle))
  if isinstance(mul, np.ndarray):
    h = np.zeros_like(mul)
    for i, m in enumerate(mul):
      if cotcot[i] > 1.:
        h[i] = mu*m
      elif cotcot[i] < -1.:
        h[i] = 0.
      else:
        phit = np.arccos(cotcot[i])
        h[i] = 1./np.pi * (mu*m*phit + np.sqrt(1.-mu**2.)*\
            np.sqrt(1.-m**2.)*np.sin(phit))
  else:
    if cotcot > 1.:
      h = mu*mul
    elif cotcot < -1.:
      h = 0.
    else:
      phit = np.arccos(cotcot)
      h = 1./np.pi * (mu*mul*phit + np.sqrt(1.-mu**2.)*\
          np.sqrt(1.-mul**2.)*np.sin(phit))
  return h

def Big_psi(view,sun,leaf,trans_refl):
  '''The kernel which replaces the azimuth dependence
  of the double integral based on the Myneni V.20.
  Input: view - the view zenith angle in radians,
    sun - the sun/illumination zenith angle,
    leaf - the leaf zenith angel in radians,
    trans_refl - 'r' reflectance or 't' transmittance.    
  Output: The kernel.
  '''
  if trans_refl == 't':
    B_psi = H(view,leaf)*H(sun,leaf) + H(-view,leaf)*H(-sun,leaf)
  elif trans_refl == 'r':
    B_psi = H(view,leaf)*H(-sun,leaf) + H(-view,leaf)*H(sun,leaf)
  else:
    raise Exception('IncorrectRTtype')
  return B_psi

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
  B = view - sun
  gam = 4.*np.pi/(refl + trans)*f(view, refl=refl, trans=trans)\
      /3./np.pi*(np.sin(B) - B*np.cos(B)) + trans/3.*np.cos(B) # Myneni V.15
   
  return gam 

#def P(

def plotgl():
  '''A function to plot the LAD distribution for each 
  archetype.
  Output: plots of gl functions
  '''
  types = ['p','e','s','m','x','u','k','b']
  colors = ['g','b','+r','xy','--c','p','k','y']
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
  types = ['p','e','s','m','x','u','k','b']
  colors = ['g','b','+r','xy','--c','p','k','y']
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

def plotGamma():
  '''A function that plots the Area Scattering Phase 
  function various values of leaf scattering albedo.
  Uncomment the first part of Gamma function.
  Output: plots of the Gamma function
  '''
  tr_rf= np.arange(0.,.6,0.1) # single scattering albedos
  views = np.linspace(np.pi, 0., 100)
  xcos = np.linspace(-1., 1., 100)
  for trans in tr_rf:
    refl = 1. - trans
    gam = Gamma(views, refl=refl, trans=trans)
    plt.plot(xcos, gam, label=str(trans))
    #pdb.set_trace()
  plt.title('Areas Scattering Phase Function (Gamma)')
  plt.xlabel('Cos(Phase Angle)')
  plt.ylabel(r'$\Gamma$/albedo')
  plt.legend(title='trans./albedo')
  plt.show()

def plotH():
  '''A function that plots the H values for use in the 
  Area Scattering Phase Function.
  '''
  x = np.linspace(0., np.pi, 500) # view or sun zenith angles
  y = x.copy() # leaf normal zenith angles
  xx, yy = np.meshgrid(x,y)
  h = np.zeros_like(xx)
  for i in range(len(x)):
    for j in range(len(y)):
      h[i,j] = H(xx[i,j], yy[i,j])
  #pdb.set_trace()
  plt.pcolormesh(xx, yy, h)
  plt.axis([xx.max(),xx.min(),yy.max(),yy.min()])
  plt.title('H Function Values')
  plt.xlabel('Zenith angle view/sun (radians)')
  plt.ylabel('Zenith angle leaf normal (radians)')
  plt.colorbar()
  #print np.where(h==np.inf)#np.diagonal(np.fliplr(h))[20:35]
  plt.show()

def plotBpsi():
  '''A function that plots the Big psi values for a selection
  of view angles.
  Noticed that negative values for reflectance are correct and
  positive values for transmittance also. If you follow each case
  note that this does actually tell you alot about the influence
  of angles on the reflectance and transmittance of the radiation.
  '''
  x = np.linspace(0., np.pi, 100) # sun zenith angles
  y = x.copy() # leaf zenith angles
  xx, yy = np.meshgrid(x,y)
  view = np.linspace(0.,np.pi,5) # view zenith angles
  trans_refl = ('r','t')
  Bpsi = np.zeros_like(xx)
  #pdb.set_trace()
  fig, axarr = plt.subplots(2, len(view), sharex=True, sharey=True)
  for row, rt in enumerate(trans_refl):
    for col, v in enumerate(view):
      if row != 0:
        add = len(view)
      else:
        add = 0
      #plt.subplot(2, len(view), col+1 + add)
      for i in range(len(x)):
        for j in range(len(y)):
          Bpsi[i,j] = Big_psi(v, xx[i,j], yy[i,j], rt)
      ax = axarr[row][col]
      im = ax.pcolormesh(xx, yy, Bpsi, vmin=-0.4, vmax=1.)
      #ax.colorbar()
      if row == 0:
        ax.set_title('Kernel view zenith\nangle @ %.1f degrees'\
            % (v*180./np.pi))
      #pdb.set_trace()
      ax.axis([xx.max(),xx.min(),yy.max(),yy.min()])
      if row == 1:
        ax.set_xlabel('Zenith angle illum. (radians)')
      if col == 0:
        if row == 0:
          ax.set_ylabel('Zenith angle leaf (radians)\n'+
              'Reflectance or back to source')
        else:
          ax.set_ylabel('Zenith angle leaf (radians)\n'+
              'Transmittance or away from source')
  cbar_ax = fig.add_axes([0.9, 0.15, 0.022, 0.7])
  fig.subplots_adjust(wspace=0., hspace=0.)
  fig.colorbar(im, cax=cbar_ax)
  plt.show()

#pdb.set_trace()
#plotgl()
#plotG()
#plotGamma()
#plotH()
plotBpsi()
