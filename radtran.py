#!/usr/bin/python

'''This will turn out to be the library for all the functions
necessary in our implimentation of the Radiative Transfer of 
radiation through a canopy. Classes may be implimented when 
deemed beneficial. Need to include a specular component as 
described in Knyazikhin 2004 paper. 
This will require non-rotational invariant information.
'''

import numpy as np
import scipy as sc
from scipy.integrate import fixed_quad
from scipy.integrate import quad
import matplotlib.pylab as plt
import warnings
import pickle
import leaf_angle_distributions as lad
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

lut = np.loadtxt('H_LUT.csv', delimiter=',')

def gl(angle, arch, par=None):
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
  #angle = np.abs(angle)
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
  Input: angle - the leaf zenith angle in radians,
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

def G(view, arch):
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
      G[j] = fixed_quad(g, 0., np.pi/2., args=(v, arch),n=16)[0]
  else:
    G = fixed_quad(g, 0., np.pi/2., args=(view, arch),n=16)[0] # integrate leaf angles between 0 to pi/2.
  return G

def K(view, arch):
  '''The Extinction Coefficient for direct beam radiation
  based on Myneni IV.7.
  Input: view - the view or solar zenith angle in radians,
    arch - archetype, see gl function for description of each.
  Output: The Extinction coefficient (K)
  '''
  return -G(view, arch)/np.cos(view)

def P0(view, arch, L, N, Disp='pois'):
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

def f(view, angle, sun, arch, refl, trans):
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
    h = np.where(h<0.,0.,-h)
  else:
    if cotcot > 1.:
      h = mu*mul
    elif cotcot < -1.:
      h = 0.
    else:
      phit = np.arccos(cotcot)
      h = 1./np.pi * (mu*mul*phit + np.sqrt(1.-mu**2.)*\
          np.sqrt(1.-mul**2.)*np.sin(phit))
    if h < 0.:
      h = -h
  return h

def H_LUT(view_sun, angle):
  '''The H function based on a LUT approach based on a corrected
  surface with discontinuities removed and interpolated values
  within hyperbolic boundaries. The H_LUT.py script was used to 
  create the LUT. 
  Input: view - view or sun zenith angle, angle - leaf zenith angle.
  Output: H function value.
  ''' 
  dim = np.shape(lut)[0]-1
  if view_sun==0.:
    x = 2
    #view_sun = 1.0e-10
  else:
    x = np.floor(view_sun/np.pi*dim)
  if isinstance(angle, np.ndarray):
    h = np.zeros_like(angle)
    for i, a in enumerate(angle):
      y = np.floor(a/np.pi*dim)
      h[i] = lut[y,x]
  else:
    y = np.floor(angle/np.pi*dim)
    h = lut[y,x]
  return h

def Big_psi(view,sun,leaf,trans_refl):
  '''The kernel which replaces the azimuth dependence
  of the double integral based on the Myneni V.20.
  Input: view - the view zenith angle in radians,
    sun - the sun/illumination zenith angle,
    leaf - the leaf zenith angle in radians,
    trans_refl - 'r' reflectance or 't' transmittance.    
  Output: The kernel.
  '''
  if view==0.:
    view = 1.0e-10
  if trans_refl == 't':
    B_psi = H_LUT(view,leaf)*H_LUT(sun,leaf) + \
        H_LUT(-view,leaf)*H_LUT(-sun,leaf)
  elif trans_refl == 'r':
    B_psi = H_LUT(view,leaf)*H_LUT(-sun,leaf) + \
        H_LUT(-view,leaf)*H_LUT(sun,leaf)
  else:
    raise Exception('IncorrectRTtype')
  return B_psi

def Gamma(view, sun, arch, refl, trans):
  '''The Area Scattering Phase Function based on Myneni V.18
  and Shultis (17) isotropic scattering assumption. A more 
  elaborate function will be needed see V.18. This is the 
  phase function of the scattering in a particular direction
  based also on the amount of interception in the direction.
  This function can be elaborated using the formula for the 
  true phase angle based on spehrical trigonometry or the dot
  product: 
  np.arccos(np.dot(x,y)/np.sqrt(np.sum(x**2))/np.sqrt(np.sum(y**2)))
  where x and y are the sun/view and leaf angles.
  Input: view - view zenith angle, sun - the solar zenith angle, 
    arch - archetype, see gl function for description, 
    refl - fraction reflected, trans - fraction transmitted.
  Output: Area Scattering Phase function value.
  '''
  '''
  # uncomment/comment the code below for bi-lambetian Gamma.
  B = sun - view # uncomment these lines to run test plot
  gam = (refl + trans)/np.pi/3.*(np.sin(B) - B*np.cos(B)) +\
      trans/np.pi*np.cos(B) # Myneni V.15
  '''
  func = lambda leaf, view, sun, arch, refl, trans: gl(leaf, arch)\
      *(refl*Big_psi(view,sun,leaf,'r') + (trans*Big_psi(view,sun,leaf,'t')))
      # the integral as defined in Myneni V.18.
  if isinstance(sun, np.ndarray):
    # to remove singularity at sun==0.
    sun = np.where(sun==0.,1.0e-10,sun) 
    gam = np.zeros_like(sun)
    for j,s in enumerate(sun):
      gam[j] = fixed_quad(func, 0., np.pi/2.,\
          args=(view,s,arch,refl,trans),n=16)[0]
  else:
    if sun==0.:
      sun = 1.0e-10 # to remove singularity at sun==0.
    gam = fixed_quad(func, 0., np.pi/2.,\
        args=(view,sun,arch,refl,trans),n=16)[0] 
    # integrate leaf angles between 0 to pi/2.
  return gam 

def P(view, sun, arch, refl, trans):
  '''The Normalized Scattering Phase Function as described in 
  Myneni VII.A.13. 
  Input: view - view zenith angle, sun - the solar zenith angle, 
    arch - archetype, see gl function for description, 
    refl - fraction reflected, trans - fraction transmitted.
  Output: Normalized Scattering Phase function value.
  '''
  p = 4.*Gamma(view, sun, arch, refl, trans)/(refl+trans)/\
      G(sun, arch)
  return p

def dot(dir1,dir2):
  '''The dot product of 2 sherical sets of angles such
  as (zenith, azimuth).
  The coordinate transformation is:
  x = sin(zen)*cos(azi)
  y = sin(zen)*sin(azi)
  z = cos(zen)
  Input: dir1 - (zenith,azimuth), dir2 - (zenith,azimuth).
  Output: cos of the 3D angle.
  '''
  zen1, azi1 = (dir1[0], dir1[1])
  zen2, azi2 = (dir2[0], dir2[1])
  vec_v = np.array([np.sin(zen1) * np.cos(azi1), np.sin(zen1) * \
      np.sin(azi1), np.cos(zen1)])
  norm_v = np.linalg.norm(vec_v)
  vec_v = vec_v / norm_v
  vec_s = np.array([np.sin(zen2) * np.cos(azi2), np.sin(zen2) * \
      np.sin(azi2), np.cos(zen2)])
  norm_s = np.linalg.norm(vec_s)
  vec_s = vec_s / norm_s
  cos = np.dot(vec_v,vec_s)
  return cos

def Big_psi2(view, sun, leaf_ze):
  '''A function that returns the psi kernel value for the 
  portion of the integrand that is positive or negative 
  as a tuple.
  Based on Myneni (1988c) eq. (13).
  Input: view - tuple(zenith, azimuth), sun - tuple(zenith, 
  azimuth), leaf_ze - leaf zenith.
  Output: psi kernel (positive, negative)
  '''
  def fun_pos(leaf_az, leaf_ze, view, sun):
    leaf = (leaf_ze,leaf_az)
    integ = dot(leaf,sun) * dot(leaf,view)
    if integ >= 0.:
      return integ
    else:
      return 0.
  def fun_neg(leaf_az, leaf_ze, view, sun):
    leaf = (leaf_ze,leaf_az)
    integ = dot(leaf,sun) * dot(leaf,view)
    if integ <= 0.:
      return -integ
    else:
      return 0.
  N = 32 # no. of leaf azimuth angles
  ph_mu = np.array(gauss_mu[str(N)])
  ph_wt = np.array(gauss_wt[str(N)])
  arr_pos = []
  arr_neg = []
  f_mu = lambda mu: np.pi * mu + np.pi
  for ph in ph_mu:
    arr_pos.append(fun_pos(f_mu(ph), leaf_ze, view, sun))
    arr_neg.append(fun_neg(f_mu(ph), leaf_ze, view, sun))
    #pdb.set_trace()
  psi_pos = np.pi * np.sum(np.multiply(arr_pos,ph_wt))
  psi_neg = np.pi * np.sum(np.multiply(arr_neg,ph_wt))
  return (psi_pos, psi_neg)

def Gamma2(view, sun, arch, refl, trans):
  '''The two-angle Areas Scattering Phase Function.
  Based on Myneni (1988c) eq. (12).
  Input: view - tuple(view zenith, view azimuth), 
  sun - tuple(sun zenith, sun azimuth), refl, trans,\
      arch - archetype.
  Output: 2 angle Gamma value.
  '''
  N = 16 # no. leaf zenith angles
  g_mu = np.array(gauss_mu[str(N)])
  g_wt = np.array(gauss_wt[str(N)])
  def fun(mu_l, view, sun, refl, trans, arch):
    leaf_ze = np.arccos(mu_l)
    arr = []
    for lz in leaf_ze:
      Bp = Big_psi2(view, sun, lz)
      arr.append(gl(lz, arch) * (trans * Bp[0] + refl * Bp[1]))
    return arr
  mu_l = g_mu[:N/2]
  mu_w = g_wt[:N/2]
  g = np.sum(np.multiply(fun(mu_l,view,sun,refl,trans,arch),\
      mu_w))
  #print 'integration of view zenith & azimuth:\
  #    %.2f, %.2f is:%.4f' % (view[0], view[1], g)
  return g

def P2(view, sun, arch, refl, trans):
  '''The two-angle Normalized Scattering Phase Function as 
  described in Myneni 1988(c) eq (22).
  It requires tuples/lists/arrays for the zenith and azimuth
  angle combinations. The gaussian weight and cos(angles) 
  need to be passes as well.
  Input: view - (zenith, azimuth), sun - (zenith, azimuth),
  arch - archetype, refl, trans, g_wt, mu_l.
  Output: Normalized Scattering Phase function value.
  '''
  # a factor for each archetype due to integration error.
  if arch == 'u':
    fact = 1./np.pi
  elif arch == 's':
    fact = 3./4./np.pi
  elif arch == 'p':
    fact = 4./3./np.pi
  elif arch == 'e':
    fact = 2./3./np.pi
  elif arch == 'm':
    fact = 4./5./np.pi
  elif arch == 'x':
    fact = 25./18./np.pi
  else:
    raise Exception('IncorrectArchetype')
  p = 4.*Gamma2(view, sun, arch, refl, trans)/(refl+trans)/\
      G(sun[0], arch) * fact 
  return p

# an older version of P2 below. not good results.
#def P2(view, sun, arch, refl, trans, g_wt, mu_l):
  '''The two-angle Normalized Scattering Phase Function as 
  described in Myneni 1988(c) eq (22).
  It requires tuples/lists/arrays for the zenith and azimuth
  angle combinations. The gaussian weight and cos(angles) 
  need to be passes as well.
  Input: view - (zenith, azimuth), sun - (zenith, azimuth),
  arch - archetype, refl, trans, g_wt, mu_l.
  Output: Normalized Scattering Phase function value.
  '''
  '''mu_v, mu_s = np.cos((view[0],sun[0]))
  ph_v, ph_s = (view[1],sun[1])
  ph_l = np.pi*mu_l + np.pi
  mid = len(mu_l)/2
  zl = np.arccos(mu_l[:mid])
  #pdb.set_trace()
  DP1 = np.abs(mu_s*mu_l + np.sqrt(1.-mu_s**2) * \
      np.sqrt(1.-mu_l**2) * np.cos(ph_s - ph_l))
  DP2 = np.abs(mu_v*mu_l + np.sqrt(1.-mu_v**2) * \
      np.sqrt(1.-mu_l**2) * np.cos(ph_v - ph_l)),
  integ1 = np.sum(np.multiply(g_wt[:mid],gl(zl,arch)))
  gl_h = 1.*g_wt # assuming that needs to be weighted.
  DPs = np.multiply(DP1,DP2)
  integ2 = np.sum(np.multiply(gl_h, DPs))
  p = 2./G(sun[0],arch) * integ1 * integ2
  return p'''

def plotGamma2():
  '''A function that plots the Gamma2 function.
  '''
  n = 16
  refl = 0.5
  trans = 0.5
  arch = 'u'
  view_az = np.ones(n)*0. #1st half 0 then rest pi
  view_ze = np.arccos(gauss_mu[str(n)])
  view = []
  for a,z in zip(view_az,view_ze):
    view.append((z,a))
  sun_az = 0.
  sun_ze = np.pi
  sun = (sun_ze, sun_az)
  g = []
  for v in view:
    gam = Gamma2(v, sun, arch, refl, trans)
    g.append(gam)
  fig, ax = plt.subplots(1)
  plt.plot(view_ze*180/np.pi,g,'r')
  s = '''sun zenith:%.2f, sun azimuth:%.2f,
  view azimuth:%.2f, refl:%.2f,
  trans:%.2f, arch:%s''' % \
      (sun_ze*180/np.pi,sun_az*180/np.pi,\
      view_az[0]*180/np.pi, refl, trans, arch)
  props = dict(boxstyle='round',facecolor='wheat',alpha=0.5)
  plt.text(.5,.5, s, bbox = props, transform=ax.transAxes,\
      horizontalalignment='center', verticalalignment='center')
  plt.show()


def plotBigPsi2():
  '''A function that plots the big psi kernel.
  '''
  n = 16
  view_az = np.ones(n)*0. #1st half 0 then rest pi
  view_ze = np.linspace(0.,np.pi,n)
  view = []
  for a,z in zip(view_az,view_ze):
    view.append((z,a))
  sun_az = 0.
  sun_ze = np.pi
  sun = (sun_ze, sun_az)
  leaf_ze = 0.#np.pi/4.
  y_pos = []
  y_neg = []
  for v in view:
    pos, neg = Big_psi2(v, sun, leaf_ze)
    y_pos.append(pos)
    y_neg.append(neg)
  fig, ax = plt.subplots(1)
  plt.plot(view_ze*180/np.pi,y_pos,'r',label='trans.')
  plt.plot(view_ze*180/np.pi,y_neg,'b',label='refl.')
  plt.legend(loc=7)
  s = '''sun zenith:%.2f, sun azimuth:%.2f,
  view azimuth:%.2f, leaf zenith:%.2f''' % \
      (sun_ze*180/np.pi,sun_az*180/np.pi,\
      view_az[0]*180/np.pi,leaf_ze*180/np.pi)
  props = dict(boxstyle='round',facecolor='wheat',alpha=0.5)
  plt.text(.5,.5, s, bbox = props, transform=ax.transAxes,\
      horizontalalignment='center', verticalalignment='center')
  plt.show()

def plotP2():
  '''A function to plot the two-angle Normalized Scattering
  Phase Function. 
  Output: plot of P2 function
  '''
  N = 16
  g_wt = np.array(gauss_wt[str(N)])
  mu_l = np.array(gauss_mu[str(N)])
  zen = np.arccos(mu_l)
  a = 42.7*np.pi/180. # view azimuth
  view = []
  for z in zen:
    view.append((z,a))
  sun = (180./180.*np.pi,0./180.*np.pi) # sun zenith, azimuth
  arch = 'u'
  refl = 0.5
  trans = 0.5
  y = []
  for v in view:
    y.append(P2(v, sun, arch, refl, trans))
  fig, ax = plt.subplots(1)
  plt.plot(mu_l,y, 'r--')
  s = '''sun zenith:%.2f, sun azimuth:%.2f, arch:%s
  view azimuth:%.2f, refl:%.2f, trans:%.2f''' % \
      (sun[0]*180/np.pi,sun[1]*180/np.pi, arch,\
      a*180/np.pi, refl, trans)
  props = dict(boxstyle='round',facecolor='wheat',alpha=0.5)
  plt.text(.5,.5, s, bbox = props, transform=ax.transAxes,\
      horizontalalignment='center', verticalalignment='center')
  plt.xlabel(r"$\mu$ (cosine of exit zenith)")
  plt.ylabel(r"P($\Omega$, $\Omega$0)")
  plt.title("P (Normalized Scattering Phase Function)") 
  plt.show()

def plotgl():
  '''A function to plot the LAD distribution for each 
  archetype.
  Output: plots of gl functions
  '''
  types = ['p','e','s','m','x','ug','k','b']
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
  tr_rf= np.arange(0.,.6,0.1) # transmittance factor for albedo = 1.
  angles = np.linspace(np.pi, 0., 100)
  xcos = np.linspace(-1., 1., 100)
  maxy = 0.
  miny = 0.
  for trans in tr_rf:
    refl = 1. - trans
    gam = P(view=0., sun=angles, refl=refl, trans=trans, arch='u')
    plt.plot(angles, gam, label=str(trans))
    maxy = max(maxy,gam.max())
    #pdb.set_trace()
  plt.axis([angles.max(),angles.min(),0.,maxy+0.02])
  plt.title('Areas Scattering Phase Function (Gamma)')
  plt.xlabel('Phase Angle (radians)')
  plt.ylabel(r'$\Gamma$/albedo')
  plt.legend(title='trans./albedo')
  plt.show()

def plotH():
  '''A function that plots the H values for use in the 
  Area Scattering Phase Function, and saves the H values to 
  disk.
  '''
  x = np.linspace(0., np.pi, 500) # view or sun zenith angles
  y = x.copy() # leaf normal zenith angles
  xx, yy = np.meshgrid(x,y)
  h = np.zeros_like(xx)
  for i in range(len(x)):
    for j in range(len(y)):
      h[i,j] = H(xx[i,j], yy[i,j])
  #pdb.set_trace()
  np.savetxt('view_sun.csv', xx, delimiter=',')
  np.savetxt('leaf.csv', yy, delimiter=',')
  np.savetxt('H_values.csv', h, delimiter=',')
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
  for row, rt in enumerate(trans_refl): #refl or transm
    for col, v in enumerate(view): #views
      for i in range(len(x)): #sun
        for j in range(len(y)): #leaf
          Bpsi[i,j] = Big_psi(v, xx[i,j], yy[i,j], rt)
          #print xx[i,j], yy[i,j]
          #pdb.set_trace()
      ax = axarr[row][col]
      im = ax.pcolormesh(xx, yy, Bpsi, vmin=0., vmax=1.)
      if row == 0:
        ax.set_title('Kernel view zenith\nangle @ %.1f degrees'\
            % (v*180./np.pi))
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
#plotBpsi()
#plotP2()
