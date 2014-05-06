#!/usr/bin/python

'''This is the library for all the reusable functions that are
necessary in our implimentation of the Radiative Transfer of 
radiation through a canopy. Classes are implimented in modules 
such as one_angle.py and two_angle which include the remainder 
of the funstions such as the J and Q terms.
'''

import numpy as np
import scipy as sc
from scipy.integrate import fixed_quad
from scipy.integrate import quad
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
import warnings
import pickle
import leaf_angle_distributions as lad
import pdb

# Gaussian quadratures sets used in integration of one dimensional
# equations. A level-symmetric quadrature set is used in the
# two_angle.py module for the case where integration is required
# over a sphere as opposed to the over one dimension.
# A full gaussian quadrature set was created and saved as a 
# abscissa and weight dictionary file.

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

# H-function LUT was created using the H_LUT.py script, which is
# used instead of the H function. The need for the LUT was due to
# discontinuities in the original H-function by Myneni (1989).
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
  distributions. See the lad module for details. These have
  not been tested.
  ---------------------------------------------------------
  Input: angle - leaf normal angle in radians,
    arch - archetype ie. 'p'-planophile, 'e'-erectophile, 
    's'-spherical/random, 'm'-plagiophile, 'x'-extremophile,
    'u'-uniform, 'k'-kuusk, 'b'-campbell, par - tuple of
    additional parameters for the kuusk and campbell 
    distributions.
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
  '''Used in the G-projection function as the kernel which replaces 
  the azimuth dependence of the double integral based on the 
  Myneni III.13.
  ----------------------------------------------------------------
  Input: angle - the leaf zenith angle in radians,
  view - the view zenith angle in radians.
  Output: The psi kernel.
  '''
  with warnings.catch_warnings(): 
    # the will only work in single thread app
    warnings.simplefilter("ignore")
    temp = 1./np.tan(angle)/np.tan(view) # inf at angle = 0.
    ctns = np.abs(temp) 
    # value used to check for inf below so ignore warning
    phit = np.arccos(-temp)
  psiv = np.where(ctns>1., np.abs(np.cos(view)*np.cos(angle)),\
      np.cos(angle)*np.cos(view)*(2.*phit/np.pi - 1.) + 2./np.pi*\
      np.sqrt(1. - np.cos(angle)**2)*np.sqrt(1. - np.cos(view)**2)\
      *np.sin(phit))
  return psiv

def G(view, arch):
  '''The Geometry factor for a specific view or solar
  direction based on Myneni III.16. The projection of 1 unit 
  area of leaves within a unit volume onto the plane perpen-
  dicular to the view or solar direction.
  ------------------------------------------------------------
  Input: view - the view or solar zenith angle in radians, 
    arch - archetype, see gl function for description of each.
  Output: The integral of the Geometry function (G).
  '''
  g = lambda angle, view, arch: gl(angle, arch)\
      *psi(angle,view) # the G function as defined in Myneni III.16.
  if isinstance(view, np.ndarray):
    if arch == 's': # avoid integration in case of isometric distr.
      G = np.ones_like(view) * 0.5
      return G
    view = np.where(view > np.pi, 2.*np.pi - view, view)
    G = np.zeros_like(view)
    for j,v in enumerate(view):
      G[j] = fixed_quad(g, 0., np.pi/2., args=(v, arch),n=16)[0]
  else:
    if arch == 's': # avoid integration, see above...
      G = 0.5
      return G
    if view > np.pi: # symmetry of distribution about z axis
      view = np.pi - view
    G = fixed_quad(g, 0., np.pi/2., args=(view, arch),n=16)[0] 
    # integrate g function between 0 to pi/2.
  return G

def P0(view, arch, L, N, Disp='pois'):
  '''The Gap Probability or Zero Term based on Myneni III.33-
  III.35. Simply the fraction of unit horisontal area at 
  depth L that is sunlit. The 3 distributions are as follows:
  Regular (Pos. Binomial), Random (Poisson) and Clumped 
  (Neg. Binomial). Untested.......
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
  ----------------------------------------------------------
  Input: view - the view or solar zenith angle, angle -
    leaf normal zenith angle, sun - the solar zenith angle,
    arch - archetype, see gl function for description of each,
    refl - fraction reflected, trans - fraction transmitted.
  Output: Leaf phase function value.
  '''
  return (refl + trans)/4./np.pi

def H(view, angle):
  '''The H function as described in Shultis 1988 (2.25) and Myneni
  V.20 - V.22 and Knyazikhin 2004. This was used to generate the 
  values used in out H_LUT. H_LUT has been amended though to make 
  it more smooth. Current code uses H_LUT and not this function.
  -----------------------------------------------------------------
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
  -----------------------------------------------------------------
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
  '''Used in the Gamma function as the kernel which 
  replaces the azimuth dependence of the double integral 
  based on the Myneni V.20.
  ------------------------------------------------------
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
  '''The one angle Area Scattering Phase Function based on 
  Myneni V.18 and Shultis (17) isotropic scattering assumption. 
  This is the phase function of the scattering in a particular 
  direction based also on the amount of interception in the direction.
  -------------------------------------------------------------
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
  '''The one angle Normalized Scattering Phase Function as 
  described in Myneni VII.A.13.
  -------------------------------------------------------------
  Input: view - view zenith angle, sun - the solar zenith angle, 
    arch - archetype, see gl function for description, 
    refl - fraction reflected, trans - fraction transmitted.
  Output: Normalized Scattering Phase function value.
  '''
  p = 4.*Gamma(view, sun, arch, refl, trans)/(refl+trans)/\
      G(sun, arch)
  return p

def dot(dir1,dir2):
  '''The dot product of 2 spherical sets of angles such
  as (zenith, azimuth).
  The coordinate transformation is:
  x = sin(zen)*cos(azi)
  y = sin(zen)*sin(azi)
  z = cos(zen)
  The result of the dot product is the cosine of the spherical 
  angle between the 2 vectors. Based on:
  http://en.wikibooks.org/wiki/Calculus/Vectors
  and others.
  --------------------------------------------------------
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
  '''Used in the two angle Gamma2 function as the psi kernel value
  which is used in the integration of the leaf angles over the 
  azimuth. It can be either positive or negative. It return a tuple
  with a value for each.
  Based on Myneni (1988c) eq. (12 and 13).
  --------------------------------------------------------
  Input: view - tuple(zenith, azimuth), sun - tuple(zenith, 
  azimuth), leaf_ze - leaf zenith.
  Output: psi kernel tuple(positive, negative)
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
  mu_s = np.array(gauss_mu[str(N)])
  mu_wt = np.array(gauss_wt[str(N)])
  arr_pos = []
  arr_neg = []
  # see notes on 25/02/14. Based on gaussian quad with a
  # change in interval. Note / by 2 not by 2pi.....
  f_mu = lambda mu: np.pi * mu + np.pi # changing interval
  for mu in mu_s:
    arr_pos.append(fun_pos(f_mu(mu), leaf_ze, view, sun))
    arr_neg.append(fun_neg(f_mu(mu), leaf_ze, view, sun))
  psi_pos = np.sum(np.multiply(arr_pos,mu_wt)) / 2.
  psi_neg = np.sum(np.multiply(arr_neg,mu_wt)) / 2.
  '''psi_pos = np.sum(np.multiply(arr_pos,mu_wt)) / 2. 
  psi_neg = np.sum(np.multiply(arr_neg,mu_wt)) / 2.'''
  return (psi_pos, psi_neg)

def Gamma2(view, sun, arch, refl, trans):
  '''The two-angle Area Scattering Phase Function.
  Based on Myneni (1988c) eq. (12).
  --------------------------------------------------
  Input: view - tuple(view zenith, view azimuth), 
    sun - tuple(sun zenith, sun azimuth), arch - archetype, 
    refl, trans.
  Output: two angle Gamma value.
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
  f = fun(mu_l, view, sun, refl, trans, arch)
  g = np.sum(np.multiply(f, mu_w)) * np.pi / 2. #/ np.pi / 2.
  #g = np.sum(np.multiply(f, mu_w)) / np.pi / 2. 
  # the *2/pi seems to make the plots agree with Myneni fig. 11.
  # the above factor is not part of the original text.
  # the nose test integral still is out by pi / 2 * 100....
  return g

def sigma_s2(view, sun, arch, refl, trans, ul):
  '''The differential scattering coefficient or volume scattering
  phase function. Based on Myneni et al (1990) eq 8. This is for 
  the 2 angle case.
  ---------------------------------------------------------
  Input: view - (zenith, azimuth), sun - (zenith, azimuth),
  arch - archetype, refl, trans, ul - leaf area density.
  Output: Volume Scattering Phase function value.
  '''
  s = ul / np.pi * Gamma2(view, sun, arch, refl, trans)
  return s

def sigma(view, arch, ul):
  '''The total interaction cross section or volume extinction
  coefficient. Based on Myneni et al (1990) eq 3.
  ---------------------------------------------------------
  Input: view - the view or solar zenith angle in radians, 
  arch - archetype, ul - leaf area density.
  Output: Total Interaction Cross Section.
  '''
  s = ul * G(view, arch)
  return s

def P2(view, sun, arch, refl, trans):
  '''The two-angle Normalized Scattering Phase Function as 
  described in Myneni 1988(c) eq (22). It uses the Gamma2
  function as the basis of its calculation.
  ---------------------------------------------------------
  Input: view - (zenith, azimuth), sun - (zenith, azimuth),
  arch - archetype, refl, trans.
  Output: Normalized Scattering Phase function value.
  '''
  p = 4.*Gamma2(view, sun, arch, refl, trans)/(refl+trans)/\
      G(sun[0], arch) * 4. / np.pi
  '''p = 4.*Gamma2(view, sun, arch, refl, trans)/(refl+trans)/\
      G(sun[0], arch)'''
      # the *4./pi factor is not part of the original text.
      # it does seem to make the plots agree with Myneni 1988c fig.1. 
  return p

def plotGamma2():
  '''A function that plots the Gamma2 function. For comparison see
  Myneni 1989 fig.11 p 41.
  '''
  n = 16
  refl = 0.5
  trans = 0.5
  albedo = refl + trans
  arch = 'u'
  view_az = np.ones(n)* 0. / 180. * np.pi  #1st half 0 then rest pi
  view_ze = np.arccos(gauss_mu[str(n)])
  view = []
  for a,z in zip(view_az,view_ze):
    view.append((z,a))
  sun_az = 0. * np.pi
  sun_ze = 180. / 180. * np.pi
  sun = (sun_ze, sun_az)
  g = []
  for v in view:
    gam = Gamma2(v, sun, arch, refl, trans) / albedo
    # 
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
  plt.xlabel(r"$\theta$ (Exit zenith angle)")
  plt.ylabel(r"$\Gamma$($\Omega$, $\Omega$0)/$\omega$")
  plt.title(r"$\Gamma$ (Area Scattering Phase Function)") 
  plt.show()

def plotBigPsi2():
  '''A function that plots the big psi two angle kernel.
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
  plt.title(r"$\Psi$' Kernel")
  plt.xlabel(r"$\theta$ (zenith angle)")
  plt.ylabel(r"$\Psi$'")
  plt.show()

def plotP2():
  '''A function to plot the two-angle Normalized Scattering
  Phase Function. For comparison see Myneni 1988c fig.1. 
  Output: plot of P2 function
  '''
  N = 16
  g_wt = np.array(gauss_wt[str(N)])
  mu_l = np.array(gauss_mu[str(N)])
  zen = np.arccos(mu_l)
  a = 180.*np.pi/180. # view azimuth
  view = []
  for z in zen:
    view.append((z,a))
  sun = (110./180.*np.pi,0./180.*np.pi) # sun zenith, azimuth
  arch = 'e'
  refl = 0.07
  trans = 0.03
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

def plotP2_3d():
  '''A function that plots the normalized scattering phase function for
  diffuse scattering. For comparison see Myneni 1991 Fig 3.
  '''
  N = 16
  g_wt = np.array(gauss_wt[str(N)])
  mu_l = np.array(gauss_mu[str(N)])
  azi = np.linspace(0., 2.*np.pi, N)
  zen = np.linspace(0., np.pi, N)
  sun = (110./180.*np.pi,180./180.*np.pi) # sun zenith, azimuth
  arch = 'e'
  refl = 0.07
  trans = 0.03
  p = np.zeros((N,N)) 
  for i, a in enumerate(azi):
    for j, z in enumerate(zen):
      p[j, i] = P2((z, a), sun, arch, refl, trans)
  fig = plt.figure(figsize=plt.figaspect(0.5))
  ax = fig.add_subplot(1, 1, 1, projection='3d')
  a, z = np.meshgrid(azi*180./np.pi, zen*180./np.pi)
  surf = ax.plot_surface(a, z, p, rstride=1, cstride=1, cmap=cm.coolwarm,
      linewidth=0, antialiased=False)
  #ax.set_zlim3d(0.62, 1.94)
  fig.colorbar(surf, shrink=0.5, aspect=10)
  s = '''sun zenith:%.2f, sun azimuth:%.2f, arch:%s,
  refl:%.2f, trans:%.2f''' % \
      (sun[0]*180/np.pi,sun[1]*180/np.pi, arch,\
      refl, trans)
  props = dict(boxstyle='round',facecolor='wheat',alpha=0.5)
  '''plt.text(.5,.5, s, bbox = props, transform=ax.transAxes,\
      horizontalalignment='center', verticalalignment='center')'''
  ax.set_xlabel("Azimuth angle")
  ax.set_ylabel("Zenith angle")
  ax.set_zlabel("P")
  plt.title("P (Normalized Scattering Phase Function)") 
  plt.show()


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
  For comparison see Myneni 1989 fig.11 p41.
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
    gam = Gamma(view=0., sun=angles, refl=refl, trans=trans, arch='s')
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
  disk. The H function plotted is not the final one as it
  has been trandformed into a LUT. See H_LUT above.
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
  plt.show()

def plotBpsi():
  '''A function that plots the Big psi values for a selection
  of view angles. See my notes on 09/01/14 about the plots. 
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
  fig, axarr = plt.subplots(2, len(view), sharex=True, sharey=True)
  for row, rt in enumerate(trans_refl): #refl or transm
    for col, v in enumerate(view): #views
      for i in range(len(x)): #sun
        for j in range(len(y)): #leaf
          Bpsi[i,j] = Big_psi(v, xx[i,j], yy[i,j], rt)
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

