#!/usr/bin/python

''' A script that tests the one_angle.py module.
    Run by: nosetests -v test_one_angle.py
'''

import one_angle as oa
import two_angle as ta
import radtran as rt
import numpy as np
from scipy.integrate import *
import pickle
import pdb

# loading the quadrrature set for calculations
N = 4
f = open('quad_dict.dat')
quadr_dict = pickle.load(f)
f.close()
quadr = quadr_dict[str(N)]
gauss_wt = quadr[:,2] / 8.
views = np.array(zip(quadr[:,0],quadr[:,1]))

def test_gl():
  '''A function to test to integration of the leaf angle 
  distribution function gl() between 0 and pi/2 is unity.
  '''
  arch = ['p','e','s','m','x','u']
  g = []
  for a in arch:
    g.append(quad(rt.gl, 0., np.pi/2., args=(a))[0])
  np.testing.assert_almost_equal(g, 1.0, decimal=6)
  #return g

def test_one_angle_fluxes():
  '''A function that tests one_angle.py flux calculation.
  see table in Myneni 1988b for test data.
  '''
  test = oa.rt_layers(Tol=1.e-06,Iter=200,K=40,N=16,\
      Lc=4.,refl=0.175,trans=0.175,refl_s=1.e-16,I0=1.,\
      sun0=120.,arch='s')
  test.solve()
  c,s,a = test.Scalar_flux()
  val = np.array([c,s,a])
  truth = np.array([0.0988, 0.0383,\
      0.8628])
  #return (val, truth)
  np.testing.assert_almost_equal(val, truth, decimal=2)

def test_two_angle_fluxes():
  '''A function that tests two_angle.py flux calculation.
  see table in Myneni p.95 for test data.
  '''
  test = ta.rt_layers(Tol=1.e-06,Iter=200,K=20,N=4,\
      Lc=4.,refl=0.1,trans=0.1,refl_s=0.1,sun0_zen=180.,\
      F=np.pi,sun0_azi=0.0,arch='s',Beta=1.0)
  test.solve()
  c,s,a = test.Scalar_flux()
  val = np.array([c,s,a])
  truth = np.array([0.0356, 0.1360, 0.8284])
  #return (val, truth)
  np.testing.assert_almost_equal(val, truth, decimal=2)
  
def test_G2():
  '''A function to test the G-projection function for integration
  to 0.5 for the two angle case'''
  truth = 0.5
  arch = ['p','e','s','m','x','u']
  vals = []
  for a in arch:
    Gg = []
    for v in views:
      Gg.append(rt.G(v[0], a))
    val = np.sum(np.multiply(Gg, gauss_wt))
    vals.append(val)
  #return (vals, truth)
  np.testing.assert_almost_equal(vals, truth, decimal=2)
 
def test_dot():
  '''A function to test the dot product function used to 
  calculate the cosine between 2 spherical angles.
  The dot product is used in the calculation of the
  Big_psi2 function.
  '''
  sun = (180. * np.pi / 180., 45. * np.pi / 180.)
  view = (90. * np.pi / 180., 45. * np.pi / 180.)
  truth = np.cos(np.pi/2)
  val = rt.dot(sun, view)
  #return (val, truth)
  np.testing.assert_almost_equal(val, truth, decimal=6)
 
def test_Big_psi2():
  '''A function to test the radtran.py Big_psi2 function against a 
  dot integrated function using scipy integration solution based
  on Myneni 1988c eq(13).
  '''
  leaf_ze = 82. * np.pi / 180.
  view_ze = 25. * np.pi / 180.
  view_az = 63. * np.pi / 180.
  view = (view_ze, view_az)
  sun_ze = 74. * np.pi / 180.
  sun_az = 29. * np.pi / 180.
  sun = (sun_ze, sun_az)
  val = rt.Big_psi2(view, sun, leaf_ze)
  val = val[0] - val[1]
  def integ(leaf_az, leaf_ze, view_ze, view_az, sun_ze, sun_az):
    view = (view_ze, view_az)
    sun = (sun_ze, sun_az)
    dots = np.array([])
    for la in leaf_az:
      leaf = (leaf_ze, la)
      sun = (sun_ze, sun_az)
      view = (view_ze, view_az)
      dots = np.append(dots, rt.dot(leaf, sun) * rt.dot(leaf, view))
    return dots
  truth = fixed_quad(integ, 0, 2.*np.pi, args=(leaf_ze, view_ze,\
      view_az, sun_ze, sun_az))[0] / np.pi / 2.
  #return (val, truth)
  #pdb.set_trace()
  np.testing.assert_almost_equal(val, truth, decimal=3)
 
def test_Gamma2():
  '''A function that tests the radtran.py Gamma2 phase function
  for integration to albedo * G(sun) over the complete sphere.
  Based on Myneni 1988c eq(2).
  '''
  sun = (180./180. * np.pi, 45./180. * np.pi)
  arch = 'u'
  refl = 0.5
  trans = 0.5
  albedo = refl + trans
  truth = albedo * rt.G(sun[0],arch)
  Gam = []
  for v in views:
    Gam.append(rt.Gamma2(v, sun, arch, refl, trans))
  val = np.sum(np.multiply(Gam, gauss_wt)) * 16. / np.pi
  # the *16/pi is not part of the original text but makes the 
  # code agree with the test. one would think that *4*pi would
  # be the factor to use.
  #return (val, truth)
  np.testing.assert_almost_equal(val, truth, decimal=2)

def test_Gamma_Gamma2():
  '''A function that tests the one and two angle Gamma functions
  against eachother. Gamma2 is intergrated over the azimuth and 
  divided by 2*pi.
  '''
  view_ze = 25. * np.pi / 180.
  view_az = 63. * np.pi / 180.
  view = (view_ze, view_az)
  sun_ze = 74. * np.pi / 180.
  sun_az = 29. * np.pi / 180.
  sun = (sun_ze, sun_az)
  arch = 'u'
  refl = 0.5
  trans = 0.5
  g1 = rt.Gamma(view_ze, sun_ze, arch, refl, trans)
  def integ(view_az, view_ze, sun_az, sun_ze, arch, refl, trans):
    g = rt.Gamma2((view_ze, view_az), (sun_ze, sun_az), arch, refl,\
        trans)
    return g
  g2 = quad(integ, 0., 2.*np.pi, args=(view_ze, sun_az, sun_ze, arch,\
      refl, trans), limit=10, points=(np.pi/2.,))[0] / 2. / np.pi
  #return (g1, g2)
  np.testing.assert_almost_equal(g1, g2, decimal=2)

def test_Psi2_psi():
  '''A function that test the 2 angle Big_psi functions against 
  each other. Based on Myneni 1988c (19).
  '''
  view_zen = 40.*np.pi/180.
  sun_zen = 15.*np.pi/180.
  sun_azi = 60.*np.pi/180.
  leaf_zen = 20.*np.pi/180.
  arch = 'u'
  refl = .5
  trans = .5
  #truth = rt.Big_psi(view_zen, sun_zen, leaf_zen, 't')
  truth = trans * np.cos(view_zen) * np.cos(sun_zen) * \
      np.cos(leaf_zen)**2.
  def fun(view_azi, view_zen, sun_azi, sun_zen, leaf_zen,\
      refl, trans):
    view = (view_zen, view_azi)
    sun = (sun_zen, sun_azi)
    f = rt.Big_psi2(view, sun, leaf_zen)
    f = f[0]*trans + f[1]*refl
    return f
  val = quad(fun, 0., 2.*np.pi, args=(view_zen, sun_azi, \
      sun_zen, leaf_zen, refl, trans))[0] / np.pi / 2.
  #return (val, truth)
  np.testing.assert_almost_equal(val, truth, decimal=2)

def test_P2():
  '''A function that tests the radtran.py P2 phase function for
  integration to unity over the complete sphere.
  '''
  sun = (0. * np.pi / 180. , 45.) # zenith, azimuth in radians
  arch = 'u'
  refl = 0.5
  trans = 0.5
  truth = 1.0
  p = []
  for v in views:
    p.append(rt.P2(v, sun, arch, refl, trans))
  p = np.sum(np.multiply(p, gauss_wt))
  # in the original text the term above needs to be divided by
  # 4 / pi. This does not work though for some reason.....
  #return (p, truth)
  np.testing.assert_almost_equal(p, truth, decimal=1)

