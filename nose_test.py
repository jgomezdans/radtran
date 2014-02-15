#!/usr/bin/python

''' A script that tests the one_angle.py module.
    Run by: nosetests -v test_one_angle.py
'''

import one_angle as oa
import radtran as rt
import numpy as np
from scipy.integrate import *

def test_one_angle_fluxes():
  '''A function that tests one_angle.py flux calculation.
  '''
  test = oa.rt_layers(Tol=1.e-06,Iter=200,K=40,N=16,\
      Lc=4.,refl=0.175,trans=0.175,refl_s=1.e-16,I0=1.,\
      sun0=120.,arch='s')
  test.solve()
  c,s,a = test.Scalar_flux()
  val = np.array([c,s,a])
  truth = np.array([0.10779073156287999, 0.054603375852551532,\
      0.83760589258456852])
  err = max(abs(val - truth))
  np.testing.assert_almost_equal(err, 0.0)

def test_P2():
  '''A function that tests the radtran.py P2 phase function for
  integration to unity over the complete sphere.
  '''
  n = 16
  sun = (np.pi, 0.)
  arch = 'u'
  refl = 0.5
  trans = 0.5
  g_mu = np.array(rt.gauss_mu[str(n)])
  g_wt = np.array(rt.gauss_wt[str(n)])
  # integrate over mu
  def f_theta(v_ph, sun, arch, refl, trans):
    p_mu = []
    g_theta = np.arccos(g_mu)
    for v_th in g_theta:
      view = (v_th, v_ph)
      p_mu.append(rt.P2(view, sun, arch, refl, trans))
    p = np.sum(np.multiply(p_mu, g_wt))
    return p
  #integrate over phi
  p_ph = []
  f_mu = lambda mu: np.pi * mu + np.pi
  for v_mu in g_mu:
    p_ph.append(f_theta(f_mu(v_mu), sun, arch, refl, trans))
  p2 = np.pi * np.sum(np.multiply(p_ph, g_wt)) / 4. / np.pi
  err = abs(p2 - 1.)
  np.testing.assert_almost_equal(err, 0.0, decimal=2)
  #return p2

# impliment a version for the P() one angle version.

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

