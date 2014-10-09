#!/usr/bin/python
''' A script to setup the quadrature set as dictionary with
array values for cosines and weights, based on a defined
file input format. The format should be as follows:
  s4,,
  1,0.3500212,0.3333333
  2,0.8688903,
  s6,,
  .....
The file should be named quad_set.csv.
Where s6 is the level of the qaudrature ie. 6.
The 1,2 etc. is the point no. (n) and weight no.
The 0.3500212 is the ordinate or cosine.
And 0.3333333 is the weight of point type 1.4
A seperate file named quad_pyramid.csv is required 
which containts the numbers of the weights in order 
from top to bottom and left to right in the octal 
reference frame as described in Lewis and Miller 
(1984) p.161 and 162.
The script will create a dictionary saved on disk for 
retrieval by programs with the level as the key, and the
array with [[zenith,azimuth,mu,eta,xi,weight],...] as value.
Run this script interactively and use the function to plot or
save the dictionary to disk.
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import pickle
import pdb

file_name = 'quad_set.csv'
file_txt = open(file_name, 'r')
file_dict = {}
for line in file_txt:
  if line[0] == 's':
    level_key = line[1:].split(',')[0]
    level_vals = []
    continue
  else:
    line_list = line[:-1].split(',')
    level_vals.append(line_list)
  if int(level_key) == int(line.split(',')[0])*2:
    file_dict[level_key] = level_vals[::-1] # reverse order
file_txt.close()

file_txt = open('quad_pyramid.csv','r')
pyr = file_txt.readlines()
file_txt.close()
pyr_dict = {}
for line in pyr:
  line = line.split(',')
  line[-1] = line[-1].split('\n')[0]
  pyr_dict[line[0]] = line[1:]

oct_dict = {}
for key,value in file_dict.iteritems():
  mus = []
  wts = []
  wts_lst = pyr_dict[key]
  nos = []
  for pairs in value:
    nos.append(pairs[0])
    mus.append(float(pairs[1]))
    if pairs[2] != '':
      wts.append(float(pairs[2]))
    else:
      wts.append(999.)
  wts_dict = dict(zip(nos,wts))
  xis = mus[::-1]
  mexw = []
  j = 0
  for i, m in enumerate(mus):
    for x in xis[:i+1]:
      e = np.around(np.sqrt(1. - x**2 - m**2),8)
      wn = int(wts_lst[j])
      j += 1
      w = wts_dict[str(wn)]
      mexw.append([m,e,x,w,wn])
  mexw = np.array(mexw)
  oct_dict[key] = mexw

glob_dict = {}
for key, value in oct_dict.iteritems():
  A = value.copy()
  temp = A[:,1].copy()
  A[:,1] = A[:,2]
  A[:,2] = temp
  A[:,1] = -A[:,1]
  B = value.copy()
  B[:,1] = -B[:,1]
  B[:,2] = -B[:,2]
  C = value.copy()
  temp = C[:,1].copy()
  C[:,1] = C[:,2]
  C[:,2] = temp
  C[:,2] = -C[:,2]
  upper = np.append(value,A,axis=0)
  upper = np.append(upper,B,axis=0)
  upper = np.append(upper,C,axis=0)
  lower = upper.copy()
  lower[:,0] = -lower[:,0]
  glob = np.append(upper,lower,axis=0)
  glob_dict[key] = glob

for key, glob in glob_dict.iteritems():
  Zen = np.array([])
  Azi = np.array([])
  for g in glob:
    eta = g[1]
    xi = g[2]
    mu = g[0]
    zen = np.arccos(g[0])
    if (eta >= 0. and xi >= 0.) or (eta < 0. and xi >= 0.):
      azi = np.arccos(eta/np.sqrt(1. - mu**2))
    elif (eta < 0. and xi < 0.):
      azi = np.abs(np.arcsin(xi/np.sqrt(1. - mu**2))) + np.pi
    elif (eta >= 0. and xi < 0.):
      azi = np.arcsin(xi/np.sqrt(1. - mu**2)) + np.pi*2
    Zen = np.append(Zen,zen)
    Azi = np.append(Azi,azi)
  glob = np.insert(glob, 0, Azi, axis=1)
  glob = np.insert(glob, 0, Zen, axis=1)
  glob_dict[key] = glob

def plot_glob(key):
  '''A function that plots the full global sphere
  of a quadrant with weights depicted as colours.
  Need to have run the program once so that glob_dict
  is in memory before running this function.
  Input: key - a glob_dict[key] key item which points
  to an array of quadrature elements.
  Output: 3D plot of the elements.
  '''
  glob = glob_dict[key]
  fig = plt.figure()
  ax = fig.add_subplot(111,projection='3d')
  x = glob[:,3]
  y = glob[:,4]
  z = glob[:,2]
  c = glob[:,6]
  scat = ax.scatter(x,y,z,c=c)
  fig.colorbar(scat, shrink=0.5, aspect=10)
  ax.set_xlabel(r'$\eta$')
  ax.set_ylabel(r'$\xi$')
  ax.set_zlabel(r'$\mu$')
  plt.title('S%s Discrete Ordinates Quadrature Set' % (key)) 
  plt.show()

def save_globs(glob_dict):
  '''A function to save the dictionary of quadrature sets 
  in zenith, azimuth and weight order for every point.
  The dictionary will have keys corresponding to the order
  of the set eg. '16' for S16. The main program needs to have 
  run before this function can be used. The global_dict variable
  is passed as parameter.
  Input: global_dict.
  Output: saves the dictionary in 'quad_dict.dat'.
  '''
  glob = {}
  for key, value in glob_dict.iteritems():
    arr = np.array([])
    for v in value:
      v  = np.array([v[0], v[1], v[5]])
      arr = np.append(arr, v)
    arr = np.resize(arr,(np.size(arr)/3,3))
    glob[key] = arr
  pdb.set_trace()
  file_name = 'quad_dict.dat'
  fb = open(file_name,'wb')
  pickle.dump(glob,fb)
  print 'Saved the dictionary to %s' % (file_name)
