#!/usr/bin/python
''' A script that creates a 3D grid of leaf area density (LAD).
The grid spacing, extent, number and position of trees (random
or fixed), leaf area index (LAI) per tree, is required in a 
config file. See scene_in.dat for description of parameters. The 
script require a file name and save the grid together with the 
coordinate mesh in a file. The file is in a pickled format that
can be imported into the 3D model.
The equations are taken from Myneni et al. 1990.
--------------------------------------------------------------
Input: type lad_model_quad.py --help. 
Output: saves data to a disk file and plots scene.
--------------------------------------------------------------
'''

import numpy as np
import matplotlib.pylab as plt
import argparse
import pickle
import pdb

parser = argparse.ArgumentParser(description='Creates a 3D vegetation scene.', epilog='Reads the scene parameters from a config file and saves scene to a .dat file.')
parser.add_argument('-i', dest='ifile', help=\
    'The scene parameter file name.')
parser.add_argument('-o', dest='ofile', help=\
    'The scene output file name.')
parser.add_argument('-p', dest='plot', choices=('y','n'), help=\
    'Choose to plot the scene or not (y or n).')
options = parser.parse_args()

lines = {}
if options.ifile:
  ifile = options.ifile
else:
  ifile = 'scene_in.dat'

ifile = open(ifile, 'r')
i = 0
for line in ifile:
  if line[0] == '#':
    continue
  lines[np.str(i)] = line
  i += 1
space = np.float(lines['0'])
extent = np.array(map(float, lines['1'].split()))
lai = np.float(lines['2'])
pos = np.array(map(float, lines['3'].split()))
pos = np.reshape(pos, (-1, 2))
dim = np.array(map(float, lines['4'].split()))
ifile.close()

if options.ofile:
  ofile = options.ofile
else:
  ofile = 'scene_out.dat' 

if options.plot:
  plot = options.plot
else:
  plot = 'y'

xx = np.arange(0., extent[0], space) + space/2.
yy = np.arange(0., extent[1], space) + space/2.
zz = np.arange(0., extent[2], space) + space/2.
grid = np.zeros((len(xx), len(yy), len(zz)))
mesh = np.meshgrid(xx, yy, zz)
for i, x in enumerate(xx):
  for j, y in enumerate(yy):
    for k, z in enumerate(zz):
      for p in pos:
        x0, y0 = p
        z0 = dim[2]/2.
        dx = np.abs(x - x0)
        dy = np.abs(y - y0)
        dz = np.abs(z - z0)
        if dx < dim[0] and dy < dim[1] and dz < dim[2]: # (eq.39)
          X = dx/dim[0] # (eq.38)
          Y = dy/dim[1]
          Z = dz/dim[2]
          grid[i,j,k] += 1.6875 * lai / dim[2] * (1. - X**2 - Y**2\
            - Z**2 + X**2*Z**2 + Y**2*Z**2 + X**2*Y**2 - \
            X**2*Y**2*Z**2) * space**3. # (eq.37)

if plot == 'y':
  ave_lai = np.sum(grid)/extent[0]/extent[1]
  ext = (0., extent[0], 0., extent[1])
  flatg = np.sum(grid, axis=2) / space**2
  plt.imshow(flatg.T, extent=ext, origin='lower', aspect='equal')
  plt.ylim((0., extent[1]))
  plt.xlim((0., extent[0]))
  plt.title('3D Vegetation Scene')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.colorbar(label='LAI')
  plt.plot(pos.T[0], pos.T[1], 'kx', label='Tree positions')
  plt.legend()
  text = 'Ave. LAI: %.3f' %(ave_lai)
  plt.text(0.5, 0.5, text, horizontalalignment='left', \
      verticalalignment='bottom', bbox=dict(facecolor='white', \
      alpha=1.))
  plt.show()

dic = {}
dic['mesh'] = mesh
dic['grid'] = grid
 
odfile = open(ofile, 'wb')
pickle.dump(dic, odfile)
odfile.close()
print '3D grid and mesh saved to %s' %(ofile)
