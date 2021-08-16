# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# standard imports 
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
import sys
import os
parent_path = '..\\nistapttools'
if parent_path not in sys.path:
    sys.path.append(os.path.abspath(parent_path))    
import time

# custom imports
import apt_fileio
import plotting_stuff
import initElements_P3

import peak_param_determination as ppd

from histogram_functions import bin_dat



fns =['R20_18155-v01.epos',
      'R20_18156-v01.epos',
      'R20_18157-v01.epos',
      'R20_18160-v01.epos',
      'R20_18161-v01.epos',
      'R20_18162-v01.epos']

fold = "GaN epos files"

fig = plt.figure(1)
fig.clf()
ax = fig.gca()

for i in range(len(fns)):
    epos = apt_fileio.read_epos_numpy(fold+"\\"+fns[i])
    plotting_stuff.plot_histo(
        epos['m2q'],
        1,
        user_label=fns[i],
        clearFigure=False,
        user_xlim=[0, 100],
        user_bin_width=0.03,
        scale_factor=10**i,
        user_color=None,
    )

CSR=[0.038,0.005,0.1,0.09,0.24,0.01]
N=[34.1,17,41.5,40.5,49.7,23.2]
Ga=[65.9,83,58.5,59.5,53.3,76.8]    
fig = plt.figure(5)
plt.plot(CSR,Ga,'ko',label="Ga")
plt.plot(CSR,N,'rs',label="N")
plt.xscale('log')
plt.legend()
    

    
    
#fn = fold+"\\"+'R20_18161-v01.epos'
#epos = apt_fileio.read_epos_numpy(fn)
#
#plotting_stuff.plot_m2q_vs_time(
#    epos['m2q'],
#    epos,
#    2,
#    clearFigure=True,
#    user_ylim=[0, 100],
#)
#
#
#     
#pk_lim1 = [22.92, 23.2]
#pk_lim2 = [23.6, 23.74]
#
#idxs = np.where(((epos['m2q']>pk_lim1[0]) & (epos['m2q']<pk_lim1[1])) | ((epos['m2q']>pk_lim2[0]) & (epos['m2q']<pk_lim2[1])))[0]
#
#
#fig = plt.figure(3)
#fig.clf()
#ax = fig.gca()
#
#plt.scatter(epos['x_det'][idxs], epos['y_det'][idxs])











    
    
    
    