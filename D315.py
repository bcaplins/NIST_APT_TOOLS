# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 18:02:56 2020

@author: lnm
"""

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

# Read in data
D315 = np.loadtxt(r"./D315_2.txt")

Ga_d=D315[:,0]
Ga_c=D315[:,1]
Conc_d=D315[:,2]
Conc_c=D315[:,3]

fig = plt.figure()
plt.plot(Conc_d, Conc_c,color='k')
plt.plot(Ga_d, Ga_c,color='r')
plt.yscale('log')
plt.ylim(1E16,1E22)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
line1 = ax1.plot(Conc_d, Conc_c,color='k')
plt.ylabel('Mg concentration (atoms/cm3)')
plt.xlabel('Depth (microns)')
plt.yscale('log')
plt.ylim(1E16,1E22)
plt.xlim(0,5)
ax2 = fig1.add_subplot(111, sharex=ax1, frameon=False)
line2 = ax2.plot(Ga_d, Ga_c,color='b')
plt.ylabel('Secondary Ions (sounts/s)')
plt.yscale('log')
plt.ylim(1E2,1E9)
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")

fig1.savefig('Mg_SIMS.pdf')
fig1.savefig('Mg_SIMS.jpg', dpi=300)
