# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 17:29:36 2019

@author: bwc
"""
import csv

import numpy as np
import matplotlib.pyplot as plt

csr = []
ga = []

with open(r'Q:\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\csv_vs_ga.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        csr.append(float(row[0]))
        ga.append(float(row[1]))
        
        
        
csr = np.array(csr)
ga = np.array(ga)


fig = plt.figure(num=1000)
fig.clear()
ax = fig.gca()

ax.plot(csr,ga,'rs',label='Ga')
ax.plot(csr,1-ga,'ko',label='N')
ax.plot([1e-4, 1e1],[0.5, 0.5],'k--')


ax.set(xlabel='CSR (Ga$^{++}$/Ga$^{+}$)', ylabel='apparent atomic %')
ax.set_xlim([2e-4, 10])
ax.set_ylim([0, 1])
#ax.grid(which='both')
ax.set_xscale('log')    

fig.tight_layout()


ax.legend()
