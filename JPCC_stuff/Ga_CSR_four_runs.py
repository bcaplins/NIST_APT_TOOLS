# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 16:54:22 2020

@author: lnm
"""

# standard imports 
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

CSR=[1.1,0.55,0.33,0.13]
Ga=[48.1, 49.0,49.7,50.1]
error=[0.2,0.3,0.3,0.4]

csr_fig = plt.figure()
csr_fig.set_size_inches(w=3.345, h=3.345)

ax = csr_fig.gca()
ax.errorbar(CSR,Ga,yerr=error,fmt='.',capsize=4)


xlim = [1e-2, 1e1]

ax.set(xlabel='CSR', ylabel='Ga %', ylim=[0, 1], xlim=xlim)
ax.plot(xlim,[0.5,0.5],'k--', label='nominal')
    
ax.set_ylim(40,60)
#ax.legend()
ax.set_title('det radius and time based chunking')
ax.set_xscale('log')
ax.grid(b=True)
csr_fig.tight_layout()

csr_fig.savefig('GaN_CSR_four_runs.pdf')
#csr_fig.savefig('GaN_CSR_plot.jpg', dpi=300)

    
#ax.errorbar(csr.flatten(),Ga_comp.flatten(),yerr=Ga_comp_std.flatten(),fmt='.',capsize=4,label='det based (radial)')


