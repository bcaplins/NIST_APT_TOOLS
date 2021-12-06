# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# standard imports 
import numpy as np
import matplotlib.pyplot as plt

# custom imports
import GaN_fun
import GaN_type_peak_assignments

import peak_param_determination as ppd
from histogram_functions import bin_dat

plt.close('all')

# Read in data
epos = GaN_fun.load_epos(run_number='R20_07094', 
                         epos_trim=[5000, 4998],
                         fig_idx=999)



import colorcet as cc


def _extents(f):
    """Helper function to determine axis extents based off of the bin edges"""
    delta = f[1] - f[0]
    return [f[0] - delta/2, f[-1] + delta/2]


def plot_2d_histo(ax, N, x_edges, y_edges):
    """Helper function to plot a histogram on an axis"""
    ax.imshow(np.log10(1+np.transpose(N)), aspect='auto',
              extent=_extents(x_edges) + _extents(y_edges),
              origin='lower', cmap=cc.cm.CET_L8,
              interpolation='nearest')


def corrhist(epos, delta=0.1):
    # decide whether m2q or tof correlation histogram and set roi
    dat = epos['m2q']
    roi = [0, 150]
    
    # Determine number of bins and create histogram array
    N = int(np.ceil((roi[1]-roi[0])/delta))
    corrhist = np.zeros([N,N], dtype=int)
    
    # Get multihit indicies
    multi_idxs = np.where(epos['ipp']>1)[0]
    for multi_idx in multi_idxs:
        # Get the multiplicity of the multihit and retieve the m/z data
        n_hits = epos['ipp'][multi_idx]
        cluster = dat[multi_idx:multi_idx+n_hits]
        
        # Loop over each unique set of hit pairs in multihit event and add to histogram
        for i in range(n_hits):
            for j in range(i+1,n_hits):
                idx1 = int(np.floor(cluster[i]/delta))
                idx2 = int(np.floor(cluster[j]/delta))
                if idx1 < N and idx2 < N:
                    corrhist[idx1,idx2] += 1
    
    # Create bin edges
    edges = np.arange(roi[0],roi[1]+delta,delta)
    assert edges.size-1 == N
    
    # Symmetrize the correlation histogram during return
    return (edges, corrhist+corrhist.T-np.diag(np.diag(corrhist)))



# Create correlation histogram
edges, ch = corrhist(epos, delta=0.1)

# add one to get rid of zeros and logscale
ch_log = np.log2(1+ch)

fig = plt.figure(num=66, figsize=(8,8))
fig.clf()
ax = fig.gca()   
plot_2d_histo(ax, ch_log, edges, edges)
ax.set_aspect('equal', 'box')
ax.set_xlabel('ion 1 (m/z)')
ax.set_ylabel('ion 2 (m/z)')

# You can change the color limits if you want to blow out the colorscale
#for im in ax.get_images():
#    im.set_clim(0, 1)

lims = [0,80]
ax.set(xlim=lims, ylim=lims)
ax.set_title('Pairwise Correlation Histogram (log color scale)')
fig.savefig('GaN_m2q_corr_hist.png')




#import initElements_P3
#ed = initElements_P3.initElements()
#
#m1 = ed['N'].isotopes[14][0]
#m2 = ed['Al'].isotopes[27][0]
#mp = (m1+m2)/2
#
#Vd_V0 = np.linspace(0,1,2**5)
#
#m1_eff = m1/(1-Vd_V0*(1-m1/mp))
#m2_eff = m2/(1-Vd_V0*(1-m2/mp))
#
#plt.plot(m1_eff,m2_eff,'w--', alpha=0.5)


