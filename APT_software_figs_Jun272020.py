
# standard imports 
import numpy as np
import matplotlib.pyplot as plt
import time

# custom imports
import apt_fileio
import m2q_calib
import plotting_stuff
import initElements_P3

import peak_param_determination as ppd

from histogram_functions import bin_dat
from voltage_and_bowl import do_voltage_and_bowl
import voltage_and_bowl

plt.close('all')


fn = r'C:\Users\capli\Google Drive\NIST\pos_and_epos_files\GaN_manuscript\R20_07148-v01_vbm_corr.epos'
epos = apt_fileio.read_epos_numpy(fn)


# Voltage and bowl correct ToF data
p_volt = np.array([])
p_bowl = np.array([])
tof_vbcorr, p_volt, p_bowl = do_voltage_and_bowl(epos,p_volt,p_bowl)        

tof_vcorr = voltage_and_bowl.mod_full_voltage_correction(p_volt,epos['tof'],epos['v_dc'])
tof_bcorr = voltage_and_bowl.mod_geometric_bowl_correction(p_bowl,epos['tof'],epos['x_det'],epos['y_det'])



epos_vb = epos.copy()
epos_vb['tof'] = tof_vbcorr.copy()



user_ylim=[670,740]

# Plot raw
fig = plt.figure(num=1)
ax = fig.gca()
delta = 3
# Plot x cut
idxs = np.nonzero(np.abs(epos['y_det']) < delta)

ax.plot(epos['x_det'][idxs],tof_vcorr[idxs],'.', 
        markersize=1,
        marker='.',
        markeredgecolor='#1f77b4aa')
ax.set(xlabel='x_det', ylabel='ToF (ns)', ylim=user_ylim)
fig.savefig('APT_Soft_bowl.png')




user_ylim=[600,900]

# Plot raw
fig = plt.figure(num=2)
ax = fig.gca()
delta = 3
# Plot x cut
STEP = 10
ax.plot(np.arange(0,epos.size,STEP )/1000,tof_bcorr[0::STEP ],'.', 
        markersize=1,
        marker='.',
        markeredgecolor='tab:blue')
ax.set(xlabel='ion index (thousands)', ylabel='ToF (ns)', ylim=user_ylim)

ax_twin = ax.twinx()

ax_twin.plot(np.arange(0,epos.size,STEP )/1000,epos['v_dc'][0::STEP ], color='tab:orange', lw=2)
ax_twin.set(ylabel='V_dc')

fig.savefig('APT_Soft_volt.png')



# Plot raw
ax = plotting_stuff.plot_histo(epos['tof'],fig_idx=3,user_label='raw tof', user_bin_width=0.25, user_xlim=[0,1000])


# Plot raw
ax = plotting_stuff.plot_histo(tof_vbcorr,fig_idx=3,user_label='V+B corrected tof', user_bin_width=0.25, user_xlim=[0,1000], clearFigure=False)
ax.set(ylim=[5e1,1e6])
plt.gcf().savefig('APT_Soft_corr_histos.png')



fig = plt.figure(num=3)
ax = fig.gca()
delta = 3
# Plot x cut
STEP = 10
ax.plot(np.arange(0,epos.size,STEP )/1000,tof_bcorr[0::STEP ],'.', 
        markersize=1,
        marker='.',
        markeredgecolor='tab:blue')
ax.set(xlabel='ion index (thousands)', ylabel='ToF (ns)', ylim=user_ylim)

ax_twin = ax.twinx()

ax_twin.plot(np.arange(0,epos.size,STEP )/1000,epos['v_dc'][0::STEP ], color='tab:orange', lw=2)
ax_twin.set(ylabel='V_dc')

fig.savefig('APT_Soft_volt.png')




import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc



def create_histogram(xs, ys, x_roi=None, delta_x=0.1, y_roi=None, delta_y=0.1):
    """Create a 2d histogram of the data, specifying the bin intensity, region
    of interest (on the y-axis), and the spacing of the y bins"""
    # even number
    num_x = int(np.ceil((x_roi[1]-x_roi[0])/delta_x))
    num_y = int(np.ceil((y_roi[1]-y_roi[0])/delta_y))

    return np.histogram2d(xs, ys, bins=[num_x, num_y],
                          range=[x_roi, y_roi],
                          density=False)

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


def corrhist(epos, delta=1):
    dat = epos['tof']
    roi = [0, 5000]
    
    N = int(np.ceil((roi[1]-roi[0])/delta))
    
    corrhist = np.zeros([N,N], dtype=int)
    
    multi_idxs = np.where(epos['ipp']>1)[0]
    
    for multi_idx in multi_idxs:
        n_hits = epos['ipp'][multi_idx]
        cluster = dat[multi_idx:multi_idx+n_hits]
        
        idx1 = -1
        idx2 = -1
        for i in range(n_hits):
            for j in range(i+1,n_hits):
                idx1 = int(np.floor(cluster[i]/delta))
                idx2 = int(np.floor(cluster[j]/delta))
                if idx1 < N and idx2 < N:
                    corrhist[idx1,idx2] += 1
    
    edges = np.arange(roi[0],roi[1]+delta,delta)
    assert edges.size-1 == N
    
    return (edges, corrhist+corrhist.T-np.diag(np.diag(corrhist)))
                
def orthocorrhist(epos):
    dat = epos['tof']
    roi_x = [0, 10000]
    roi_y = [0, 500]
    delta_x = 2
    delta_y = 0.25
    
    Nx = int(np.ceil((roi_x[1]-roi_x[0])/delta_x))
    Ny = int(np.ceil((roi_y[1]-roi_y[0])/delta_y))
    
    corrhist = np.zeros([Nx,Ny], dtype=int)
    
    multi_idxs = np.where(epos['ipp']>1)[0]
    
    for multi_idx in multi_idxs:
        n_hits = epos['ipp'][multi_idx]
        cluster = dat[multi_idx:multi_idx+n_hits]
        
        idx1 = -1
        idx2 = -1
        for i in range(n_hits):
            for j in range(i+1,n_hits):
                new_y = np.abs((cluster[i]-cluster[j])/np.sqrt(2))
                new_x = np.abs((cluster[i]+cluster[j])/np.sqrt(2))
                
                idx1 = int(np.floor(new_x/delta_x))
                idx2 = int(np.floor(new_y/delta_y))
                
                if idx1 < Nx and idx2 < Ny:
                    corrhist[idx1,idx2] += 1
    
    x_edges = np.arange(roi_x[0],roi_x[1]+delta_x,delta_x)
    y_edges = np.arange(roi_y[0],roi_y[1]+delta_y,delta_y)
    
    return (x_edges, y_edges, corrhist)




edges, ch = corrhist(epos_vb)
fig = plt.figure(num=333, figsize=(8,8))
fig.clf()
ax = fig.gca()   
plot_2d_histo(ax, ch, edges, edges)
ax.set_aspect('equal', 'box')
ax.set_xlabel('t1 (ns)')
ax.set_ylabel('t2 (ns)')

for im in ax.get_images():
    im.set_clim(0, 1)

lims = [0,800]
ax.set(xlim=lims, ylim=lims)
fig.savefig('APT_Soft_zoomed_in_corr_hist_gan_bvcorr.png')




edges, ch = corrhist(epos_vb, delta=2)
fig = plt.figure(num=333, figsize=(8,8))
fig.clf()
ax = fig.gca()   
plot_2d_histo(ax, ch, edges, edges)
ax.set_aspect('equal', 'box')
ax.set_xlabel('t1 (ns)')
ax.set_ylabel('t2 (ns)')

for im in ax.get_images():
    im.set_clim(0, 1)
lims = [0,5000]
ax.set(xlim=lims, ylim=lims)
fig.savefig('APT_Soft_zoomed_out_corr_hist_gan_bvcorr.png')



lims = [0,1000]
ax.set(xlim=lims, ylim=lims)
fig.savefig('APT_Soft_zoomed_in_corr_hist_gan_bvcorr.png')


lims = [4000,5000]
ax.set(xlim=lims, ylim=lims)
fig.savefig('APT_Soft_zoomed_in_longtime_corr_hist_gan_bvcorr.png')




edges, ch = corrhist(epos, delta=1)
fig = plt.figure(num=444, figsize=(8,8))
fig.clf()
ax = fig.gca()   
plot_2d_histo(ax, ch, edges, edges)
ax.set_aspect('equal', 'box')
ax.set_xlabel('t1 (ns)')
ax.set_ylabel('t2 (ns)')

for im in ax.get_images():
    im.set_clim(0, 1)
lims = [0,5000]
ax.set(xlim=lims, ylim=lims)
fig.savefig('APT_Soft_zoomed_out_corr_hist_gan_nocorr.png')

lims = [4000,5000]
ax.set(xlim=lims, ylim=lims)
fig.savefig('APT_Soft_zoomed_in_corr_hist_gan_nocorr.png')


def calc_deltas(epos,tof):
    mult_idxs = np.where(epos['ipp']>1)[0]
    
    deltas = np.zeros_like(tof)
    
    import scipy.stats.mstats
    
    for idx in mult_idxs:
        num_ions = epos['ipp'][idx]
        deltas[idx:idx+num_ions] = np.mean(tof[idx:idx+num_ions])-360
    #            deltas[idx:idx+num_ions] = np.sqrt(np.sum(tof[idx:idx+num_ions]**2))/np.sqrt(num_ions)-410
    #        deltas[idx:idx+num_ions] = 2*np.sum(tof[idx:idx+num_ions])/np.prod(tof[idx:idx+num_ions])-360
    #        deltas[idx:idx+num_ions] = scipy.stats.mstats.gmean(tof[idx:idx+num_ions])-300
    #        deltas[idx:idx+num_ions] = np.sqrt(tof[idx:idx+num_ions]**2)/num_ions
    #        
    #        if num_ions == 2:
    #            deltas[idx:idx+num_ions] = np.mean(tof[idx:idx+num_ions])
    #        else:
    #            deltas[idx:idx+num_ions] = np.median(tof[idx:idx+num_ions])
    #        
        
    return deltas



DEL = calc_deltas(epos,epos['tof'])
tof_vcorr = voltage_and_bowl.mod_full_voltage_correction(p_volt,epos['tof']-DEL,epos['v_dc'])+DEL
#tof_vcorr = voltage_and_bowl.mod_full_voltage_correction(p_volt,epos['tof'],epos['v_dc'])
#tof_corr = bah_humbug(p_bowl,tof_vcorr,epos['x_det'],epos['y_det'],D)
#DEL = 2500
DEL = calc_deltas(epos,tof_vcorr)
tof_corr = voltage_and_bowl.mod_geometric_bowl_correction(p_bowl,tof_vcorr-DEL,epos['x_det'],epos['y_det'])+DEL

epos_vb_mod = epos.copy()
epos_vb_mod['tof'] = tof_corr.copy()



edges, ch = corrhist(epos_vb_mod, delta=1)
fig = plt.figure(num=555, figsize=(8,8))
fig.clf()
ax = fig.gca()   
plot_2d_histo(ax, ch, edges, edges)
ax.set_aspect('equal', 'box')
ax.set_xlabel('t1 (ns)')
ax.set_ylabel('t2 (ns)')

for im in ax.get_images():
    im.set_clim(0, 1)
lims = [0,5000]
ax.set(xlim=lims, ylim=lims)
fig.savefig('APT_Soft_zoomed_out_corr_hist_gan_meancorr.png')

lims = [4000,5000]
ax.set(xlim=lims, ylim=lims)
fig.savefig('APT_Soft_zoomed_in_corr_hist_gan_meancorr.png')


lims = [000,1000]
ax.set(xlim=lims, ylim=lims)
fig.savefig('APT_Soft_zoomed_in_earlycorr_hist_gan_meancorr.png')





idxs = np.where(epos['ipp'] == 2)[0]

mean_t = 0.5*(epos['tof'][idxs]+epos['tof'][idxs+1])

idxs = idxs[mean_t>1000]


dts_raw = np.abs(epos['tof'][idxs]-epos['tof'][idxs+1])/np.sqrt(2)
dts_mod = np.abs(epos_vb_mod['tof'][idxs]-epos_vb_mod['tof'][idxs+1])/np.sqrt(2)
dts = np.abs(epos_vb['tof'][idxs]-epos_vb['tof'][idxs+1])/np.sqrt(2)

lims = [0,600]
bin_width = 1


plotting_stuff.plot_histo(dts_raw,321,user_label='raw',clearFigure=True,user_xlim=lims ,user_bin_width=bin_width)
plotting_stuff.plot_histo(dts,321,user_label='normal V+B',clearFigure=False,user_xlim=lims ,user_bin_width=bin_width)
plotting_stuff.plot_histo(dts_mod,321,user_label='adj V+B',clearFigure=False,user_xlim=lims ,user_bin_width=bin_width)

for line in plt.gca().lines:
    line.set_linewidth(2)

plt.gcf().gca().set_xlabel('$\Delta$t (ns)')

plt.gcf().savefig('APT_Soft_deltaT_hists_after1000ns.png')





