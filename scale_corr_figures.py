# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 14:34:28 2019

@author: bwc
"""

# standard imports 
import numpy as np
import matplotlib.pyplot as plt

# custom imports
import apt_fileio
import plotting_stuff

import peak_param_determination as ppd

from histogram_functions import bin_dat
import scipy.interpolate
import image_registration.register_images

import sel_align_m2q_log_xcorr

import scipy.interpolate
import time
import m2q_calib
import initElements_P3

from voltage_and_bowl import do_voltage_and_bowl
import voltage_and_bowl 

import colorcet as cc
import matplotlib._color_data as mcd


def extents(f):
  delta = f[1] - f[0]
  return [f[0] - delta/2, f[-1] + delta/2]

def create_histogram(ys,cts_per_slice=2**10,y_roi=None,delta_y=1.6e-3):
    
    

    num_y = int(np.ceil(np.abs(np.diff(y_roi))/delta_y/2)*2) # even number
#    num_ly = int(2**np.round(np.log2(np.abs(np.diff(ly_roi))/delta_ly)))-1 # closest power of 2
    print('number of points in ly = ',num_y)
    num_x = int(ys.size/cts_per_slice)
    
    xs = np.arange(ys.size)
    
    N,x_edges,y_edges = np.histogram2d(xs,ys,bins=[num_x,num_y],range=[[1,ys.size],y_roi],density=False)
    return (N,x_edges,y_edges)

def edges_to_centers(*edges):
    centers = []
    for es in edges:
        centers.append((es[0:-1]+es[1:])/2)
    if len(centers)==1:
        centers = centers[0]
    return centers

plt.close('all')


fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\R44_02203-v01.epos"

epos = apt_fileio.read_epos_numpy(fn)
epos = epos[100000::10]


# Voltage and bowl correct ToF data
p_volt = np.array([])
p_bowl = np.array([])
t_i = time.time()
tof_corr, p_volt, p_bowl = do_voltage_and_bowl(epos,p_volt,p_bowl)
print("time to voltage and bowl correct:    "+str(time.time()-t_i)+" seconds")

# Only apply bowl correction
tof_bcorr = voltage_and_bowl.mod_geometric_bowl_correction(p_bowl,epos['tof'],epos['x_det'],epos['y_det'])
ax = plotting_stuff.plot_TOF_vs_time(tof_bcorr,epos,2)


# Plot histogram for steel
fig = plt.figure(figsize=(2*3.14961,2*3.14961),num=321,dpi=100)
plt.clf()
ax1, ax2 = fig.subplots(2,1,sharex=True)


N,x_edges,y_edges = create_histogram(tof_bcorr,y_roi=[400,600],cts_per_slice=2**10,delta_y=0.5)
ax1.imshow(np.log10(1+1*np.transpose(N)), aspect='auto', 
           extent=extents(x_edges) + extents(y_edges), origin='lower', cmap=cc.cm.CET_L8,
           interpolation='bilinear')

ax1.set(ylabel='flight time (ns)')


ax1twin = ax1.twinx()
ax1twin.plot(epos['v_dc'],'-', 
        linewidth=2,
        color=mcd.XKCD_COLORS['xkcd:white'])
ax1twin.set(ylabel='applied voltage (volts)',ylim=[0, 6000],xlim=[0, 400000])


N,x_edges,y_edges = create_histogram(tof_corr,y_roi=[425,475],cts_per_slice=2**10,delta_y=0.5)
ax2.imshow(np.log10(1+1*np.transpose(N)), aspect='auto', 
           extent=extents(x_edges) + extents(y_edges), origin='lower', cmap=cc.cm.CET_L8,
           interpolation='bilinear')

ax2.set(xlabel='ion sequence',ylabel='corrected flight time (ns)')


fig.tight_layout()

fig.savefig(r'Q:\users\bwc\APT\scale_corr_paper\metal_not_wandering.svg', format='svg', dpi=600)


# 

fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\R45_data\R45_04472-v02_allVfromAnn.epos"
epos = apt_fileio.read_epos_numpy(fn)
epos = epos[25000:]



# Voltage and bowl correct ToF data
p_volt = np.array([])
p_bowl = np.array([])
t_i = time.time()
tof_corr, p_volt, p_bowl = do_voltage_and_bowl(epos,p_volt,p_bowl)
print("time to voltage and bowl correct:    "+str(time.time()-t_i)+" seconds")

# Only apply bowl correction
tof_bcorr = voltage_and_bowl.mod_geometric_bowl_correction(p_bowl,epos['tof'],epos['x_det'],epos['y_det'])
ax = plotting_stuff.plot_TOF_vs_time(tof_bcorr,epos,2)


# Plot histogram for sio2
fig = plt.figure(figsize=(2*3.14961,2*3.14961),num=4321,dpi=100)
plt.clf()
ax1, ax2 = fig.subplots(2,1,sharex=True)


N,x_edges,y_edges = create_histogram(tof_bcorr,y_roi=[280,360],cts_per_slice=2**9,delta_y=.5)
ax1.imshow(np.log10(1+1*np.transpose(N)), aspect='auto', 
           extent=extents(x_edges) + extents(y_edges), origin='lower', cmap=cc.cm.CET_L8,
           interpolation='bilinear')

ax1.set(ylabel='flight time (ns)')


ax1twin = ax1.twinx()
ax1twin.plot(epos['v_dc'],'-', 
        linewidth=2,
        color=mcd.XKCD_COLORS['xkcd:white'])
ax1twin.set(ylabel='applied voltage (volts)',ylim=[0000, 8000],xlim=[0, 400000])


N,x_edges,y_edges = create_histogram(tof_corr,y_roi=[280,360],cts_per_slice=2**9,delta_y=.5)
ax2.imshow(np.log10(1+1*np.transpose(N)), aspect='auto', 
           extent=extents(x_edges) + extents(y_edges), origin='lower', cmap=cc.cm.CET_L8,
           interpolation='bilinear')

ax2.set(xlabel='ion sequence',ylabel='corrected flight time (ns)')


fig.tight_layout()
fig.savefig(r'Q:\users\bwc\APT\scale_corr_paper\SiO2_NUV_wandering.svg', format='svg', dpi=600)




# 

fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\R20_07080-v01.epos"
epos = apt_fileio.read_epos_numpy(fn)
#epos = epos[165000:582000]
plotting_stuff.plot_TOF_vs_time(epos['tof'],epos,1,clearFigure=True,user_ylim=[0,1000])


# Voltage and bowl correct ToF data
p_volt = np.array([])
p_bowl = np.array([])
t_i = time.time()
tof_corr, p_volt, p_bowl = do_voltage_and_bowl(epos,p_volt,p_bowl)
print("time to voltage and bowl correct:    "+str(time.time()-t_i)+" seconds")

# Only apply bowl correction
tof_bcorr = voltage_and_bowl.mod_geometric_bowl_correction(p_bowl,epos['tof'],epos['x_det'],epos['y_det'])
ax = plotting_stuff.plot_TOF_vs_time(tof_bcorr,epos,2)


# Plot histogram for sio2
fig = plt.figure(figsize=(2*3.14961,2*3.14961),num=54321,dpi=100)
plt.clf()
ax1, ax2 = fig.subplots(2,1,sharex=True)


N,x_edges,y_edges = create_histogram(tof_bcorr,y_roi=[320,380],cts_per_slice=2**9,delta_y=.5)
ax1.imshow(np.log10(1+1*np.transpose(N)), aspect='auto', 
           extent=extents(x_edges) + extents(y_edges), origin='lower', cmap=cc.cm.CET_L8,
           interpolation='bilinear')

ax1.set(ylabel='flight time (ns)')


ax1twin = ax1.twinx()
ax1twin.plot(epos['v_dc'],'-', 
        linewidth=2,
        color=mcd.XKCD_COLORS['xkcd:white'])
ax1twin.set(ylabel='applied voltage (volts)',ylim=[0000, 5000],xlim=[0, 400000])


N,x_edges,y_edges = create_histogram(tof_corr,y_roi=[320,380],cts_per_slice=2**9,delta_y=.5)
ax2.imshow(np.log10(1+1*np.transpose(N)), aspect='auto', 
           extent=extents(x_edges) + extents(y_edges), origin='lower', cmap=cc.cm.CET_L8,
           interpolation='bilinear')

ax2.set(xlabel='ion sequence',ylabel='corrected flight time (ns)')


fig.tight_layout()


fig.savefig(r'Q:\users\bwc\APT\scale_corr_paper\SiO2_EUV_wandering.svg', format='svg', dpi=600)





## Plot histogram for sio2
#fig = plt.figure(figsize=(2*3.14961,2*3.14961),num=654321,dpi=100)
#plt.clf()
#ax1,ax2, ax3 = fig.subplots(1,3,sharey=True)
#N,x_edges,y_edges = create_histogram(tof_bcorr,y_roi=[0,1000],cts_per_slice=2**10,delta_y=.125)
##ax1.imshow(np.log10(1+1*np.transpose(N)), aspect='auto', 
##           extent=extents(x_edges) + extents(y_edges), origin='lower', cmap=cc.cm.CET_L8,
##           interpolation='bilinear')
#
#event_idx_range_ref = [10000, 20000]
#event_idx_range_mov = [70000, 80000]
#
#x_centers = edges_to_centers(x_edges)
#idxs_ref = (x_centers>=event_idx_range_ref[0]) & (x_centers<=event_idx_range_ref[1])
#idxs_mov = (x_centers>=event_idx_range_mov[0]) & (x_centers<=event_idx_range_mov[1])
#
#ref_hist = np.sum(N[idxs_ref,:],axis=0)
#mov_hist = np.sum(N[idxs_mov,:],axis=0)
#
#y_centers = edges_to_centers(y_edges)
#sc = 300
#
#
#ax1.set(xlim=[84, 96])
#ax2.set(xlim=[348,362])
#ax3.set(xlim=[498,512])
#
#
#ax1.plot(y_centers,ref_hist+mov_hist+2*sc)
#ax2.plot(y_centers,ref_hist+mov_hist+2*sc)
#ax3.plot(y_centers,ref_hist+mov_hist+2*sc)
#
#
#ax1.plot(y_centers,mov_hist+5*sc)
#ax2.plot(y_centers,mov_hist+5*sc)
#ax3.plot(y_centers,mov_hist+5*sc)
#
#N,x_edges,y_edges = create_histogram(1.003*tof_bcorr,y_roi=[0,1000],cts_per_slice=2**10,delta_y=.125)
#mov_hist = np.sum(N[idxs_mov,:],axis=0)
#
#
#
#ax1.plot(y_centers,ref_hist+6*sc)
#ax2.plot(y_centers,ref_hist+6*sc)
#ax3.plot(y_centers,ref_hist+6*sc)
#
#
#ax1.plot(y_centers,mov_hist+4*sc)
#ax2.plot(y_centers,mov_hist+4*sc)
#ax3.plot(y_centers,mov_hist+4*sc)
#
#
#ax1.plot(y_centers,mov_hist+ref_hist+1*sc)
#ax2.plot(y_centers,mov_hist+ref_hist+1*sc)
#ax3.plot(y_centers,mov_hist+ref_hist+1*sc)
#
#N,x_edges,y_edges = create_histogram(1.006*tof_bcorr,y_roi=[0,1000],cts_per_slice=2**10,delta_y=.125)
#mov_hist = np.sum(N[idxs_mov,:],axis=0)
#
#
#ax1.plot(y_centers,mov_hist+3*sc)
#ax2.plot(y_centers,mov_hist+3*sc)
#ax3.plot(y_centers,mov_hist+3*sc)
#
#
#ax1.plot(y_centers,mov_hist+ref_hist)
#ax2.plot(y_centers,mov_hist+ref_hist)
#ax3.plot(y_centers,mov_hist+ref_hist)
#
#
#
#
#
#fig.tight_layout()
#
#
#fig.savefig(r'Q:\users\bwc\APT\scale_corr_paper\correction_idea.svg', format='svg', dpi=600)




#
#def shaded_plot(ax,x,y,idx):
#    sc = 250
#    cols = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
#
#    xlim = ax.get_xlim()
#    
#    idxs = np.nonzero((x>=xlim[0]) & (x<=xlim[1]))
#
#    ax.fill_between(x[idxs], y[idxs]+idx*sc, (idx-0.005)*sc, color=cols[idx])
##    ax.plot(x,y+idx*sc, color='k')
#    return
#
#    
#    
#
## Plot histogram for sio2
#fig = plt.figure(figsize=(2*3.14961,2*3.14961),num=654321,dpi=100)
#plt.clf()
#ax1,ax2 = fig.subplots(1,2,sharey=True)
#N,x_edges,y_edges = create_histogram(tof_bcorr,y_roi=[80,400],cts_per_slice=2**10,delta_y=.125)
##ax1.imshow(np.log10(1+1*np.transpose(N)), aspect='auto', 
##           extent=extents(x_edges) + extents(y_edges), origin='lower', cmap=cc.cm.CET_L8,
##           interpolation='bilinear')
#
#event_idx_range_ref = [10000, 20000]
#event_idx_range_mov = [70000, 80000]
#
#x_centers = edges_to_centers(x_edges)
#idxs_ref = (x_centers>=event_idx_range_ref[0]) & (x_centers<=event_idx_range_ref[1])
#idxs_mov = (x_centers>=event_idx_range_mov[0]) & (x_centers<=event_idx_range_mov[1])
#
#ref_hist = np.sum(N[idxs_ref,:],axis=0)
#mov_hist = np.sum(N[idxs_mov,:],axis=0)
#
#y_centers = edges_to_centers(y_edges)
#
#
#ax1.set(xlim=[87, 93])
#ax2.set(xlim=[352,360])
##ax3.set(xlim=[498,512])
#
#
#shaded_plot(ax1,y_centers,ref_hist+mov_hist,2)
#shaded_plot(ax2,y_centers,ref_hist+mov_hist,2)
#
#shaded_plot(ax1,y_centers,mov_hist,5)
#shaded_plot(ax2,y_centers,mov_hist,5)
#
#N,x_edges,y_edges = create_histogram(1.003*tof_bcorr,y_roi=[80,400],cts_per_slice=2**10,delta_y=.125)
#mov_hist = np.sum(N[idxs_mov,:],axis=0)
#
#shaded_plot(ax1,y_centers,ref_hist,6)
#shaded_plot(ax2,y_centers,ref_hist,6)
#
#
#shaded_plot(ax1,y_centers,mov_hist,4)
#shaded_plot(ax2,y_centers,mov_hist,4)
#
#
#shaded_plot(ax1,y_centers,mov_hist+ref_hist,1)
#shaded_plot(ax2,y_centers,mov_hist+ref_hist,1)
#
#
#N,x_edges,y_edges = create_histogram(1.006*tof_bcorr,y_roi=[80,400],cts_per_slice=2**10,delta_y=.125)
#mov_hist = np.sum(N[idxs_mov,:],axis=0)
#
#
#shaded_plot(ax1,y_centers,mov_hist,3)
#shaded_plot(ax2,y_centers,mov_hist,3)
#
#
#shaded_plot(ax1,y_centers,mov_hist+ref_hist,0)
#shaded_plot(ax2,y_centers,mov_hist+ref_hist,0)
#
#
#
#fig.tight_layout()
#
#
#fig.savefig(r'Q:\users\bwc\APT\scale_corr_paper\correction_idea.svg', format='svg', dpi=600)



def shaded_plot(ax,x,y,idx,col_idx=None):
    if col_idx is None:
        col_idx = idx
        
    sc = 100
    cols = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    xlim = ax.get_xlim()
    
    idxs = np.nonzero((x>=xlim[0]) & (x<=xlim[1]))

    ax.fill_between(x[idxs], y[idxs]+idx*sc, (idx-0.005)*sc, color=cols[col_idx])
#    ax.plot(x,y+idx*sc, color='k')
    return

    
    

# Plot histogram for sio2
fig = plt.figure(figsize=(2*3.14961,2*3.14961),num=654321,dpi=100)
plt.clf()
ax2 = fig.subplots(1,1)
N,x_edges,y_edges = create_histogram(tof_bcorr,y_roi=[80,400],cts_per_slice=2**10,delta_y=0.0625)
#ax1.imshow(np.log10(1+1*np.transpose(N)), aspect='auto', 
#           extent=extents(x_edges) + extents(y_edges), origin='lower', cmap=cc.cm.CET_L8,
#           interpolation='bilinear')

event_idx_range_ref = [10000, 20000]
event_idx_range_mov = [70000, 80000]

x_centers = edges_to_centers(x_edges)
idxs_ref = (x_centers>=event_idx_range_ref[0]) & (x_centers<=event_idx_range_ref[1])
idxs_mov = (x_centers>=event_idx_range_mov[0]) & (x_centers<=event_idx_range_mov[1])

ref_hist = np.sum(N[idxs_ref,:],axis=0)
mov_hist = np.sum(N[idxs_mov,:],axis=0)

y_centers = edges_to_centers(y_edges)



ax2.set(xlim=[352,366])
#ax3.set(xlim=[498,512])

N,x_edges,y_edges = create_histogram(0.998*tof_bcorr,y_roi=[80,400],cts_per_slice=2**10,delta_y=0.0625)
mov_hist = np.sum(N[idxs_mov,:],axis=0)

#shaded_plot(ax2,y_centers,ref_hist+mov_hist,2)
shaded_plot(ax2,y_centers,mov_hist,2,2)

N,x_edges,y_edges = create_histogram(1.00317*tof_bcorr,y_roi=[80,400],cts_per_slice=2**10,delta_y=0.0625)
mov_hist = np.sum(N[idxs_mov,:],axis=0)

shaded_plot(ax2,y_centers,ref_hist,3,3)
shaded_plot(ax2,y_centers,mov_hist,1,1)
#shaded_plot(ax2,y_centers,mov_hist+ref_hist,1)


N,x_edges,y_edges = create_histogram(1.008*tof_bcorr,y_roi=[80,400],cts_per_slice=2**10,delta_y=0.0625)
mov_hist = np.sum(N[idxs_mov,:],axis=0)


shaded_plot(ax2,y_centers,mov_hist,0,col_idx=0)
#shaded_plot(ax2,y_centers,mov_hist+ref_hist,0)



fig.tight_layout()


fig.savefig(r'Q:\users\bwc\APT\scale_corr_paper\correction_idea1.svg', format='svg', dpi=600)




cs = np.linspace(0.99, 1.01, 256)
dp = np.zeros_like(cs)
for idx, c in enumerate(cs):
    N,x_edges,y_edges = create_histogram(c*tof_bcorr,y_roi=[80,400],cts_per_slice=2**10,delta_y=0.0625)
    mov_hist = np.sum(N[idxs_mov,:],axis=0)
    dp[idx] = np.sum((mov_hist/np.sum(mov_hist))*(ref_hist/np.sum(ref_hist)))
    
    

# Plot histogram for sio2

fig = plt.figure(figsize=(2*3.14961,2*3.14961),num=7654321,dpi=100)
plt.clf()
ax1 = fig.subplots(1,1)



ax1.set(xlim=[0.99, 1.01],ylim=[0,1.1])

f = scipy.interpolate.interp1d(cs,dp/np.max(dp))

cols = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

xq = [0.998, 1.00317, 1.008]
for idx in [0,1,2]:
    ax1.plot(xq[idx],f(xq[idx]),'o',markersize=10,color=cols[2-idx])


ax1.plot(cs,dp/np.max(dp),'k')
fig.tight_layout()


fig.savefig(r'Q:\users\bwc\APT\scale_corr_paper\correction_idea2.svg', format='svg', dpi=600)










fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\R45_data\R45_04472-v03.epos"
#fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\R45_data\R45_04472-v02.epos"
#    fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\R20_07148-v01.epos" # Mg doped
#    fn = fn[:-5]+'_vbm_corr.epos'


epos = apt_fileio.read_epos_numpy(fn)
epos = epos[25000:]
epos = epos[:400000]

fake_tof = np.sqrt((296/312)*epos['m2q']/1.393e-4)


cts_per_slice=2**8
#m2q_roi = [0.9,190]
tof_roi = [0, 1000]
import time
t_start = time.time()
pointwise_scales,piecewise_scales = sel_align_m2q_log_xcorr.get_all_scale_coeffs(epos['m2q'],
                                                         m2q_roi=[0.8,80],
                                                         cts_per_slice=cts_per_slice,
                                                         max_scale=1.15)
t_end = time.time()
print('Total Time = ',t_end-t_start)

fake_tof_corr = fake_tof/np.sqrt(pointwise_scales)

m2q_corr = epos['m2q']/pointwise_scales

# Plot histogram for sio2
fig = plt.figure(figsize=(2*3.14961,2*3.14961),num=87654321,dpi=100)
plt.clf()
ax1, ax2 = fig.subplots(2,1,sharex=True)


N,x_edges,y_edges = create_histogram(fake_tof,y_roi=[280,360],cts_per_slice=cts_per_slice,delta_y=.5)
ax1.imshow(np.log10(1+1*np.transpose(N)), aspect='auto', 
           extent=extents(x_edges) + extents(y_edges), origin='lower', cmap=cc.cm.CET_L8,
           interpolation='bilinear')

ax1.set(ylabel='flight time (ns)')


ax1twin = ax1.twinx()
ax1twin.plot(pointwise_scales,'-', 
        linewidth=1,
        color=mcd.XKCD_COLORS['xkcd:white'])
ax1twin.set(ylabel='correction factor, c',ylim=[0.95, 1.3],xlim=[0, 400000])


N,x_edges,y_edges = create_histogram(fake_tof_corr,y_roi=[280,360],cts_per_slice=cts_per_slice,delta_y=.5)
ax2.imshow(np.log10(1+1*np.transpose(N)), aspect='auto', 
           extent=extents(x_edges) + extents(y_edges), origin='lower', cmap=cc.cm.CET_L8,
           interpolation='bilinear')

ax2.set(xlabel='ion sequence',ylabel='corrected flight time (ns)')


fig.tight_layout()
fig.savefig(r'Q:\users\bwc\APT\scale_corr_paper\SiO2_NUV_corrected.svg', format='svg', dpi=600)










def shaded_plot(ax,x,y,idx,col_idx=None,min_val=None):
    if col_idx is None:
        col_idx = idx
        
    if min_val is None:
        min_val = np.min(y)
    
    sc = 150
    cols = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    xlim = ax.get_xlim()
    
    idxs = np.nonzero((x>=xlim[0]) & (x<=xlim[1]))

    ax.fill_between(x[idxs], y[idxs], min_val, color=cols[col_idx])
#    ax.plot(x,y+idx*sc, color='k')
    return







fig = plt.figure(constrained_layout=True,figsize=(2*3.14961,2*3.14961),num=87654321,dpi=100)
plt.clf()

gs = plt.GridSpec(2, 3, figure=fig)
ax0 = fig.add_subplot(gs[0, :])
# identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
ax1 = fig.add_subplot(gs[1,0:2])
#ax2 = fig.add_subplot(gs[1,1])
ax3 = fig.add_subplot(gs[1,2])






dat = epos['m2q']
user_bin_width = 0.03
user_xlim = [0,65]
ax0.set(xlim=user_xlim)



dat = m2q_corr
xs, ys = bin_dat(dat,isBinAligned=True,bin_width=user_bin_width,user_roi=user_xlim)
shaded_plot(ax0,xs,100*(1+ys),1,min_val=100)


dat = epos['m2q']
xs, ys = bin_dat(dat,isBinAligned=True,bin_width=user_bin_width,user_roi=user_xlim)
shaded_plot(ax0,xs,1+ys,0,min_val=1)






ax0.set(xlabel='m/z (Da)', ylabel='counts', xlim=user_xlim)
ax0.set_yscale('log')




user_bin_width = 0.01
user_xlim = [13,19]
ax1.set(xlim=user_xlim)




dat = m2q_corr
xs, ys = bin_dat(dat,isBinAligned=True,bin_width=user_bin_width,user_roi=user_xlim)
shaded_plot(ax1,xs,100*(1+ys),1,min_val=100)


dat = epos['m2q']
xs, ys = bin_dat(dat,isBinAligned=True,bin_width=user_bin_width,user_roi=user_xlim)
shaded_plot(ax1,xs,1+ys,0,min_val=1)


ax1.set(xlabel='m/z (Da)', ylabel='counts', xlim=user_xlim)
ax1.set_yscale('log')

#
#
##user_bin_width = 0.01
#user_xlim = [30,34]
#ax2.set(xlim=user_xlim)
#
#
#dat = m2q_corr
#xs, ys = bin_dat(dat,isBinAligned=True,bin_width=user_bin_width,user_roi=user_xlim)
#shaded_plot(ax2,xs,100*(1+ys),1,min_val=100)
#
#
#dat = epos['m2q']
#xs, ys = bin_dat(dat,isBinAligned=True,bin_width=user_bin_width,user_roi=user_xlim)
#shaded_plot(ax2,xs,1+ys,0,min_val=1)
#
#
#ax2.set(xlabel='m/z (Da)', ylabel='counts', xlim=user_xlim)
#ax2.set_yscale('log')




#user_bin_width = 0.01
user_xlim = [58,64]
ax3.set(xlim=user_xlim)


dat = m2q_corr
xs, ys = bin_dat(dat,isBinAligned=True,bin_width=user_bin_width,user_roi=user_xlim)
shaded_plot(ax3,xs,100*(1+ys),1,min_val=100)


dat = epos['m2q']
xs, ys = bin_dat(dat,isBinAligned=True,bin_width=user_bin_width,user_roi=user_xlim)
shaded_plot(ax3,xs,1+ys,0,min_val=1)


ax3.set(xlabel='m/z (Da)', ylabel='counts', xlim=user_xlim)
ax3.set_yscale('log')


ax0.set(ylim=[1,None])
ax1.set(ylim=[1,None])
ax2.set(ylim=[1,None])
ax3.set(ylim=[1,None])

fig.tight_layout()

fig.savefig(r'Q:\users\bwc\APT\scale_corr_paper\SiO2_NUV_corrected_hist.svg', format='svg', dpi=600)





















fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\R45_data\R45_00504-v56.epos"
#fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\R45_data\R45_04472-v02.epos"
#    fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\R20_07148-v01.epos" # Mg doped
#    fn = fn[:-5]+'_vbm_corr.epos'


epos = apt_fileio.read_epos_numpy(fn)
#epos = epos[25000:]
#epos = epos[:400000]


cts_per_slice=2**8

import time
t_start = time.time()
pointwise_scales,piecewise_scales = sel_align_m2q_log_xcorr.get_all_scale_coeffs(epos['m2q'],
                                                         m2q_roi=[10,250],
                                                         cts_per_slice=cts_per_slice,
                                                         max_scale=1.15)
t_end = time.time()
print('Total Time = ',t_end-t_start)

m2q_corr = epos['m2q']/pointwise_scales








def shaded_plot(ax,x,y,idx,col_idx=None,min_val=None):
    if col_idx is None:
        col_idx = idx
        
    if min_val is None:
        min_val = np.min(y)
    
    
    cols = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    xlim = ax.get_xlim()
    
    idxs = np.nonzero((x>=xlim[0]) & (x<=xlim[1]))

    ax.fill_between(x[idxs], y[idxs], min_val, color=cols[col_idx])
#    ax.plot(x,y+idx*sc, color='k')
    return







fig = plt.figure(constrained_layout=True,figsize=(2*3.14961,2*3.14961),num=87654321,dpi=100)
plt.clf()

gs = plt.GridSpec(2, 3, figure=fig)
ax0 = fig.add_subplot(gs[0, :])
# identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
ax1 = fig.add_subplot(gs[1,0:2])
#ax2 = fig.add_subplot(gs[1,1])
ax3 = fig.add_subplot(gs[1,2])





dat = epos['m2q']
user_bin_width = 0.03
user_xlim = [0,200]
ax0.set(xlim=user_xlim)



dat = m2q_corr
xs, ys = bin_dat(dat,isBinAligned=True,bin_width=user_bin_width,user_roi=user_xlim)
shaded_plot(ax0,xs,10*(1+ys),1,min_val=10)


dat = epos['m2q']
xs, ys = bin_dat(dat,isBinAligned=True,bin_width=user_bin_width,user_roi=user_xlim)
shaded_plot(ax0,xs,1+ys,0,min_val=1)






ax0.set(xlabel='m/z (Da)', ylabel='counts', xlim=user_xlim)
ax0.set_yscale('log')
ax0.set(ylim=[10,None])



user_bin_width = 0.01
user_xlim = [45,55]
ax1.set(xlim=user_xlim)




dat = m2q_corr
xs, ys = bin_dat(dat,isBinAligned=True,bin_width=user_bin_width,user_roi=user_xlim)
shaded_plot(ax1,xs,10*(1+ys),1,min_val=10)


dat = epos['m2q']
xs, ys = bin_dat(dat,isBinAligned=True,bin_width=user_bin_width,user_roi=user_xlim)
shaded_plot(ax1,xs,1+ys,0,min_val=1)


ax1.set(xlabel='m/z (Da)', ylabel='counts', xlim=user_xlim)
ax1.set_yscale('log')
ax1.set(ylim=[10,None])

#
#
##user_bin_width = 0.01
#user_xlim = [30,34]
#ax2.set(xlim=user_xlim)
#
#
#dat = m2q_corr
#xs, ys = bin_dat(dat,isBinAligned=True,bin_width=user_bin_width,user_roi=user_xlim)
#shaded_plot(ax2,xs,100*(1+ys),1,min_val=100)
#
#
#dat = epos['m2q']
#xs, ys = bin_dat(dat,isBinAligned=True,bin_width=user_bin_width,user_roi=user_xlim)
#shaded_plot(ax2,xs,1+ys,0,min_val=1)
#
#
#ax2.set(xlabel='m/z (Da)', ylabel='counts', xlim=user_xlim)
#ax2.set_yscale('log')




#user_bin_width = 0.01
user_xlim = [168,178]
ax3.set(xlim=user_xlim)


dat = m2q_corr
xs, ys = bin_dat(dat,isBinAligned=True,bin_width=user_bin_width,user_roi=user_xlim)
shaded_plot(ax3,xs,10*(1+ys),1,min_val=10)


dat = epos['m2q']
xs, ys = bin_dat(dat,isBinAligned=True,bin_width=user_bin_width,user_roi=user_xlim)
shaded_plot(ax3,xs,1+ys,0,min_val=1)


ax3.set(xlabel='m/z (Da)', ylabel='counts', xlim=user_xlim)
ax3.set_yscale('log')
ax3.set(ylim=[10,None])



fig.tight_layout()

fig.savefig(r'Q:\users\bwc\APT\scale_corr_paper\Ceria_NUV_corrected_hist.svg', format='svg', dpi=600)
