# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 16:46:13 2019

@author: bwc
"""

import matplotlib.pyplot as plt
import numpy as np

from histogram_functions import bin_dat
import peak_param_determination as ppd

def plot_TOF_vs_time(tof,epos,fig_idx,clearFigure=True,user_ylim=[0,1200]):
        
    fig = plt.figure(num=fig_idx)
    if clearFigure:
        fig.clear()
    ax = fig.gca()

    event_idx = np.arange(0,epos.size)    
    ax.plot(event_idx,tof,'.', 
            markersize=.1,
            marker=',',
            markeredgecolor='#1f77b4aa')
    ax.set(xlabel='event index', ylabel='ToF (ns)', ylim=user_ylim)

    ax.grid()

    ax2 = ax.twinx()
    ax2.plot(event_idx,epos['v_dc'],'-', 
            markersize=.1,
            marker=',',
            markeredgecolor='#1f77b4aa',
            color='tab:red')
    ax2.set(ylabel='V_dc')

    fig.tight_layout()
    fig.canvas.manager.window.raise_()


    plt.pause(0.1)
    
    return ax

def plot_m2q_vs_time(m2q,epos,fig_idx,clearFigure=True,user_ylim=[0,100]):
        
    fig = plt.figure(num=fig_idx)
    if clearFigure:
        fig.clear()
    ax = fig.gca()

    event_idx = np.arange(0,epos.size)    
    ax.plot(event_idx,m2q,'.', 
            markersize=.1,
            marker=',',
            markeredgecolor='#1f77b4aa')
    ax.set(xlabel='event index', ylabel='m/z (Da)', ylim=user_ylim)

    ax.grid()

    ax2 = ax.twinx()
    ax2.plot(event_idx,epos['v_dc'],'-', 
            markersize=.1,
            marker=',',
            markeredgecolor='#1f77b4aa',
            color='tab:red')
    ax2.set(ylabel='V_dc')

    fig.tight_layout()
    fig.canvas.manager.window.raise_()


    plt.pause(0.1)
    
    return ax

def plot_bowl_slices(tof,epos,fig_idx,clearFigure=True,user_ylim=[0,1200]):
        
    fig = plt.figure(num=fig_idx)
    if clearFigure:
        fig.clear()
    
    axes = fig.subplots(2,2,sharey=True)

    delta = 5
    
    # Plot x cut
    idxs = np.nonzero(np.abs(epos['y_det']) < delta)
    
    axes[0,0].plot(epos['x_det'][idxs],tof[idxs],'.', 
            markersize=1,
            marker='.',
            markeredgecolor='#1f77b4aa')
    axes[0,0].set(xlabel='x_det', ylabel='ToF (ns)', ylim=user_ylim)
    axes[0,0].set_title('cut along x')
    
    # Plot y cut
    idxs = np.nonzero(np.abs(epos['x_det']) < delta)
    
    axes[0,1].plot(epos['y_det'][idxs],tof[idxs],'.', 
            markersize=1,
            marker='.',
            markeredgecolor='#1f77b4aa')
    axes[0,1].set(xlabel='y_det', ylabel='ToF (ns)', ylim=user_ylim)
    axes[0,1].set_title('cut along y')
    
    
    # Plot diagonal cut (y=x)
    r = np.sign(epos['x_det']+epos['y_det'])*np.sqrt(np.square(epos['x_det'])+np.square(epos['y_det']))
    
    delta = delta*np.sqrt(2)
    idxs = np.nonzero(np.abs(epos['y_det']-epos['x_det']) < delta)
        
    axes[1,0].plot(r[idxs],tof[idxs],'.', 
            markersize=1,
            marker='.',
            markeredgecolor='#1f77b4aa')
    axes[1,0].set(xlabel='r_det', ylabel='ToF (ns)', ylim=user_ylim)
    axes[1,0].set_title('cut along y=x')

    # Plot diagonal cut (y=-x)
    r = np.sign(epos['x_det']-epos['y_det'])*np.sqrt(np.square(epos['x_det'])+np.square(epos['y_det']))
    idxs = np.nonzero(np.abs(epos['y_det']+epos['x_det']) < delta)
        
    axes[1,1].plot(r[idxs],tof[idxs],'.', 
            markersize=1,
            marker='.',
            markeredgecolor='#1f77b4aa')
    axes[1,1].set(xlabel='r_det', ylabel='ToF (ns)', ylim=user_ylim)
    axes[1,1].set_title('cut along y=-x')
    
    for ax in axes.flatten():
        ax.grid()        

    fig.tight_layout()
    fig.canvas.manager.window.raise_()

    plt.pause(0.1)
    
    return axes

def plot_histo(dat,fig_idx,user_label='histo',clearFigure=True,user_xlim=[0,100],user_bin_width=0.01, scale_factor=1):
       
    fig = plt.figure(num=fig_idx)
    if clearFigure:
        fig.clear()
    ax = fig.gca()

    xs, ys = bin_dat(dat,isBinAligned=True,bin_width=user_bin_width,user_roi=user_xlim)
#    ys_sm = ppd.do_smooth_with_gaussian(ys, std=5)

    ax.plot(xs,scale_factor*ys,label=user_label,linewidth=1)
    
    ax.set(xlabel='m/z (Da)', ylabel='counts', xlim=user_xlim)

    ax.grid()

    fig.tight_layout()
    fig.canvas.manager.window.raise_()

    ax.set_yscale('log')
    
    ax.legend()
    
    plt.pause(0.1)
    
    return ax
#
#
#
#plt.close('all')
#
#fig = plt.figure()
#ax = plt.axes()
#ax.plot(pulse_idx, epos['v_dc'])
#ax.set(xlabel='x', ylabel='DC (volts)',
#       title='My First Plot')
#
#
#
#fig = plt.figure(num=1)
#fig.clear()
#ax = plt.axes()
#
#
##ax.plot(wall_time,epos['tof'],'.', 
#ax.plot(wall_time,tof_corr,'.', 
#        markersize=.1,
#        marker=',',
#        markeredgecolor='#1f77b4aa')
#
#ax.set(xlabel='x', ylabel='DC (volts)', ylim=[200, 500],
#       title='My First Plot')
#
#plt.pause(0.1)
#
#
#
#
#
#def plot_vs_time_kde():
#    
#
#    
#    
#    full_new_tof = v_corr.mod_full_voltage_correction(p_volt,epos['tof'],epos['v_dc'])
#    full_new_tof = b_corr.mod_geometric_bowl_correction(p_bowl,full_new_tof,epos['x_det'],epos['y_det'])
#    
#    xmin = np.min(wall_time)
#    xmax = np.max(wall_time)
#    ymin = 400
#    ymax = 865
#    
#    N_PIX = 1024;
#    xedges = np.mgrid[xmin:xmax:N_PIX*1j]
#    yedges = np.mgrid[ymin:ymax:N_PIX*1j]
#    
#    x = wall_time
#    y = full_new_tof
#    
#    H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
#    H = np.log2(1+H.T)
#    
#    fig = plt.figure(num=5)
#    fig.clear()
#    ax = plt.axes()
#    
#    ax.imshow(H, interpolation='nearest', origin='low')
#    
#    
##    plt.imshow(H, interpolation='nearest', origin='low',
##               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
##    
##    
#    plt.pause(0.1)
#    
#    return 0
#
#
#def plot_vs_radius():
#    
#    r = np.sqrt(epos['x_det']**2+epos['y_det']**2)
#    
#    full_new_tof = v_corr.mod_full_voltage_correction(p_volt,epos['tof'],epos['v_dc'])
#    full_new_tof = b_corr.mod_geometric_bowl_correction(p_bowl,full_new_tof,epos['x_det'],epos['y_det'])
#    
#    fig = plt.figure(num=2)
#    fig.clear()
#    ax = plt.axes()
#    
#    
#    #ax.plot(wall_time,epos['tof'],'.', 
#    ax.plot(r,full_new_tof,'.', 
#            markersize=.1,
#            marker=',',
#            markeredgecolor='#1f77b4aa')
#    
#    ax.set(xlabel='x', ylabel='DC (volts)', ylim=[200, 500],
#           title='My First Plot')
#    
#    plt.pause(0.1)
#    
#    return 0
#
#def plot_histos():
#    full_new_tof = v_corr.mod_full_voltage_correction(p_volt,epos['tof'],epos['v_dc'])
#    full_new_tof = b_corr.mod_geometric_bowl_correction(p_bowl,full_new_tof,epos['x_det'],epos['y_det'])
#    
#    full_new_m2q = full_new_tof**2
#    com = np.mean(full_new_m2q)
#    com_ref = np.mean(epos['m2q'])
#    full_new_m2q = (com_ref/com)*full_new_m2q
#    
#    m2q_roi = [0, 100]
#    m2q_bin_width = 0.01;
#    m2q_num_bins = int(np.rint((m2q_roi[1]-m2q_roi[0])/m2q_bin_width))
#    
#    ivas_hist = np.histogram(epos['m2q'],bins=m2q_num_bins,range=(m2q_roi[0],m2q_roi[1]))
#    new_hist = np.histogram(full_new_m2q,bins=m2q_num_bins,range=(m2q_roi[0],m2q_roi[1]))
#    
#    fig = plt.figure(num=3)
#    fig.clear()
#    ax = plt.axes()
#    
#    ax.hist(epos['m2q'],bins=new_hist[1],log=True,histtype='step',label='ivas')
#    ax.hist(full_new_m2q,bins=new_hist[1],log=True,histtype='step',label='new')
#    
#    ax.legend()
#    
#    
#    
#    
#    #ax.plot(wall_time,epos['tof'],'.', 
##    ax.plot(r,full_new_tof,'.', 
##            markersize=.1,
##            marker=',',
##            markeredgecolor='#1f77b4aa')
##   
#    
#    ax.set(xlabel='m2q', ylabel='counts', xlim=[12, 18],
#           title='My First Plot')
#    
#    plt.pause(0.1)
#    
#    return 0
#
#
#
#
