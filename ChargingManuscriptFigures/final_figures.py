# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 15:15:12 2019

@author: bwc
"""


# standard imports 
import numpy as np
import matplotlib.pyplot as plt

# custom imports
import apt_fileio

import peak_param_determination as ppd

from histogram_functions import bin_dat
import scipy.interpolate
import image_registration.register_images

#import sel_align_m2q_log_xcorr
import scaling_correction
import time
import m2q_calib

from voltage_and_bowl import do_voltage_and_bowl
import voltage_and_bowl 

import colorcet as cc
import matplotlib._color_data as mcd
import matplotlib

FIGURE_SCALE_FACTOR = 2



def colorbar():
    fig = plt.gcf()
    ax = fig.gca()
#    
#    norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=False)
#    
    fig.colorbar()

    return None


def extents(f):
  delta = f[1] - f[0]
  return [f[0] - delta/2, f[-1] + delta/2]

def create_histogram(ys,cts_per_slice=2**10,y_roi=None,delta_y=0.1):
    # even number
    num_y = int(np.ceil(np.abs(np.diff(y_roi))/delta_y/2)*2) 
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

def plot_2d_histo(ax,N,x_edges,y_edges):
    pos1 = ax.imshow(np.log10(1+1*np.transpose(N)), aspect='auto', 
                     extent=extents(x_edges) + extents(y_edges), origin='lower', cmap=cc.cm.CET_L8,
                     interpolation='bicubic')
    
    ax.set_xticks([0,100000,200000,300000,400000])#    ax.set_xticklabels(["\n".join(x) for x in data.index])
    
    return pos1


def steel():
    
    # Load and subsample data (for easier processing)    
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
        
    # Plot histogram for steel
    fig = plt.figure(figsize=(FIGURE_SCALE_FACTOR*3.14961,FIGURE_SCALE_FACTOR*3.14961),num=1,dpi=100)
    plt.clf()
    ax1, ax2 = fig.subplots(2,1,sharex=True)
        
    N,x_edges,y_edges = create_histogram(tof_bcorr,y_roi=[400,600],cts_per_slice=2**9,delta_y=0.25)
    im = plot_2d_histo(ax1,N,x_edges,y_edges)    
#    plt.colorbar(im)
    ax1.set(ylabel='flight time (ns)')

    
    ax1twin = ax1.twinx()
    ax1twin.plot(epos['v_dc'],'-', 
            linewidth=2,
            color=mcd.XKCD_COLORS['xkcd:white'])
    ax1twin.set(ylabel='applied voltage (volts)',ylim=[0, 6000],xlim=[0, 400000])
    
    N,x_edges,y_edges = create_histogram(tof_corr,y_roi=[425,475],cts_per_slice=2**9,delta_y=0.25)
    im = plot_2d_histo(ax2,N,x_edges,y_edges)
#    plt.colorbar(im)
    
    ax2.set(xlabel='ion sequence',ylabel='corrected flight time (ns)')    
    
    fig.tight_layout()
    
    fig.savefig(r'Q:\users\bwc\APT\scale_corr_paper\metal_not_wandering.pdf', format='pdf', dpi=600)
    
    return 0
    

def sio2_R45():
    
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
    
    # Plot histogram for sio2
    fig = plt.figure(figsize=(FIGURE_SCALE_FACTOR*3.14961,FIGURE_SCALE_FACTOR*3.14961),num=2,dpi=100)
    plt.clf()
    ax1, ax2 = fig.subplots(2,1,sharex=True)
    
    N,x_edges,y_edges = create_histogram(tof_bcorr,y_roi=[280,360],cts_per_slice=2**9,delta_y=.25)
    im = plot_2d_histo(ax1,N,x_edges,y_edges)    
    plt.colorbar(im)

    ax1.set(ylabel='flight time (ns)')
    
    
    ax1twin = ax1.twinx()
    ax1twin.plot(epos['v_dc'],'-', 
            linewidth=2,
            color=mcd.XKCD_COLORS['xkcd:white'])
    ax1twin.set(ylabel='applied voltage (volts)',ylim=[0000, 8000],xlim=[0, 400000])
        
    N,x_edges,y_edges = create_histogram(tof_corr,y_roi=[280,360],cts_per_slice=2**9,delta_y=.25)
    im = plot_2d_histo(ax2,N,x_edges,y_edges)    
    plt.colorbar(im)

    ax2.set(xlabel='ion sequence',ylabel='corrected flight time (ns)')
    
    fig.tight_layout()
    fig.savefig(r'Q:\users\bwc\APT\scale_corr_paper\SiO2_NUV_wandering.pdf', format='pdf', dpi=600)
    
    return 0


def sio2_R44():
            
    fn = r"C:\Users\bwc\Documents\NetBeansProjects\R44_03200\recons\recon-v02\default\R44_03200-v02.epos"
    epos = apt_fileio.read_epos_numpy(fn)
        
    # Voltage and bowl correct ToF data
    p_volt = np.array([])
    p_bowl = np.array([])
    t_i = time.time()
    tof_corr, p_volt, p_bowl = do_voltage_and_bowl(epos,p_volt,p_bowl)
    print("time to voltage and bowl correct:    "+str(time.time()-t_i)+" seconds")
    
    # Only apply bowl correction
    tof_bcorr = voltage_and_bowl.mod_geometric_bowl_correction(p_bowl,epos['tof'],epos['x_det'],epos['y_det'])    
    
    # Plot histogram for sio2
    fig = plt.figure(figsize=(FIGURE_SCALE_FACTOR*3.14961,FIGURE_SCALE_FACTOR*3.14961),num=3,dpi=100)
    plt.clf()
    ax1, ax2 = fig.subplots(2,1,sharex=True)
    
    roi = [1400000,1800000]
    
    N,x_edges,y_edges = create_histogram(tof_bcorr[roi[0]:roi[1]],y_roi=[300,310],cts_per_slice=2**7,delta_y=.2)
    im = plot_2d_histo(ax1,N,x_edges,y_edges)    
    plt.colorbar(im)


    
    ax1.set(ylabel='flight time (ns)')
    
    
    ax1twin = ax1.twinx()
    ax1twin.plot(epos['v_dc'][roi[0]:roi[1]],'-', 
            linewidth=2,
            color=mcd.XKCD_COLORS['xkcd:white'])
    ax1twin.set(ylabel='applied voltage (volts)',ylim=[0000, 7000],xlim=[0,None])
    
    
    N,x_edges,y_edges = create_histogram(tof_corr[roi[0]:roi[1]],y_roi=[300,310],cts_per_slice=2**7,delta_y=0.2)
    im = plot_2d_histo(ax2,N,x_edges,y_edges)    
    plt.colorbar(im)


    ax2.set(xlabel='ion sequence',ylabel='corrected flight time (ns)')
    
    ax2.set_xlim(0,roi[1]-roi[0])
    
    
    fig.tight_layout()
    fig.savefig(r'Q:\users\bwc\APT\scale_corr_paper\Figure_R44NUV.pdf', format='pdf', dpi=600)
    
    
    

    
    return 0 

def sio2_R20():
    
    fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\R20_07080-v01.epos"
    epos = apt_fileio.read_epos_numpy(fn)
    #epos = epos[165000:582000]
    
    # Voltage and bowl correct ToF data
    p_volt = np.array([])
    p_bowl = np.array([])
    t_i = time.time()
    tof_corr, p_volt, p_bowl = do_voltage_and_bowl(epos,p_volt,p_bowl)
    print("time to voltage and bowl correct:    "+str(time.time()-t_i)+" seconds")
    
    # Only apply bowl correction
    tof_bcorr = voltage_and_bowl.mod_geometric_bowl_correction(p_bowl,epos['tof'],epos['x_det'],epos['y_det'])

    # Plot histogram for sio2
    fig = plt.figure(figsize=(FIGURE_SCALE_FACTOR*3.14961,FIGURE_SCALE_FACTOR*3.14961),num=4,dpi=100)
    plt.clf()
    ax1, ax2 = fig.subplots(2,1,sharex=True)
    
    N,x_edges,y_edges = create_histogram(tof_bcorr,y_roi=[320,380],cts_per_slice=2**9,delta_y=.25)
    plot_2d_histo(ax1,N,x_edges,y_edges)    
    ax1.set(ylabel='flight time (ns)')
    
    ax1twin = ax1.twinx()
    ax1twin.plot(epos['v_dc'],'-', 
            linewidth=2,
            color=mcd.XKCD_COLORS['xkcd:white'])
    ax1twin.set(ylabel='applied voltage (volts)',ylim=[0000, 5000],xlim=[0, 400000])
    
    N,x_edges,y_edges = create_histogram(tof_corr,y_roi=[320,380],cts_per_slice=2**9,delta_y=.25)
    plot_2d_histo(ax2,N,x_edges,y_edges)        
    ax2.set(xlabel='ion sequence',ylabel='corrected flight time (ns)')
    
    fig.tight_layout()
    
    fig.savefig(r'Q:\users\bwc\APT\scale_corr_paper\SiO2_EUV_wandering.pdf', format='pdf', dpi=600)

    
    return 0




def corr_idea():

    
    fig = plt.figure(num=5)
    plt.close(fig)
    fig = plt.figure(constrained_layout=True,figsize=(FIGURE_SCALE_FACTOR*3.14961,FIGURE_SCALE_FACTOR*1.5*3.14961),num=5,dpi=100)
    
    gs = fig.add_gridspec(3, 1)
    
    ax2 = fig.add_subplot(gs[:2, :])
    ax1 = fig.add_subplot(gs[2, :])

    
    def shaded_plot(ax,x,y,idx,col_idx=None):
        if col_idx is None:
            col_idx = idx
        sc = 50
        cols = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        xlim = ax.get_xlim()
        idxs = np.nonzero((x>=xlim[0]) & (x<=xlim[1]))    
        ax.fill_between(x[idxs], y[idxs]+idx*sc, (idx-0.005)*sc, color=cols[col_idx])
    #    ax.plot(x,y+idx*sc, color='k')
        return
    
    fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\R45_data\R45_04472-v02_allVfromAnn.epos"
    epos = apt_fileio.read_epos_numpy(fn)
    epos = epos[25000:]
    
    # Voltage and bowl correct ToF data
    p_volt = np.array([])
    p_bowl = np.array([])
    t_i = time.time()
    tof_corr, p_volt, p_bowl = do_voltage_and_bowl(epos,p_volt,p_bowl)
    print("time to voltage and bowl correct:    "+str(time.time()-t_i)+" seconds")
    
    # Plot histogram for sio2
#    fig = plt.figure(figsize=(2*3.14961,2*3.14961),num=654321,dpi=100)
#    plt.clf()
#    ax2 = fig.subplots(1,1)
    N,x_edges,y_edges = create_histogram(tof_corr,y_roi=[80,400],cts_per_slice=2**10,delta_y=0.0625)
    #ax1.imshow(np.log10(1+1*np.transpose(N)), aspect='auto', 
    #           extent=extents(x_edges) + extents(y_edges), origin='lower', cmap=cc.cm.CET_L8,
    #           interpolation='bilinear')
    
    event_idx_range_ref = [0, 0+1024]
    event_idx_range_mov = [124000, 124000+1024]
    
    x_centers = edges_to_centers(x_edges)
    idxs_ref = (x_centers>=event_idx_range_ref[0]) & (x_centers<=event_idx_range_ref[1])
    idxs_mov = (x_centers>=event_idx_range_mov[0]) & (x_centers<=event_idx_range_mov[1])
    
    ref_hist = np.sum(N[idxs_ref,:],axis=0)
    mov_hist = np.sum(N[idxs_mov,:],axis=0)
    
    y_centers = edges_to_centers(y_edges)
    
    
    
    ax2.set(xlim=[290,320])
    
    N,x_edges,y_edges = create_histogram(0.98*tof_corr,y_roi=[80,400],cts_per_slice=2**10,delta_y=0.0625)
    mov_hist = np.sum(N[idxs_mov,:],axis=0)
    
    shaded_plot(ax2,y_centers,mov_hist,2,2)
    
    N,x_edges,y_edges = create_histogram(0.99*tof_corr,y_roi=[80,400],cts_per_slice=2**10,delta_y=0.0625)
    mov_hist = np.sum(N[idxs_mov,:],axis=0)
    
    shaded_plot(ax2,y_centers,ref_hist,3,3)
    shaded_plot(ax2,y_centers,mov_hist,1,1)
    
    
    N,x_edges,y_edges = create_histogram(1.0*tof_corr,y_roi=[80,400],cts_per_slice=2**10,delta_y=0.0625)
    mov_hist = np.sum(N[idxs_mov,:],axis=0)
    
    
    shaded_plot(ax2,y_centers,mov_hist,0,col_idx=0)

    
    cs = np.linspace(0.975, 1.005, 256)
    dp = np.zeros_like(cs)
    for idx, c in enumerate(cs):
        N,x_edges,y_edges = create_histogram(c*tof_corr,y_roi=[80,400],cts_per_slice=2**10,delta_y=0.0625)
        mov_hist = np.sum(N[idxs_mov,:],axis=0)
        dp[idx] = np.sum((mov_hist/np.sum(mov_hist))*(ref_hist/np.sum(ref_hist)))
    
    ax1.set(xlim=[0.975, 1.005],ylim=[-0.1,1.1])
    
    f = scipy.interpolate.interp1d(cs,dp/np.max(dp))
    
    cols = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    xq = [0.98, 0.99017, 1.0]
    for idx in [0,1,2]:
        ax1.plot(xq[idx],f(xq[idx]),'o',markersize=14,color=cols[2-idx])
    
    
    ax1.plot(cs,dp/np.max(dp),'k')
    
    
    

    ax1.set_xlabel('correction factor, c')
    ax1.set_ylabel('dot product (norm)')

    ax2.set_xlabel('corrected time of flight (ns)')
    ax2.set_ylabel('counts')
    plt.pause(0.1)
    fig.tight_layout()
    
    fig.savefig(r'Q:\users\bwc\APT\scale_corr_paper\correction_idea.pdf', format='pdf', dpi=600)
    
    return 0

def sio2_R45_corr():
    
    
    fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\R45_data\R45_04472-v03.epos"
    epos = apt_fileio.read_epos_numpy(fn)
    epos = epos[25000:]
    epos = epos[:400000]

    # Voltage and bowl correct ToF data
    p_volt = np.array([])
    p_bowl = np.array([])
    t_i = time.time()
    tof_corr, p_volt, p_bowl = do_voltage_and_bowl(epos,p_volt,p_bowl)
    print("time to voltage and bowl correct:    "+str(time.time()-t_i)+" seconds")
    
    
#    fake_tof = np.sqrt((296/312)*epos['m2q']/1.393e-4)
        
    m2q_roi=[0.8,80]
    
    m2q_to_tof = 613/np.sqrt(59)
    
    tof_roi = [m2q_roi[0]*m2q_to_tof, m2q_roi[1]*m2q_to_tof]
    
    
    cts_per_slice=2**7
    #m2q_roi = [0.9,190]
#    tof_roi = [0, 1000]

    t_start = time.time()
    pointwise_scales,piecewise_scales = scaling_correction.get_all_scale_coeffs(tof_corr,
                                                             m2q_roi=tof_roi,
                                                             cts_per_slice=cts_per_slice,
                                                             max_scale=1.075)
    t_end = time.time()
    print('Total Time = ',t_end-t_start)
    
#    fake_tof_corr = fake_tof/np.sqrt(pointwise_scales)
    q_tof_corr = tof_corr/pointwise_scales
    
#    m2q_corr = epos['m2q']/pointwise_scales
    
    # Plot histogram for sio2
    fig = plt.figure(figsize=(FIGURE_SCALE_FACTOR*3.14961,FIGURE_SCALE_FACTOR*3.14961),num=6,dpi=100)
    plt.clf()
    ax1, ax2 = fig.subplots(2,1,sharex=True)
    
    
    N,x_edges,y_edges = create_histogram(tof_corr,y_roi=[280,360],cts_per_slice=2**9,delta_y=.25)
    im = plot_2d_histo(ax1,N,x_edges,y_edges)    
    plt.colorbar(im)

    ax1.set(ylabel='flight time (ns)')
    
    
    ax1twin = ax1.twinx()
    ax1twin.plot(pointwise_scales,'-', 
            linewidth=1,
            color=mcd.XKCD_COLORS['xkcd:white'])
    ax1twin.set(ylabel='correction factor, c',ylim=[0.98, 1.2],xlim=[0, 400000])
    
    
    N,x_edges,y_edges = create_histogram(q_tof_corr,y_roi=[280,360],cts_per_slice=2**9,delta_y=.25)
    im = plot_2d_histo(ax2,N,x_edges,y_edges)    
    plt.colorbar(im)

    ax2.set(xlabel='ion sequence',ylabel='corrected flight time (ns)')
    
    
    fig.tight_layout()
    fig.savefig(r'Q:\users\bwc\APT\scale_corr_paper\SiO2_NUV_corrected.pdf', format='pdf', dpi=600)
        
    return 0

def sio2_R45_histo():
        
    def shaded_plot(ax,x,y,idx,col_idx=None,min_val=None):
        if col_idx is None:
            col_idx = idx
            
        if min_val is None:
            min_val = np.min(y)
        
        sc = 150
        cols = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
        xlim = ax.get_xlim()
        
        idxs = np.nonzero((x>=xlim[0]) & (x<=xlim[1]))
    
        ax.fill_between(x[idxs], y[idxs], min_val, color=cols[col_idx],linestyle='None',lw=0)
    #    ax.plot(x,y+idx*sc, color='k')
        return

    
    fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\R45_data\R45_04472-v03.epos"
    epos = apt_fileio.read_epos_numpy(fn)
    epos = epos[25000:]
    epos = epos[:400000]

    # Voltage and bowl correct ToF data
    p_volt = np.array([])
    p_bowl = np.array([])
    t_i = time.time()
    tof_corr, p_volt, p_bowl = do_voltage_and_bowl(epos,p_volt,p_bowl)
    print("time to voltage and bowl correct:    "+str(time.time()-t_i)+" seconds")
    
    
#    fake_tof = np.sqrt((296/312)*epos['m2q']/1.393e-4)
        
    m2q_roi=[0.8,80]
    
    m2q_to_tof = 613/np.sqrt(59)
    
    tof_roi = [m2q_roi[0]*m2q_to_tof, m2q_roi[1]*m2q_to_tof]
    
    
    cts_per_slice=2**7
    #m2q_roi = [0.9,190]
#    tof_roi = [0, 1000]

    t_start = time.time()
    pointwise_scales,piecewise_scales = scaling_correction.get_all_scale_coeffs(tof_corr,
                                                             m2q_roi=tof_roi,
                                                             cts_per_slice=cts_per_slice,
                                                             max_scale=1.075)
    t_end = time.time()
    print('Total Time = ',t_end-t_start)
    
#    fake_tof_corr = fake_tof/np.sqrt(pointwise_scales)
    q_tof_corr = tof_corr/pointwise_scales
    
    OVERALL_CALIB_FACTOR = 1.0047693561704287
    
    m2q_corr = OVERALL_CALIB_FACTOR*m2q_to_tof**-2*q_tof_corr**2
    m2q_vbcorr = OVERALL_CALIB_FACTOR*m2q_to_tof**-2*tof_corr**2
    
    
    
    
    fig = plt.figure(constrained_layout=True,figsize=(FIGURE_SCALE_FACTOR*3.14961,FIGURE_SCALE_FACTOR*3.14961),num=7,dpi=100)
    plt.clf()
    
    gs = plt.GridSpec(2, 3, figure=fig)
    ax0 = fig.add_subplot(gs[0, :])
    # identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
    ax1 = fig.add_subplot(gs[1,0:2])
    #ax2 = fig.add_subplot(gs[1,1])
    ax3 = fig.add_subplot(gs[1,2])
    
    
    
    
    
    
    dat = m2q_vbcorr
    user_bin_width = 0.02
    user_xlim = [0,65]
    ax0.set(xlim=user_xlim)
    
    
    
    dat = m2q_corr
    xs, ys = bin_dat(dat,isBinAligned=True,bin_width=user_bin_width,user_roi=user_xlim)
    shaded_plot(ax0,xs,100*(1+ys),1,min_val=100)
    
    
    dat = m2q_vbcorr
    xs, ys = bin_dat(dat,isBinAligned=True,bin_width=user_bin_width,user_roi=user_xlim)
    shaded_plot(ax0,xs,1+ys,0,min_val=1)
    
    
    
    
    
    
    ax0.set(xlabel='m/z (Da)', ylabel='counts', xlim=user_xlim)
    ax0.set_yscale('log')
    
    
    
    
#    user_bin_width = 0.02
    user_xlim = [13,19]
    ax1.set(xlim=user_xlim)
    
    
    
    
    dat = m2q_corr
    xs, ys = bin_dat(dat,isBinAligned=True,bin_width=user_bin_width,user_roi=user_xlim)
    shaded_plot(ax1,xs,100*(1+ys),1,min_val=100)
    
    
    dat = m2q_vbcorr
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
    
    
    dat = m2q_vbcorr
    xs, ys = bin_dat(dat,isBinAligned=True,bin_width=user_bin_width,user_roi=user_xlim)
    shaded_plot(ax3,xs,1+ys,0,min_val=1)
    
    
    ax3.set(xlabel='m/z (Da)', ylabel='counts', xlim=user_xlim)
    ax3.set_yscale('log')
    
    
    ax0.set(ylim=[1,None])
    ax1.set(ylim=[1,None])
#    ax2.set(ylim=[1,None])
    ax3.set(ylim=[1,None])
    
    fig.tight_layout()
    
    fig.savefig(r'Q:\users\bwc\APT\scale_corr_paper\SiO2_NUV_corrected_hist.pdf', format='pdf', dpi=600)
    
    
    
    

    
    return 0



def ceria_histo():
    
    
    def shaded_plot(ax,x,y,idx,col_idx=None,min_val=None):
        if col_idx is None:
            col_idx = idx
            
        if min_val is None:
            min_val = np.min(y)
        
        
        cols = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
        xlim = ax.get_xlim()
        
        idxs = np.nonzero((x>=xlim[0]) & (x<=xlim[1]))
    
        ax.fill_between(x[idxs], y[idxs], min_val, color=cols[col_idx],linestyle='None',lw=0)
    #    ax.plot(x,y+idx*sc, color='k')
        return
    
    
    fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\R45_data\R45_00504-v56.epos"
    epos = apt_fileio.read_epos_numpy(fn)
    red_epos = epos[100000::10]
    

    # Voltage and bowl correct ToF data
    p_volt = np.array([])
    p_bowl = np.array([])
    t_i = time.time()
    _, p_volt, p_bowl = do_voltage_and_bowl(red_epos,p_volt,p_bowl)
    print("time to voltage and bowl correct:    "+str(time.time()-t_i)+" seconds")
    
    
    tof_vcorr = voltage_and_bowl.mod_full_voltage_correction(p_volt,epos['tof'],epos['v_dc'])
    tof_corr = voltage_and_bowl.mod_geometric_bowl_correction(p_bowl,tof_vcorr,epos['x_det'],epos['y_det'])

    
    
        
    m2q_roi=[10,250]
    
    m2q_to_tof = 1025/np.sqrt(172)
    
    tof_roi = [m2q_roi[0]*m2q_to_tof, m2q_roi[1]*m2q_to_tof]
    tof_roi = [200, 1200]
    
    
    cts_per_slice=2**9
    #m2q_roi = [0.9,190]
#    tof_roi = [0, 1000]

    t_start = time.time()
    pointwise_scales,piecewise_scales = scaling_correction.get_all_scale_coeffs(tof_corr,
                                                             m2q_roi=tof_roi,
                                                             cts_per_slice=cts_per_slice,
                                                             max_scale=1.075)
    t_end = time.time()
    print('Total Time = ',t_end-t_start)
    
#    fake_tof_corr = fake_tof/np.sqrt(pointwise_scales)
    q_tof_corr = tof_corr/pointwise_scales
    
    OVERALL_CALIB_FACTOR = 0.9956265249773827
    
    m2q_corr = OVERALL_CALIB_FACTOR*m2q_to_tof**-2*q_tof_corr**2
    m2q_vbcorr = OVERALL_CALIB_FACTOR*m2q_to_tof**-2*tof_corr**2
    
    
    
    
    fig = plt.figure(constrained_layout=True,figsize=(FIGURE_SCALE_FACTOR*3.14961,FIGURE_SCALE_FACTOR*3.14961),num=8,dpi=100)
    plt.clf()
    
    gs = plt.GridSpec(2, 4, figure=fig)
    ax0 = fig.add_subplot(gs[0, :])
    # identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
    ax1 = fig.add_subplot(gs[1,0:2])
    #ax2 = fig.add_subplot(gs[1,1])
    ax3 = fig.add_subplot(gs[1,2:4])
    
    
    
    
    
    dat = m2q_vbcorr
    user_bin_width = 0.02
    user_xlim = [0,200]
    ax0.set(xlim=user_xlim)
    
    
    
    dat = m2q_corr
    xs, ys = bin_dat(dat,isBinAligned=True,bin_width=user_bin_width,user_roi=user_xlim)
    shaded_plot(ax0,xs,10*(1+ys),1,min_val=10)
    
    
    dat = m2q_vbcorr
    xs, ys = bin_dat(dat,isBinAligned=True,bin_width=user_bin_width,user_roi=user_xlim)
    shaded_plot(ax0,xs,1+ys,0,min_val=1)
    
    
    
    
    
    
    ax0.set(xlabel='m/z (Da)', ylabel='counts', xlim=user_xlim)
    ax0.set_yscale('log')
    ax0.set(ylim=[10,None])
    
    
    
#    user_bin_width = 0.02
    user_xlim = [75,85]
    ax1.set(xlim=user_xlim)
    
    
    
    
    dat = m2q_corr
    xs, ys = bin_dat(dat,isBinAligned=True,bin_width=user_bin_width,user_roi=user_xlim)
    shaded_plot(ax1,xs,10*(1+ys),1,min_val=10)
    
    
    dat = m2q_vbcorr
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
    
    
    
    
#    user_bin_width = 0.03
    user_xlim = [154,170]
    ax3.set(xlim=user_xlim)
    
    
    dat = m2q_corr
    xs, ys = bin_dat(dat,isBinAligned=True,bin_width=user_bin_width,user_roi=user_xlim)
    shaded_plot(ax3,xs,10*(1+ys),1,min_val=10)
    
    
    dat = m2q_vbcorr
    xs, ys = bin_dat(dat,isBinAligned=True,bin_width=user_bin_width,user_roi=user_xlim)
    shaded_plot(ax3,xs,1+ys,0,min_val=1)
    
    
    ax3.set(xlabel='m/z (Da)', ylabel='counts', xlim=user_xlim)
    ax3.set_yscale('log')
    ax3.set(ylim=[10,1e4])
    ax3.set_xticks(np.arange(154,170+1,4))
    
    
    
    fig.tight_layout()
    
    fig.savefig(r'Q:\users\bwc\APT\scale_corr_paper\Ceria_NUV_corrected_hist.pdf', format='pdf', dpi=600)
    

    
    
    return 0





 
def chi2_plot():
    

    def chi2(dat):
        n = dat.size
        f = np.sum(dat)    
        f_n = f/n    
        chi2 = np.sum(np.square(dat-f_n)/f_n)
        return chi2
    
    
    def get_vb_corr_sio2_tof():            
        fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\R45_data\R45_04472-v03.epos"
        epos = apt_fileio.read_epos_numpy(fn)
        epos = epos[25000:]
        epos = epos[:400000]
    
        # Voltage and bowl correct ToF data
        p_volt = np.array([])
        p_bowl = np.array([])
        t_i = time.time()
        tof_corr, p_volt, p_bowl = do_voltage_and_bowl(epos,p_volt,p_bowl)
        return tof_corr
    
    def get_vb_corr_ceria_tof():
            
        fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\R45_data\R45_00504-v56.epos"
        epos = apt_fileio.read_epos_numpy(fn)
        red_epos = epos[100000::10]        
    
        # Voltage and bowl correct ToF data
        p_volt = np.array([])
        p_bowl = np.array([])
        t_i = time.time()
        _, p_volt, p_bowl = do_voltage_and_bowl(red_epos,p_volt,p_bowl)
        print("time to voltage and bowl correct:    "+str(time.time()-t_i)+" seconds")        
        
        tof_vcorr = voltage_and_bowl.mod_full_voltage_correction(p_volt,epos['tof'],epos['v_dc'])
        tof_corr = voltage_and_bowl.mod_geometric_bowl_correction(p_bowl,tof_vcorr,epos['x_det'],epos['y_det'])
            
        return tof_corr[0:1000000]
    
    
    def do_chi2(tof,tof_roi,N_lower,N_upper):
                    
        tof1 = tof[0::2]
        tof2 = tof[1::2]
        
        N = N_upper-N_lower+1
        slicings = np.logspace(N_lower,N_upper,N,base=2)
        
        opt_res = np.zeros(N)
        time_res = np.zeros(N)
    
        for idx,cts_per_slice in enumerate(slicings):
            
            
            t_start = time.time()
            pointwise_scales,piecewise_scales = scaling_correction.get_all_scale_coeffs(tof1,
                                                                     m2q_roi=tof_roi ,
                                                                     cts_per_slice=cts_per_slice,
                                                                     max_scale=1.075,
                                                                     delta_ly=2e-4)
            t_end = time.time()
            time_res[idx] = t_end-t_start
            print('Total Time = ',time_res[idx])
                        
            # Compute corrected data
            tof_corr = tof2/pointwise_scales
                
            _, ys = bin_dat(tof_corr,isBinAligned=True,bin_width=0.1,user_roi=tof_roi)
        
            opt_res[idx] = chi2(ys)
            print(opt_res[idx])
            
        print(slicings)
        print(opt_res/np.max(opt_res))
        print(time_res)
        
        return (slicings,opt_res/np.max(opt_res))
    
    
    sio2_tof = get_vb_corr_sio2_tof()
    ceo2_tof = get_vb_corr_ceria_tof()
    
    N_sio2,chi2_sio2 = do_chi2(sio2_tof,[50,1000],4,16)
    N_ceo2,chi2_ceo2 = do_chi2(ceo2_tof,[50,1200],4,16)
    
    
    
    fig = plt.figure(num=9,figsize=(FIGURE_SCALE_FACTOR*3.14961,FIGURE_SCALE_FACTOR*3.14961*0.7))
    fig.clear()
    ax = fig.gca()

    ax.plot(N_sio2,chi2_sio2,'s-', 
            markersize=8,label='SiO2')

    ax.plot(N_ceo2,chi2_ceo2,'o-', 
            markersize=8,label='ceria')
    ax.set(xlabel='N (events per chunk)', ylabel='$\chi^2$ statistic (normalized)')
    ax.set_xscale('log')    

    
    ax.legend()
       
    ax.set_xlim(10,1e5)
    ax.set_ylim(0.15, 1.05)


    fig.tight_layout()
    
    fig.savefig(r'Q:\users\bwc\APT\scale_corr_paper\optimal_N.pdf', format='pdf', dpi=600)

    
    
    return 0

from constrNMPy import constrNM


def wid_plot():

    
    def get_vb_corr_sio2_tof():            
        fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\R45_data\R45_04472-v03.epos"
        epos = apt_fileio.read_epos_numpy(fn)
        epos = epos[25000:]
        epos = epos[:400000]
    
        # Voltage and bowl correct ToF data
        p_volt = np.array([])
        p_bowl = np.array([])
        t_i = time.time()
        tof_corr, p_volt, p_bowl = do_voltage_and_bowl(epos,p_volt,p_bowl)
        return tof_corr
    
    def get_vb_corr_ceria_tof():
            
        fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\R45_data\R45_00504-v56.epos"
        epos = apt_fileio.read_epos_numpy(fn)
        red_epos = epos[100000::10]        
    
        # Voltage and bowl correct ToF data
        p_volt = np.array([])
        p_bowl = np.array([])
        t_i = time.time()
        _, p_volt, p_bowl = do_voltage_and_bowl(red_epos,p_volt,p_bowl)
        print("time to voltage and bowl correct:    "+str(time.time()-t_i)+" seconds")        
        
        tof_vcorr = voltage_and_bowl.mod_full_voltage_correction(p_volt,epos['tof'],epos['v_dc'])
        tof_corr = voltage_and_bowl.mod_geometric_bowl_correction(p_bowl,tof_vcorr,epos['x_det'],epos['y_det'])
            
        return tof_corr[0:1000000]
    
    
#    def do_wid(tof,tof_roi,N_lower,N_upper,t_guess0,t_width):
#                    
#        tof1 = tof[0::2]
#        tof2 = tof[1::2]
#        
#        N = N_upper-N_lower+1
#        slicings = np.logspace(N_lower,N_upper,N,base=2)
#        
#        opt_res = np.zeros(N)
#        time_res = np.zeros(N)
#    
#        curr_res = None
#        for idx,cts_per_slice in enumerate(slicings):
#            prev_res = curr_res
#            
#            t_start = time.time()
#            pointwise_scales,piecewise_scales = scaling_correction.get_all_scale_coeffs(tof1,
#                                                                     m2q_roi=tof_roi ,
#                                                                     cts_per_slice=cts_per_slice,
#                                                                     max_scale=1.075,
#                                                                     delta_ly=2e-4)
#            t_end = time.time()
#            time_res[idx] = t_end-t_start
#            print('Total Time = ',time_res[idx])
#                        
#            # Compute corrected data
#            tof_corr = tof2/pointwise_scales
#                
#            if prev_res is None:
#                t_guess = t_guess0
#            else:
#                t_guess = prev_res[1]    
#            
#            curr_res = fit_to_g_off(tof_corr,t_guess,t_width)
#            
#            opt_res[idx] = curr_res[1]/curr_res[2]
#            
#            print(curr_res)
#            
##            print(opt_res[idx])
#            
#        print(slicings)
#        print(opt_res)
#        print(time_res)
#        
#        return (slicings,opt_res)
#    
#    def b_G(x,sigma,x0):
#        return np.exp(-np.square(x-x0)/(2*np.square(sigma)))
#
#    def pk_mod_fun(x,amp,x0,sigma,b):
#        return amp*b_G(x,sigma,x0)+b
#            
#    
#    
#    def fit_to_g_off(dat,t_guess,t_width):
#
#        t_ms = ppd.mean_shift_peak_location(dat,user_std=1,user_x0=t_guess)
#             
#        xs,ys = bin_dat(dat,bin_width=0.01,user_roi=[t_ms-t_width/2, t_ms+t_width/2],isBinAligned=False,isDensity=False)              
#        
#        opt_fun = lambda p: np.sum(np.square(pk_mod_fun(xs, *p)-ys))
#        
#        N4 = ys.size//4
#        mx_idx = np.argmax(ys[N4:(3*N4)])+N4
#        
#        p0 = np.array([ys[mx_idx]-np.min(ys), xs[mx_idx], 0.015,  np.percentile(ys,20)])
#            
#        # b_model2(x,amp_g,x0,sigma,b):
#        lbs = np.array([0,       np.percentile(xs,10),  0.001,   0])
#        ubs = np.array([2*p0[0],  np.percentile(xs,90), 5,   p0[0]])
#    
#        # Force in bounds
#        p_guess = np.sort(np.c_[lbs,p0,ubs])[:,1]
#        
#        
#        ret_dict = constrNM(opt_fun,p_guess,lbs,ubs,xtol=1e-5, ftol=1e-12, maxiter=2**14, maxfun=2**14, full_output=1, disp=1)
#        
#        fig = plt.figure(num=89,figsize=(FIGURE_SCALE_FACTOR*3.14961,FIGURE_SCALE_FACTOR*3.14961*0.7))
#        fig.clear()
#        ax = fig.gca()
#        
#        ax.plot(xs,ys,'-')
#        ax.plot(xs,pk_mod_fun(xs, *ret_dict['xopt']),'-')
#        
#        plt.pause(2)
#        return np.abs(ret_dict['xopt'])

    def get_wid(tof,t_guess,t_width):
        t_ms = ppd.mean_shift_peak_location(tof,user_std=1,user_x0=t_guess)             
        xs,ys = bin_dat(tof,bin_width=0.1,user_roi=[t_ms-t_width/2, t_ms+t_width/2],isBinAligned=False,isDensity=False)                      
        pk_idx = np.argmin(np.abs(t_ms-xs))
        pk_val = ys[pk_idx]


        fig = plt.figure(num=89,figsize=(FIGURE_SCALE_FACTOR*3.14961,FIGURE_SCALE_FACTOR*3.14961*0.7))
        fig.clear()
        ax = fig.gca()
        
        ax.plot(xs,ys,'-')
        
        plt.pause(2)


        rhs = 1e6
        for idx in np.arange(pk_idx,ys.size):
            if ys[idx]<0.5*pk_val:
                # compute 
                x1 = xs[idx-1]
                x2 = xs[idx]
                y1 = ys[idx-1]
                y2 = ys[idx]
                m = (y2-y1)/(x2-x1)
                b = y1-m*x1
                rhs = (0.5*pk_val-b)/m
                break
            
        
        lhs = -1e6
        for idx in np.arange(pk_idx,0,-1):
            if ys[idx]<0.5*pk_val:
                # compute 
                x1 = xs[idx+1]
                x2 = xs[idx]
                y1 = ys[idx+1]
                y2 = ys[idx]
                m = (y2-y1)/(x2-x1)
                b = y1-m*x1
                lhs = (0.5*pk_val-b)/m
                break
        
        wid = rhs-lhs
        
        return wid

    def do_wid(tof,tof_roi,N_lower,N_upper,t_guess,t_width):
                    
        tof1 = tof[0::2]
        tof2 = tof[1::2]
        
        N = N_upper-N_lower+1
        slicings = np.logspace(N_lower,N_upper,N,base=2)
        
        opt_res = np.zeros(N)
        time_res = np.zeros(N)
    
        curr_res = None
        for idx,cts_per_slice in enumerate(slicings):
            prev_res = curr_res
            
            t_start = time.time()
            pointwise_scales,piecewise_scales = scaling_correction.get_all_scale_coeffs(tof1,
                                                                     m2q_roi=tof_roi ,
                                                                     cts_per_slice=cts_per_slice,
                                                                     max_scale=1.075,
                                                                     delta_ly=2e-4)
            t_end = time.time()
            time_res[idx] = t_end-t_start
            print('Total Time = ',time_res[idx])
                        
            # Compute corrected data
            tof_corr = tof2/pointwise_scales
                
   
            
#            curr_res = fit_to_g_off(tof_corr,t_guess,t_width)
            
#            opt_res[idx] = curr_res[1]/curr_res[2]
            
            curr_res = get_wid(tof_corr,t_guess,t_width)
            
            print(curr_res)
            opt_res[idx] = 1/curr_res
            
#            print(opt_res[idx])
            
        print(slicings)
        print(opt_res)
        print(time_res)
        
        return (slicings,opt_res)
    
    
    sio2_tof = get_vb_corr_sio2_tof()
    ceo2_tof = get_vb_corr_ceria_tof()
    
    N_sio2,mrp_sio2 = do_wid(sio2_tof,[50,1000],4,16,318.25,10)
    N_ceo2,mrp_ceo2 = do_wid(ceo2_tof,[50,1200],4,16,692,10)
    
#    
#    mrp_sio2 = 1/chi2_sio2
#    mrp_ceo2 = 1/chi2_ceo2
#    
    
    fig = plt.figure(num=9,figsize=(FIGURE_SCALE_FACTOR*3.14961,FIGURE_SCALE_FACTOR*3.14961*0.7))
    fig.clear()
    ax = fig.gca()

    ax.plot(N_sio2,mrp_sio2/np.max(mrp_sio2),'s-', 
            markersize=8,label='SiO2')

    ax.plot(N_ceo2,mrp_ceo2/np.max(mrp_ceo2),'o-', 
            markersize=8,label='ceria')
    ax.set(xlabel='N (events per chunk)', ylabel='$\chi^2$ statistic (normalized)')
    ax.set_xscale('log')    

    
    ax.legend()
       
    ax.set_xlim(10,1e5)
    ax.set_ylim(0.15, 1.05)


    fig.tight_layout()
    
    fig.savefig(r'Q:\users\bwc\APT\scale_corr_paper\optimal_N_wid.pdf', format='pdf', dpi=600)

    
    
    return 0





plt.close('all')
#
#steel()
#
##plt.close('all')
#sio2_R45()
#
##plt.close('all')
#sio2_R20()
#
##plt.close('all')
#corr_idea()
#
##plt.close('all')
#sio2_R45_corr()
#
##plt.close('all')
#sio2_R44()
#
##plt.close('all')
#sio2_R45_histo()
#
##plt.close('all')
#ceria_histo()

#plt.close('all')
#chi2_plot()
    
plt.close('all')
wid_plot()










