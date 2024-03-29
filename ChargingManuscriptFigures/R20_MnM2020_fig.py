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
    
    ax.set_xticks([0,100000,200000,300000,400000])
#    ax.set_xticklabels(["\n".join(x) for x in data.index])
    
    return pos1
def sio2_R20():
    
    fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\R20_07080-v01.epos"
    epos = apt_fileio.read_epos_numpy(fn)
#    epos = epos[25000:]
    epos = epos[:400000]
    
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
    
    fig.savefig(r'SiO2_EUV_wandering.pdf', format='pdf', dpi=600)

    
    return 0





def sio2_R20_corr():
    
    
    fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\R20_07080-v01.epos"
    epos = apt_fileio.read_epos_numpy(fn)
#    epos = epos[25000:]
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
    
#    m2q_corr = epos['m2q']/pointwise_scales
    
    # Plot histogram for sio2
    fig = plt.figure(figsize=(FIGURE_SCALE_FACTOR*3.14961,FIGURE_SCALE_FACTOR*3.14961),num=6,dpi=100)
    plt.clf()
    ax1, ax2 = fig.subplots(2,1,sharex=True)
    
    
    N,x_edges,y_edges = create_histogram(tof_corr,y_roi=[320,380],cts_per_slice=2**9,delta_y=.25)
    im = plot_2d_histo(ax1,N,x_edges,y_edges)    
#    plt.colorbar(im)

    ax1.set(ylabel='flight time (ns)')
    
    
    ax1twin = ax1.twinx()
    ax1twin.plot(pointwise_scales,'-', 
            linewidth=1,
            color=mcd.XKCD_COLORS['xkcd:white'])
    ax1twin.set(ylabel='correction factor, c',ylim=[0.98, 1.2],xlim=[0, 400000])
    
    
    N,x_edges,y_edges = create_histogram(q_tof_corr,y_roi=[320,380],cts_per_slice=2**9,delta_y=.25)
    im = plot_2d_histo(ax2,N,x_edges,y_edges)    
#    plt.colorbar(im)

    ax2.set(xlabel='ion sequence',ylabel='corrected flight time (ns)')
    
#    ax2twin = ax2.twinx()
#    ax2twin.plot(pointwise_scales,'-', 
#            linewidth=1,
#            color=mcd.XKCD_COLORS['xkcd:white'])
#    ax2twin.set(ylabel='correction factor, c',ylim=[0.90, 1.04],xlim=[0, 400000])
    
    fig.tight_layout()
    fig.savefig(r'SiO2_EUV_corrected.pdf', format='pdf', dpi=600)
        
    return 0

def sio2_R20_histo():
        
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

    
    fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\R20_07080-v01.epos"
    epos = apt_fileio.read_epos_numpy(fn)
#    epos = epos[25000:]
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
    
    c = 1.2654143608271198e-4
    t0 = -0.09660555
    
    m2q_corr = c*(q_tof_corr-t0)**2
    m2q_vbcorr = c*(tof_corr-t0)**2
    
    
    
    
    fig = plt.figure(constrained_layout=True,figsize=(FIGURE_SCALE_FACTOR*3.14961,FIGURE_SCALE_FACTOR*3.14961),num=7,dpi=100)
    plt.clf()
    
    gs = plt.GridSpec(2, 3, figure=fig)
    ax0 = fig.add_subplot(gs[0, :])
    # identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
    ax1 = fig.add_subplot(gs[1,:])
    #ax2 = fig.add_subplot(gs[1,1])
#    ax3 = fig.add_subplot(gs[1,2])
    
    
    
    
    
    
    dat = m2q_vbcorr
    user_bin_width = 0.03
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
    
    
    
    
    user_bin_width = 0.01
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
#    user_xlim = [58,64]
#    ax3.set(xlim=user_xlim)
    
    
#    dat = m2q_corr
#    xs, ys = bin_dat(dat,isBinAligned=True,bin_width=user_bin_width,user_roi=user_xlim)
#    shaded_plot(ax3,xs,100*(1+ys),1,min_val=100)
    
    
#    dat = m2q_vbcorr
#    xs, ys = bin_dat(dat,isBinAligned=True,bin_width=user_bin_width,user_roi=user_xlim)
#    shaded_plot(ax3,xs,1+ys,0,min_val=1)
    
    
#    ax3.set(xlabel='m/z (Da)', ylabel='counts', xlim=user_xlim)
#    ax3.set_yscale('log')
    
    
    ax0.set(ylim=[1,None])
    ax1.set(ylim=[1,None])
#    ax2.set(ylim=[1,None])
#    ax3.set(ylim=[1,None])
    
    fig.tight_layout()
    
    fig.savefig(r'SiO2_EUV_corrected_hist.pdf', format='pdf', dpi=600)
    
    
    
    

    
    return 0






 
def chi2_plot():
    

    def chi2(dat):
        n = dat.size
        f = np.sum(dat)    
        f_n = f/n    
        chi2 = np.sum(np.square(dat-f_n)/f_n)
        return chi2
    
    
    def get_vb_corr_sio2_tof():            
        fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\R20_07080-v01.epos"
        epos = apt_fileio.read_epos_numpy(fn)
        #    epos = epos[25000:]
        epos = epos[:400000]
    
        # Voltage and bowl correct ToF data
        p_volt = np.array([])
        p_bowl = np.array([])
        t_i = time.time()
        tof_corr, p_volt, p_bowl = do_voltage_and_bowl(epos,p_volt,p_bowl)
        return tof_corr
    
  
    
    
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
    
    N_sio2,chi2_sio2 = do_chi2(sio2_tof,[50,1000],4,16)
    
    
    
    fig = plt.figure(num=9,figsize=(FIGURE_SCALE_FACTOR*3.14961,FIGURE_SCALE_FACTOR*3.14961*0.7))
    fig.clear()
    ax = fig.gca()

    ax.plot(N_sio2,chi2_sio2,'s-', 
            markersize=8,label='SiO2')

   
    ax.set(xlabel='N (events per chunk)', ylabel='$\chi^2$ statistic (normalized)')
    ax.set_xscale('log')    

    
    ax.legend()
       
    ax.set_xlim(10,1e5)
    ax.set_ylim(0.15, 1.05)


    fig.tight_layout()
    
    fig.savefig(r'EUV_optimal_N.pdf', format='pdf', dpi=600)

    
    
    return 0


plt.close('all')

sio2_R20()
sio2_R20_corr()
sio2_R20_histo()
chi2_plot()
    















