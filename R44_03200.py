# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 16:57:40 2019

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

import sel_align_m2q_log_xcorr_v2

import scipy.interpolate
import time
import m2q_calib
import initElements_P3

from voltage_and_bowl import do_voltage_and_bowl
import voltage_and_bowl 

import colorcet as cc
import matplotlib._color_data as mcd

import pandas as pd



def extents(f):
  delta = f[1] - f[0]
  return [f[0] - delta/2, f[-1] + delta/2]

def create_histogram(ys,cts_per_slice=2**10,y_roi=None,delta_y=1e-4):
    
    

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




# 

fn = r"C:\Users\bwc\Documents\NetBeansProjects\R44_03200\recons\recon-v02\default\R44_03200-v02.epos"
epos = apt_fileio.read_epos_numpy(fn)
#epos = epos[:]



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
fig = plt.figure(figsize=(2*3.14961,2*3.14961),num=45321,dpi=100)
plt.clf()
ax1, ax2 = fig.subplots(2,1,sharex=True)

roi = [1400000,1800000]

N,x_edges,y_edges = create_histogram(tof_bcorr[roi[0]:roi[1]],y_roi=[300,310],cts_per_slice=2**7,delta_y=.2)
ax1.imshow(np.log10(1+1*np.transpose(N)), aspect='auto', 
           extent=extents(x_edges) + extents(y_edges), origin='lower', cmap=cc.cm.CET_L8,
           interpolation='bilinear')

ax1.set(ylabel='flight time (ns)')


ax1twin = ax1.twinx()
ax1twin.plot(epos['v_dc'][roi[0]:roi[1]],'-', 
        linewidth=2,
        color=mcd.XKCD_COLORS['xkcd:white'])
ax1twin.set(ylabel='applied voltage (volts)',ylim=[0000, 7000],xlim=[0,None])


N,x_edges,y_edges = create_histogram(tof_corr[roi[0]:roi[1]],y_roi=[300,310],cts_per_slice=2**7,delta_y=0.2)
ax2.imshow(np.log10(1+1*np.transpose(N)), aspect='auto', 
           extent=extents(x_edges) + extents(y_edges), origin='lower', cmap=cc.cm.CET_L8,
           interpolation='bilinear')

ax2.set(xlabel='ion sequence',ylabel='corrected flight time (ns)')

ax2.set_xlim(0,roi[1]-roi[0])


fig.tight_layout()
#fig.savefig(r'Q:\users\bwc\APT\scale_corr_paper\SiO2_NUV_wandering.svg', format='svg', dpi=600)















































plt.close('all')

fn = r"C:\Users\bwc\Documents\NetBeansProjects\R44_03200\recons\recon-v02\default\R44_03200-v02.epos"

epos = apt_fileio.read_epos_numpy(fn)
epos_red = epos[0::4]


# Voltage and bowl correct ToF data
p_volt = np.array([])
p_bowl = np.array([])
t_i = time.time()
tof_corr, p_volt, p_bowl = do_voltage_and_bowl(epos_red,p_volt,p_bowl)
print("time to voltage and bowl correct:    "+str(time.time()-t_i)+" seconds")

# Only apply bowl correction
tof_bcorr = voltage_and_bowl.mod_geometric_bowl_correction(p_bowl,epos['tof'],epos['x_det'],epos['y_det'])
tof_corr = voltage_and_bowl.mod_full_voltage_correction(p_volt,tof_bcorr,epos['v_dc'])




laser_df = pd.read_csv(r'C:\Users\bwc\Documents\NetBeansProjects\R44_03200\R44_03200_LaserPositionHist.csv')
power_df = pd.read_csv(r'C:\Users\bwc\Documents\NetBeansProjects\R44_03200\R44_03200_LaserPowerHist.csv')









fig = plt.figure(num=111)
fig.clear()
ax = fig.gca()


new_x = np.arange(epos.size)*(1850666/(epos.size-1))

#ax.plot(tof_df['Ion Sequence Number'],tof_df['TOF (ns)'],'.', 
ax.plot(new_x,tof_corr,'.', 
        markersize=1,
        marker='.',
        markeredgecolor='#1f77b4aa')
ax.set(xlabel='event index', ylabel='ToF (ns)', ylim=[0,1200])

ax.grid()

ax2 = ax.twinx()

new_x = laser_df['Ion Sequence Number']*(epos.size-1)/laser_df['Ion Sequence Number'].max()
X = laser_df['Laser X Position (mic)'].values
Y = laser_df['Laser Y Position (mic)'].values
#
#ax2.plot(new_x,X,'-', 
#        markersize=5,
##        markeredgecolor='#1f77b4aa',
#        color='tab:orange',
#        label='X')
#ax2.plot(new_x,Y,'-', 
#        markersize=5,
##        markeredgecolor='#1f77b4aa',
#        color='tab:red',
#        label='Y')



ax2.plot(new_x,power_df['Laser power (nJ)'],'-', 
        markersize=5,
#        markeredgecolor='#1f77b4aa',
        color='tab:red',
        label='Y')

ax.set_xlim(0,epos.size)

ax.set_ylim(284,290)


fig.tight_layout()





cts_per_slice=2**10
import time
t_start = time.time()
pointwise_scales,piecewise_scales = sel_align_m2q_log_xcorr_v2.get_all_scale_coeffs(epos['m2q'],
                                                         m2q_roi=[0.8,80],
                                                         cts_per_slice=cts_per_slice,
                                                         max_scale=1.5)
t_end = time.time()
print('Total Time = ',t_end-t_start)






fig = plt.figure(num=111)
fig.clear()
ax = fig.gca()


new_x = np.arange(epos.size)*(1850666/(epos.size-1))

#ax.plot(tof_df['Ion Sequence Number'],tof_df['TOF (ns)'],'.', 
ax.plot(new_x,pointwise_scales)


ax.grid()

ax2 = ax.twinx()

#new_x = laser_df['Ion Sequence Number']*(epos.size-1)/laser_df['Ion Sequence Number'].max()
#X = laser_df['Laser X Position (mic)'].values
#Y = laser_df['Laser Y Position (mic)'].values
#
#ax2.plot(new_x,X,'-', 
#        markersize=5,
##        markeredgecolor='#1f77b4aa',
#        color='tab:orange',
#        label='X')
#ax2.plot(new_x,Y,'-', 
#        markersize=5,
##        markeredgecolor='#1f77b4aa',
#        color='tab:red',
#        label='Y')


new_x = power_df['Ion Sequence Number']*(epos.size-1)/power_df['Ion Sequence Number'].max()

ax2.plot(power_df['Ion Sequence Number'],power_df['Laser power (nJ)'].values,'-', 
        markersize=5,
#        markeredgecolor='#1f77b4aa',
        color='tab:red',
        label='Y')

ax.set_xlim(0,epos.size)

ax.set_ylim(0.99,1.02)


fig.tight_layout()







#wall_time = np.cumsum(np.int64(epos['pslep']))/100000.0/60/60


#fig = plt.figure(num=111)
#fig.clear()
#ax = fig.gca()
#
#event_idx = np.arange(0,epos.size)    
#ax.plot(event_idx,tof_bcorr,'.', 
#        markersize=.1,
#        marker=',',
#        markeredgecolor='#1f77b4aa')
#ax.set(xlabel='event index', ylabel='ToF (ns)', ylim=[0,1200])
#
#ax.grid()
#
#ax2 = ax.twinx()
#
#
#df = pd.read_csv(r'C:\Users\bwc\Documents\NetBeansProjects\R44_03115\R44_03115_LaserPositionHist.csv')
#
#new_x = np.linspace(0,event_idx[-1],df.shape[0])
#
#ax2.plot(new_x,df['Laser X Position (mic)'],'o', 
#        markersize=5,
#        marker='o',
##        markeredgecolor='#1f77b4aa',
#        color='tab:orange',
#        label='X')
#ax2.plot(new_x,df['Laser Y Position (mic)'],'o', 
#        markersize=5,
#        marker='o',
##        markeredgecolor='#1f77b4aa',
#        color='tab:red',
#        label='Y')
#
#ax2.legend()
#
#ax2.set(ylabel='position')
#
#
#df = pd.read_csv(r'C:\Users\bwc\Documents\NetBeansProjects\R44_03115\R44_03115_PressureHist.csv')
#new_x = np.linspace(0,epos.size-1,df.shape[0])
#
#
#ax2.plot(new_x,df['Pressure (Torr)'],'-', 
#        markersize=.1,
#        marker=',',
#        markeredgecolor='#1f77b4aa',
#        color='tab:orange',
#        label='X')
#
#
#ax2.legend()
#
#ax2.set(ylabel='pressure')
#











#
#
#fig.tight_layout()
#
#ax.set_ylim(310,380)

#
#
#fig = plt.figure(num=222)
#fig.clear()
#ax = fig.gca()
#ax.plot(event_idx,wall_time)
#
#
#
#laser_df = pd.read_csv(r'C:\Users\bwc\Documents\NetBeansProjects\R44_03115\R44_03115_LaserPositionHist.csv')
#freq_df = pd.read_csv(r'C:\Users\bwc\Documents\NetBeansProjects\R44_03115\R44_03115_FreqHist.csv')
#tof_df = pd.read_csv(r'C:\Users\bwc\Documents\NetBeansProjects\R44_03115\R44_03115_TOFHist.csv')
#pres_df = pd.read_csv(r'C:\Users\bwc\Documents\NetBeansProjects\R44_03115\R44_03115_PressureHist.csv')
#


R44_03200_LaserPowerHist.csv
R44_03200_LaserPositionHist.csv


fig = plt.figure(num=111)
fig.clear()
ax = fig.gca()

#
start_idx = 0

new_x = np.arange(epos.size)*(1850666/(epos.size-1))

#ax.plot(tof_df['Ion Sequence Number'],tof_df['TOF (ns)'],'.', 
ax.plot(new_x,tof_corr,'.', 
        markersize=.1,
        marker=',',
        markeredgecolor='#1f77b4aa')
ax.set(xlabel='event index', ylabel='ToF (ns)', ylim=[0,1200])

ax.grid()

ax2 = ax.twinx()

new_x = laser_df['Ion Sequence Number']*(epos.size-1)/laser_df['Ion Sequence Number'].max()
X = laser_df['Laser X Position (mic)'].values
Y = laser_df['Laser Y Position (mic)'].values

ax2.plot(new_x,X,'-', 
        markersize=5,
#        markeredgecolor='#1f77b4aa',
        color='tab:orange',
        label='X')
ax2.plot(new_x,Y,'-', 
        markersize=5,
#        markeredgecolor='#1f77b4aa',
        color='tab:red',
        label='Y')

ax.set_xlim(start_idx,epos.size)

ax.set_ylim(320,340)


fig.tight_layout()





fig = plt.figure(num=111)
fig.clear()
ax = fig.gca()

ax.plot(new_x,np.sqrt(pointwise_scales))


start_idx = 330000


cts_per_slice=2**9
import time
t_start = time.time()
pointwise_scales,piecewise_scales = sel_align_m2q_log_xcorr_v2.get_all_scale_coeffs(epos['m2q'],
                                                         m2q_roi=[0.8,80],
                                                         cts_per_slice=cts_per_slice,
                                                         max_scale=1.5)
t_end = time.time()
print('Total Time = ',t_end-t_start)

tof_corr = tof_bcorr/np.sqrt(pointwise_scales)



#fake_tof = np.sqrt((296/312)*m2q_corr/1.393e-4)

#ax.plot(tof_df['Ion Sequence Number'],tof_df['TOF (ns)'],'.', 
ax.plot(np.arange(epos.size)[start_idx:],tof_corr[start_idx:],'.', 
        markersize=.1,
        marker=',',
        markeredgecolor='#1f77b4aa')
ax.set(xlabel='event index', ylabel='m/z', ylim=[0,1200])

ax.grid()

ax2 = ax.twinx()

new_x = laser_df['Ion Sequence Number']*(epos.size-1)/laser_df['Ion Sequence Number'].max()
X = laser_df['Laser X Position (mic)'].values
Y = laser_df['Laser Y Position (mic)'].values

ax2.plot(new_x,X,'-', 
        markersize=5,
#        markeredgecolor='#1f77b4aa',
        color='tab:orange',
        label='X')
ax2.plot(new_x,Y,'-', 
        markersize=5,
#        markeredgecolor='#1f77b4aa',
        color='tab:red',
        label='Y')

ax.set_xlim(start_idx,epos.size)

ax.set_ylim(0,1000)


fig.tight_layout()











event_idx = np.arange(epos.size)
pulse_idx = np.cumsum(np.int64(epos['pslep']))

# maps from ion seq to pulse idx


laser_df = pd.read_csv(r'C:\Users\bwc\Documents\NetBeansProjects\R44_03115\R44_03115_LaserPositionHist.csv')

tof_df = pd.read_csv(r'C:\Users\bwc\Documents\NetBeansProjects\R44_03115\R44_03115_TOFHist.csv')

freq_df = pd.read_csv(r'C:\Users\bwc\Documents\NetBeansProjects\R44_03115\R44_03115_FreqHist.csv')
ion_idx_to_pulse_freq = scipy.interpolate.interp1d(freq_df['Ion Sequence Number'].values,freq_df['Pulser Frequency (kHz)'].values*1000)


pulse_period = 1/ion_idx_to_pulse_freq(event_idx)

wall_time = np.cumsum(pulse_period*np.int64(epos['pslep']))/60/60

ion_idx_to_wall_time = scipy.interpolate.interp1d(event_idx,wall_time,fill_value='extrapolate')




fig = plt.figure(num=11331)
fig.clear()
ax = fig.gca()

#ax.plot(freq_df['Ion Sequence Number'].values,freq_df['Pulser Frequency (kHz)'].values*1000)

#ax.plot(wall_time,pulse_period)

ax.plot(wall_time,tof_bcorr,'.', 
        markersize=.1,
        marker=',',
        markeredgecolor='#1f77b4aa')
ax.set(xlabel='wall time', ylabel='m/z', ylim=[0,1200])

ax.grid()

ax2 = ax.twinx()

t = ion_idx_to_wall_time(laser_df['Ion Sequence Number'].values*(epos.size-1)/laser_df['Ion Sequence Number'].max())

X = laser_df['Laser X Position (mic)'].values
Y = laser_df['Laser Y Position (mic)'].values

ax2.plot(t,X,'-', 
#        markersize=5,
#        markeredgecolor='#1f77b4aa',
        color='tab:orange',
        label='X')
ax2.plot(t,Y,'-', 
#        markersize=5,
#        markeredgecolor='#1f77b4aa',
        color='tab:red',
        label='Y')

#ax.set_xlim(start_idx,epos.size)

ax.set_ylim(85,90)
ax.set_ylim(0,1000)


fig.tight_layout()






plotting_stuff.plot_histo(epos['m2q'],131131,user_label='raw')
plotting_stuff.plot_histo(epos['m2q']/pointwise_scales,131131,user_label='corr',clearFigure=False)





cts_per_slice=2**12
import time
t_start = time.time()
pointwise_scales,piecewise_scales = sel_align_m2q_log_xcorr_v2.get_all_scale_coeffs(epos['m2q'],
                                                         m2q_roi=[0.8,80],
                                                         cts_per_slice=cts_per_slice,
                                                         max_scale=1.15)
t_end = time.time()
print('Total Time = ',t_end-t_start)

def extents(f):
  delta = f[1] - f[0]
  return [f[0] - delta/2, f[-1] + delta/2]

def create_histogram(xs,ys,y_roi=None,delta_y=1.6e-3):
    
    

    num_y = int(np.ceil(np.abs(np.diff(y_roi))/delta_y/2)*2) # even number
#    num_ly = int(2**np.round(np.log2(np.abs(np.diff(ly_roi))/delta_ly)))-1 # closest power of 2
    print('number of points in ly = ',num_y)
    num_x = 1024
    
#    xs = np.arange(ys.size)
    
    N,x_edges,y_edges = np.histogram2d(xs,ys,bins=[num_x,num_y],range=[[np.min(xs),np.max(xs)],y_roi],density=False)
    return (N,x_edges,y_edges)

# Plot histogram for sio2
fig = plt.figure(figsize=(2*3.14961,2*3.14961),num=876543121,dpi=100)
plt.clf()
ax1 = fig.gca()


N,x_edges,y_edges = create_histogram(wall_time,epos['m2q']/pointwise_scales,y_roi=[0.0,100.0],delta_y=0.025)
ax1.imshow(np.log10(1+1*np.transpose(N)), aspect='auto', 
           extent=extents(x_edges) + extents(y_edges), origin='lower', cmap=cc.cm.CET_L8,
           interpolation='nearest')

ax1.set(ylabel='m/z')
ax1.set(xlabel='hours')



ax2 = ax1.twinx()

#t = ion_idx_to_wall_time(laser_df['Ion Sequence Number'].values*(epos.size-1)/laser_df['Ion Sequence Number'].max())

#X = laser_df['Laser X Position (mic)'].values
#Y = laser_df['Laser Y Position (mic)'].values

import colorcet

ax2.plot(t,X,'-', 
#        markersize=5,
#        markeredgecolor='#1f77b4aa',
        color=colorcet.cm.glasbey_category10(0),
        lw=2,
        label='X')
ax2.plot(t,Y,'-', 
#        markersize=5,
#        markeredgecolor='#1f77b4aa',
        lw=2,
         color=colorcet.cm.glasbey_category10(1),
        label='Y')



#ax1.plot(pointwise_scales*16)

ax1.set_ylim(0,70)










































