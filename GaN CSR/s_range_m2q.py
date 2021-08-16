# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# standard imports 
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
import sys
import os
parent_path = '..\\nistapttools'
if parent_path not in sys.path:
    sys.path.append(os.path.abspath(parent_path))    
import time

# custom imports
import apt_fileio
import plotting_stuff
import initElements_P3

import peak_param_determination as ppd

from histogram_functions import bin_dat

# Read in data
#fn = r"\\cfs2w.nist.gov\647\NIST_Projects\EUV_APT_IMS\BWC\180821_GaN_A71\R20_07094-v03.epos"
fn = r"GaN epos files\R20_18161-v01.epos" # Mg doped
#fn = r"\\cfs2w.nist.gov\647\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\R20_07148-v01_vbmq_corr.epos"
#fn = r"\\cfs2w.nist.gov\647\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\R20_07248-v01.epos"
#fn = r"\\cfs2w.nist.gov\647\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\R20_07249-v01.epos"
#fn = r"\\cfs2w.nist.gov\647\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\R20_07250-v01.epos"
#fn = r"\\cfs2w.nist.gov\647\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\190421_AlGaN50p7_A83\R20_07209-v01.epos"
#
#fn = r"\\cfs2w.nist.gov\647\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\181210_D315_A74\R20_07167-v03.epos"
#fn = r"\\cfs2w.nist.gov\647\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\181210_D315_A74\R20_07148-v02.epos"
#fn = r"\\cfs2w.campus.nist.gov\647\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\181204_InGaNQW_A73\R20_07144-v02.epos"



fn = fn[:-5]+'_vbm_corr.epos'
epos = apt_fileio.read_epos_numpy(fn)
#epos = epos[epos.size//2:-1]

# Plot m2q vs event index and show the current ROI selection
roi_event_idxs = np.arange(5000,epos.size-10000)

#roi_event_idxs = np.arange(epos.size)
ax = plotting_stuff.plot_m2q_vs_time(epos['m2q'],epos,fig_idx=1)
ax.plot(roi_event_idxs[0]*np.ones(2),[0,1200],'--k')
ax.plot(roi_event_idxs[-1]*np.ones(2),[0,1200],'--k')
ax.set_title('roi selected to start analysis')
epos = epos[roi_event_idxs]

# Compute some extra information from epos information
LASER_REP_RATE = 10000.0
wall_time = np.cumsum(epos['pslep'])/LASER_REP_RATE
pulse_idx = np.arange(0,epos.size)
isSingle = np.nonzero(epos['ipp'] == 1)

# Define peaks to range
ed = initElements_P3.initElements()
#                            N      Ga      Mg  Da
# Define possible peaks

pk_data =   np.array(    [  (1,     0,      0,  ed['N'].isotopes[14][0]/2),
                            (1,     0,      0,  ed['N'].isotopes[14][0]/1),
                            (1,     0,      0,  ed['N'].isotopes[15][0]/1),
                            (1,     0,      0,  ed['N'].isotopes[15][0]+ed['H'].isotopes[1][0]),
                            (0,     1,      0,  ed['Ga'].isotopes[69][0]/3),
                            (0,     1,      0,  ed['Ga'].isotopes[71][0]/3),
                            (2,     0,      0,  ed['N'].isotopes[14][0]*2),
                            (2,     0,      0,  ed['N'].isotopes[14][0]+ed['N'].isotopes[15][0]),
                            (2,     0,      0,  ed['N'].isotopes[14][0]+ed['N'].isotopes[15][0]+ed['H'].isotopes[1][0]),
                            (0,     1,      0,  ed['Ga'].isotopes[69][0]/2),
                            (0,     1,      0,  ed['Ga'].isotopes[71][0]/2),
                            (1,     1,      0,  (ed['N'].isotopes[14][0] + ed['Ga'].isotopes[69][0])/2),
                            (3,     0,      0,  ed['N'].isotopes[14][0]*3),
                            (1,     1,      0,  (ed['N'].isotopes[14][0] + ed['Ga'].isotopes[71][0])/2),
                            (3,     1,      0,  (ed['Ga'].isotopes[69][0]+3*ed['N'].isotopes[14][0])/2),
                            (3,     1,      0,  (ed['Ga'].isotopes[71][0]+3*ed['N'].isotopes[14][0])/2),
                            (0,     1,      0,  ed['Ga'].isotopes[69][0]),
                            (0,     1,      0,  ed['Ga'].isotopes[71][0]),
                            (0,     1,      0,  ed['Ga'].isotopes[71][0]+ed['H'].isotopes[1][0])
                            ],
                            dtype=[('N','i4'),('Ga','i4'),('In','i4'),('m2q','f4')] )

##                            N      Ga      Al  Da
## Define possible peaks
#
#pk_data =   np.array(    [  (1,     0,      0,  ed['N'].isotopes[14][0]/2),
#                            (0,     0,      1,  ed['Al'].isotopes[27][0]/3),
#                            (0,     0,      1,  ed['Al'].isotopes[27][0]/2),
#                            (1,     0,      0,  ed['N'].isotopes[14][0]/1),
#                            (1,     0,      0,  ed['N'].isotopes[15][0]/1),
#                            (1,     0,      0,  ed['N'].isotopes[15][0]+ed['H'].isotopes[1][0]),
#                            (1,     0,      1,  (ed['N'].isotopes[14][0]+ed['Al'].isotopes[27][0])/2),
#                            (0,     1,      0,  ed['Ga'].isotopes[69][0]/3),
#                            (0,     1,      0,  ed['Ga'].isotopes[71][0]/3),
#                            (0,     0,      1,  ed['Al'].isotopes[27][0]/1),
#                            (2,     0,      0,  ed['N'].isotopes[14][0]*2),
#                            (2,     0,      0,  ed['N'].isotopes[14][0]+ed['N'].isotopes[15][0]),
#                            (2,     0,      0,  ed['N'].isotopes[14][0]+ed['N'].isotopes[15][0]+ed['H'].isotopes[1][0]),
#                            (0,     1,      0,  ed['Ga'].isotopes[69][0]/2),
#                            (0,     1,      0,  ed['Ga'].isotopes[71][0]/2),
#                            (1,     1,      0,  (ed['N'].isotopes[14][0] + ed['Ga'].isotopes[69][0])/2),
#                            (3,     0,      0,  ed['N'].isotopes[14][0]*3),
#                            (1,     1,      0,  (ed['N'].isotopes[14][0] + ed['Ga'].isotopes[71][0])/2),
#                            (3,     1,      0,  (ed['Ga'].isotopes[69][0]+3*ed['N'].isotopes[14][0])/2),
#                            (3,     1,      0,  (ed['Ga'].isotopes[71][0]+3*ed['N'].isotopes[14][0])/2),
#                            (0,     1,      0,  ed['Ga'].isotopes[69][0]),
#                            (0,     1,      0,  ed['Ga'].isotopes[71][0]),
#                            (0,     1,      0,  ed['Ga'].isotopes[71][0]+ed['H'].isotopes[1][0])
#                            ],
#                            dtype=[('N','i4'),('Ga','i4'),('Al','i4'),('m2q','f4')] )


# Define which peaks to use for CSR calcs
Ga1p_m2qs = [ed['Ga'].isotopes[69][0], ed['Ga'].isotopes[71][0]]
Ga2p_m2qs = [ed['Ga'].isotopes[69][0]/2, ed['Ga'].isotopes[71][0]/2]

Ga1p_idxs = [np.argmin(np.abs(m2q-pk_data['m2q'])) for m2q in Ga1p_m2qs]
Ga2p_idxs = [np.argmin(np.abs(m2q-pk_data['m2q'])) for m2q in Ga2p_m2qs]

# Range the peaks
pk_params = ppd.get_peak_ranges(epos,pk_data['m2q'],peak_height_fraction=0.1)
    
# Determine the global background
glob_bg_param = ppd.get_glob_bg(epos['m2q'])

# Count the peaks, local bg, and global bg
cts = ppd.do_counting(epos,pk_params,glob_bg_param)

# Test for peak S/N and throw out craptastic peaks
B = np.max(np.c_[cts['local_bg'][:,None],cts['global_bg'][:,None]],1)[:,None]
T = cts['total'][:,None]
S = T-B
std_S = np.sqrt(T+B)
# Make up a threshold for peak detection... for the most part this won't matter
# since weak peaks don't contribute to stoichiometry much... except for Mg!
is_peak = S>2*np.sqrt(2*B)
for idx, ct in enumerate(cts):
    if not is_peak[idx]:
        for i in np.arange(len(ct)):
            ct[i] = 0
        
# Calculate compositions
compositions = ppd.do_composition(pk_data,cts)
ppd.pretty_print_compositions(compositions,pk_data)
    
print('Total Ranged Ions: '+str(np.sum(cts['total'])))
print('Total Ranged Local Background Ions: '+str(np.sum(cts['local_bg'])))
print('Total Ranged Global Background Ions: '+str(np.sum(cts['global_bg'])))
print('Total Ions: '+str(epos.size))

print('Overall CSR (no bg)    : '+str(np.sum(cts['total'][Ga2p_idxs])/np.sum(cts['total'][Ga1p_idxs])))
print('Overall CSR (local bg) : '+str((np.sum(cts['total'][Ga2p_idxs])-np.sum(cts['local_bg'][Ga2p_idxs]))/(np.sum(cts['total'][Ga1p_idxs])-np.sum(cts['local_bg'][Ga1p_idxs]))))
print('Overall CSR (global bg): '+str((np.sum(cts['total'][Ga2p_idxs])-np.sum(cts['global_bg'][Ga2p_idxs]))/(np.sum(cts['total'][Ga1p_idxs])-np.sum(cts['global_bg'][Ga1p_idxs]))))


# Plot all the things
xs, ys = bin_dat(epos['m2q'],user_roi=[0.5, 100],isBinAligned=True)
#ys_sm = ppd.do_smooth_with_gaussian(ys,10)
ys_sm = ppd.moving_average(ys,10)

glob_bg = ppd.physics_bg(xs,glob_bg_param)    

fig = plt.figure(num=100)
fig.clear()
ax = fig.gca()

ax.plot(xs,ys_sm,label='hist')
ax.plot(xs,glob_bg,label='global bg')

ax.set(xlabel='m/z (Da)', ylabel='counts')
ax.grid()
fig.tight_layout()
fig.canvas.manager.window.raise_()
ax.set_yscale('log')    
ax.legend()

for idx,pk_param in enumerate(pk_params):
    
    if is_peak[idx]:
        ax.plot(np.array([1,1])*pk_param['pre_rng'] ,np.array([0.5,(pk_param['amp']+pk_param['off'])]),'k--')
        ax.plot(np.array([1,1])*pk_param['post_rng'] ,np.array([0.5,(pk_param['amp']+pk_param['off'])]),'k--')
        ax.plot(np.array([1,1])*pk_param['pre_bg_rng'] ,np.array([0.5,(pk_param['amp']+pk_param['off'])]),'m--')
        ax.plot(np.array([1,1])*pk_param['post_bg_rng'] ,np.array([0.5,(pk_param['amp']+pk_param['off'])]),'m--')
        
        ax.plot(np.array([pk_param['pre_bg_rng'],pk_param['post_bg_rng']]) ,np.ones(2)*pk_param['loc_bg'],'g--')
    else:
        ax.plot(np.array([1,1])*pk_param['x0_mean_shift'] ,np.array([0.5,(pk_param['amp']+pk_param['off'])]),'r--')
        
plt.pause(0.1)




#### END BASIC ANALYSIS ####


import sys
sys.exit()


#### START EXPLORATORY ANALYSIS ####

Ga_idx =0

fig = plt.figure(num=1000)
fig.clear()
ax = fig.gca()
ax.plot(epos['v_dc'])

sub_epos = epos[140000:203000]
r = np.sqrt(sub_epos['x_det']**2+sub_epos['y_det']**2)
sub_idxs = np.nonzero(r>15)
cts = ppd.do_counting(sub_epos[sub_idxs],pk_params,glob_bg_param)
csr = np.sum(cts['total'][Ga2p_idxs])/np.sum(cts['total'][Ga1p_idxs])
Ga_comp = ppd.do_composition(pk_data,cts)[0][0][Ga_idx]
Ga_comp_std = ppd.do_composition(pk_data,cts)[0][1][Ga_idx]



#r = np.sqrt(epos['x_det']**2+epos['y_det']**2)
#
#ax.plot(r,epos['v_dc'],',')
#
#dv = np.r_[0,np.diff(epos['v_dc'])]
#
#ax.plot(dv)
#for idx,pt in enumerate(dv):
#    if pt!=0:
#        ax.plot([idx,idx],[0,pt],'r')
#    


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=8, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans.fit(epos['v_dc'].reshape(-1, 1))


fig = plt.figure(num=1000)
fig.clear()
ax = fig.gca()

ax.plot(kmeans.labels_)
ax.plot(epos['v_dc']/np.max(epos['v_dc'])*np.max(kmeans.labels_))


# Slice and dice the data in wall_time
idxs_list = []
for label in np.unique(kmeans.labels_):
    idxs_list.append(np.nonzero(kmeans.labels_ == label))


# Count and compositions
csr = []
Ga_comp = []
Ga_comp_std = []

keys = list(pk_data.dtype.fields.keys())
keys.remove('m2q')
Ga_idx = keys.index('Ga')

for loop_idx, idxs in enumerate(idxs_list):
    sub_epos = epos[idxs]
    r = np.sqrt(sub_epos['x_det']**2+sub_epos['y_det']**2)
    sub_idxs = np.nonzero(r>15)
    cts = ppd.do_counting(sub_epos[sub_idxs],pk_params,glob_bg_param)
    csr.append(np.sum(cts['total'][Ga2p_idxs])/np.sum(cts['total'][Ga1p_idxs]))
    Ga_comp.append(ppd.do_composition(pk_data,cts)[0][0][Ga_idx])
    Ga_comp_std.append(ppd.do_composition(pk_data,cts)[0][1][Ga_idx])

for loop_idx, idxs in enumerate(idxs_list):
    sub_epos = epos[idxs]
    r = np.sqrt(sub_epos['x_det']**2+sub_epos['y_det']**2)
    sub_idxs = np.nonzero(r<15)
    cts = ppd.do_counting(sub_epos[sub_idxs],pk_params,glob_bg_param)
    csr.append(np.sum(cts['total'][Ga2p_idxs])/np.sum(cts['total'][Ga1p_idxs]))
    Ga_comp.append(ppd.do_composition(pk_data,cts)[0][0][Ga_idx])
    Ga_comp_std.append(ppd.do_composition(pk_data,cts)[0][1][Ga_idx])

fig = plt.figure(num=101)
fig.clear()
ax = fig.gca()

#ax.plot(csr,Ga_comp,'.',label='time based')
ax.errorbar(csr,Ga_comp,yerr=Ga_comp_std,fmt='.',capsize=4)
ax.plot([np.min(csr),np.max(csr)],[0.25,0.25],'k--')

ax.set(xlabel='CSR', ylabel='Ga %', ylim=[0, 1], xlim=[2e-3,10])
#ax.grid()
#fig.tight_layout()
#fig.canvas.manager.window.raise_()
#ax.legend()
#fig = plt.figure(num=101)
#fig.clear()
#ax = fig.gca()
#ax.plot(np.arange(Ga_comp.size),csr,'.')


ax.set_xscale('log')
ax.grid()
fig.tight_layout()




















# Slice and dice the data in wall_time
idxs_list = []
STEP = 10000
s_idx = 0
while s_idx<epos.size:
    e_idx = np.min([epos.size,s_idx+STEP])
    idxs_list.append([s_idx,e_idx])
    s_idx = e_idx
    
# Count and compositions
csr = np.full(len(idxs_list),-1.0)
Ga_comp = np.full(len(idxs_list),-1.0)
Ga_comp_std = np.full(len(idxs_list),-1.0)
In_comp = np.full(len(idxs_list),-1.0)
In_comp_std = np.full(len(idxs_list),-1.0)



keys = list(pk_data.dtype.fields.keys())
keys.remove('m2q')
Ga_idx = keys.index('Ga')
In_idx = keys.index('In')
for loop_idx, idxs in enumerate(idxs_list):
    sub_epos = epos[idxs[0]:idxs[1]]
    r = np.sqrt(sub_epos['x_det']**2+sub_epos['y_det']**2)
    sub_idxs = np.nonzero(r>=0)
    cts = ppd.do_counting(sub_epos[sub_idxs],pk_params,glob_bg_param)
  
    Ga_comp[loop_idx] = ppd.do_composition(pk_data,cts)[0][0][Ga_idx]
    Ga_comp_std[loop_idx] = ppd.do_composition(pk_data,cts)[0][1][Ga_idx]
    
    In_comp[loop_idx] = ppd.do_composition(pk_data,cts)[0][0][In_idx]
    In_comp_std[loop_idx] = ppd.do_composition(pk_data,cts)[0][1][In_idx]

fig = plt.figure(num=101)
fig.clear()
ax = fig.gca()
ax.errorbar((np.arange(Ga_comp.size)+0.5)*STEP,In_comp,yerr=In_comp_std,fmt='.',capsize=4)

















# Slice and dice the data in wall_time
idxs_list = []
STEP = 10000
s_idx = 0
while s_idx<epos.size:
    e_idx = np.min([epos.size,s_idx+STEP])
    idxs_list.append([s_idx,e_idx])
    s_idx = e_idx
    
# Count and compositions
csr = np.full(len(idxs_list),-1.0)
Ga_comp = np.full(len(idxs_list),-1.0)
Ga_comp_std = np.full(len(idxs_list),-1.0)

keys = list(pk_data.dtype.fields.keys())
keys.remove('m2q')
Ga_idx = keys.index('Ga')
#In_idx = keys.index('In')
for loop_idx, idxs in enumerate(idxs_list):
    sub_epos = epos[idxs[0]:idxs[1]]
    r = np.sqrt(sub_epos['x_det']**2+sub_epos['y_det']**2)
    sub_idxs = np.nonzero(r>15)
    cts = ppd.do_counting(sub_epos[sub_idxs],pk_params,glob_bg_param)
    csr[loop_idx] = np.sum(cts['total'][Ga2p_idxs])/np.sum(cts['total'][Ga1p_idxs])
    Ga_comp[loop_idx] = ppd.do_composition(pk_data,cts)[0][0][Ga_idx]
    Ga_comp_std[loop_idx] = ppd.do_composition(pk_data,cts)[0][1][Ga_idx]


fig = plt.figure(num=101)
fig.clear()
ax = fig.gca()

#ax.plot(csr,Ga_comp,'.',label='time based')
ax.errorbar(csr,Ga_comp,yerr=Ga_comp_std,fmt='.',capsize=4)
ax.plot([np.min(csr),np.max(csr)],[0.5,0.5],'k--')

ax.set(xlabel='CSR', ylabel='Ga %', ylim=[0, 1], xlim=[0,2])
#ax.grid()
#fig.tight_layout()
#fig.canvas.manager.window.raise_()
#ax.legend()
fig = plt.figure(num=101)
fig.clear()
ax = fig.gca()
ax.plot(np.arange(Ga_comp.size),csr,'.')
#ax.plot(np.arange(Ga_comp.size),Ga_comp,'.')




# Slice and dice the data in detector space (polar)
idxs_list = []
STEP = 2

r = np.sqrt(np.square(epos['x_det'])+np.square(epos['y_det']))

for rq in np.arange(2,32,STEP):
    idxs = np.where((r>rq) & (r<=(rq+STEP)))[0]
    idxs_list.append(idxs)    

# Count and compositions
csr = np.full(len(idxs_list),-1.0)
Ga_comp = np.full(len(idxs_list),-1.0)
Ga_comp_std = np.full(len(idxs_list),-1.0)


Ga_idx = keys.index('Ga')
for loop_idx, idxs in enumerate(idxs_list):
    cts = ppd.do_counting(epos[idxs],pk_params,glob_bg_param)
    
    csr[loop_idx] = np.sum(cts['total'][Ga2p_idxs])/np.sum(cts['total'][Ga1p_idxs])
    Ga_comp[loop_idx] = ppd.do_composition(pk_data,cts)[0][0][Ga_idx]
    Ga_comp_std[loop_idx] = ppd.do_composition(pk_data,cts)[0][1][Ga_idx]

ax.errorbar(csr,Ga_comp,yerr=Ga_comp_std,fmt='.',capsize=4,label='det based (radial)')
ax.plot([np.min(csr),np.max(csr)],[0.25,0.25],'k--')

ax.set(xlabel='CSR', ylabel='Ga %', ylim=[0, 1], xlim=[0,4])

#ax.plot([0,5],[0.5,0.5],'k--')


#
#
## Slice and dice the data in detector space (grid)
#idxs_list = []
#STEP = 10
#
#for x in np.arange(-30,30,STEP):
#    for y in np.arange(-30,30,STEP):
#        idxs = np.where((epos['x_det']>x) & (epos['x_det']<=(x+STEP)) & (epos['y_det']>y) & (epos['y_det']<=(y+STEP)))[0]
#        idxs_list.append(idxs)    
#    
## Count and compositions
#csr = np.full(len(idxs_list),-1.0)
#Ga_comp = np.full(len(idxs_list),-1.0)
#
#for loop_idx, idxs in enumerate(idxs_list):
#    cts = ppd.do_counting(epos[idxs],pk_params,glob_bg_param)
#    csr[loop_idx] = (cts['total'][9]+cts['total'][10])/(cts['total'][14]+cts['total'][15])
#    Ga_comp[loop_idx] = ppd.do_composition(pk_data,cts)[0][1]
#
#
#ax.plot(csr,Ga_comp,'.',label='det based (grid)')
ax.plot([0,5],[0.5,0.5],'k--')

ax.set(xlabel='CSR', ylabel='Ga %', ylim=[0, 1], xlim=[0, 5])
#ax.set_xscale('log')
ax.grid()
fig.tight_layout()
fig.canvas.manager.window.raise_()
ax.legend()

