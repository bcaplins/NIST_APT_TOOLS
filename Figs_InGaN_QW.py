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
epos = GaN_fun.load_epos(run_number='R20_07199_redo', 
                         epos_trim=[5000, 5000],
                         fig_idx=999)

pk_data = GaN_type_peak_assignments.In_doped_GaN()
bg_rois=[[0.4,0.9]]
#bg_rois=[[10,11]]

pk_params, glob_bg_param, Ga1p_idxs, Ga2p_idxs = GaN_fun.fit_spectrum(
        epos=epos, 
        pk_data=pk_data, 
        peak_height_fraction=0.1, 
        bg_rois=bg_rois)

cts, compositions, is_peak = GaN_fun.count_and_get_compositions(
        epos=epos, 
        pk_data=pk_data,
        pk_params=pk_params, 
        glob_bg_param=glob_bg_param, 
        bg_frac=1, 
        noise_threshhold=2)

# Print out the composition of the full dataset
ppd.pretty_print_compositions(compositions,pk_data)

# Plot the full spectrum
xs, ys_sm = GaN_fun.bin_and_smooth_spectrum(epos=epos,
                                            user_roi=[0,150],
                                            bin_wid_mDa=30,
                                            smooth_wid_mDa=-1)

fig = plt.figure(num=1)
fig.set_size_inches(w=6.69, h=3)
fig.clear()
ax = fig.gca()

ax.plot(xs, ys_sm, lw=1, label='full spec')

glob_bg = ppd.physics_bg(xs,glob_bg_param)    
ax.plot(xs, glob_bg, lw=1, label='bg', alpha=1)

ax.set_xlim(0,120)
ax.set_ylim(1e0,1e5)
ax.grid(b=True)
ax.set(xlabel='m/z', ylabel='counts')
ax.set_yscale('log')    
ax.legend()
fig.tight_layout()

fig.savefig('InGaN_full_spectrum.pdf')
fig.savefig('InGaN_full_spectrum.jpg', dpi=300)




# Find the pole center and show it
m2q_roi = [3, 100]
sel_idxs = np.where((epos['m2q']>m2q_roi[0]) & (epos['m2q']<m2q_roi[1]))
xc,yc = GaN_fun.mean_shift(epos['x_det'][sel_idxs],epos['y_det'][sel_idxs])

# Find all the Indium events
is_In = ~np.isfinite(epos['m2q'])
for pk,param in zip(pk_data,pk_params):
    if pk['In']>0:
        is_In = is_In | ((epos['m2q']>=param['pre_rng']) & (epos['m2q']<=param['post_rng']))
        
# Fit the QW to a plane and rotate the point cloud to 'flatten' wrt z-axis
p = GaN_fun.qw_plane_fit(epos['x'][is_In],epos['y'][is_In],epos['z'][is_In],np.array([0,0,1.8]))
epos['x'],epos['y'],epos['z'] = GaN_fun.rotate_data_flat(p,epos['x'],epos['y'],epos['z'])

# USER DEFINED!!!
# Z-ROIs TO TAKE COMPOSITIONS IN
z_roi_qw = [3.0, 5.5]
z_roi_buf = [14, 23]
z_roi_gan = [6.5, 12.5]

# Find all the Ga events
is_Ga = ~np.isfinite(epos['m2q'])
for pk,param in zip(pk_data,pk_params):
    if pk['Ga']>0:
        is_Ga = is_Ga | ((epos['m2q']>=param['pre_rng']) & (epos['m2q']<=param['post_rng']))

import colorcet as cc    
cm=cc.cm.glasbey

# Plot the 'flat' interface to verify vector algebra didn't go awry
fig = plt.figure(num=11)
fig.clear()
ax = fig.gca()
#ax.plot(epos['x'][is_In],epos['z'][is_In],'.')
#ax.plot(epos['y'][is_In],epos['z'][is_In],'.')
ax.plot(epos['x'][is_Ga],epos['z'][is_Ga],'.', alpha=0.05, color=cm(2), ms=2)
ax.plot(epos['x'][is_In],epos['z'][is_In],'.', alpha=0.5, color=cm(10), ms=2)


# Plot the ROIs for visual inspection
ax.fill([-8,-8, 8,8], [z_roi_qw[0], z_roi_qw[1], z_roi_qw[1], z_roi_qw[0]], color=[1,0,0,0.5]) 
ax.fill([-8,-8, 8,8], [z_roi_buf[0], z_roi_buf[1], z_roi_buf[1], z_roi_buf[0]], color=[1,0,0,0.5]) 
ax.fill([-8,-8, 8,8], [z_roi_gan[0], z_roi_gan[1], z_roi_gan[1], z_roi_gan[0]], color=[1,0,0,0.5]) 


# Calculate the QW composition
is_roi = (epos['z']>=z_roi_qw[0]) & (epos['z']<=z_roi_qw[1])
sub_epos = epos[is_roi]

bg_frac_roi = [120,150]
bg_frac = np.sum((sub_epos['m2q']>bg_frac_roi[0]) & (sub_epos['m2q']<bg_frac_roi[1])) \
                    / np.sum((epos['m2q']>bg_frac_roi[0]) & (epos['m2q']<bg_frac_roi[1]))

# Count the peaks, local bg, and global bg.  Ignore the local bg based info
cts, compositions, is_peak = GaN_fun.count_and_get_compositions(
        epos=sub_epos, 
        pk_data=pk_data, 
        pk_params=pk_params, 
        glob_bg_param=glob_bg_param, 
        bg_frac=bg_frac, 
        noise_threshhold=2)

ppd.pretty_print_compositions(compositions,pk_data)

# Plot the QW spectrum
xs, ys_sm = GaN_fun.bin_and_smooth_spectrum(epos=sub_epos,
                                            user_roi=[0,150],
                                            bin_wid_mDa=30,
                                            smooth_wid_mDa=-1)

fig = plt.figure(num=2)
fig.set_size_inches(w=6.69, h=3)
fig.clear()
ax = fig.gca()

ax.plot(xs, ys_sm, lw=1, label='QW spec',color='k')

glob_bg = ppd.physics_bg(xs,bg_frac*glob_bg_param)    
ax.plot(xs, glob_bg, lw=1, label='bg', alpha=1,color='r')

ax.set_xlim(0,120)
ax.set_ylim(1e0,1e3)
ax.grid(b=True)
ax.set(xlabel='m/z', ylabel='counts')
ax.set_yscale('log')    
ax.legend()
fig.tight_layout()

fig.savefig('InGaN_QW_spectrum.pdf')
fig.savefig('InGaN_QW_spectrum.jpg', dpi=300)



# Calculate the buffer composition
is_roi = (epos['z']>=z_roi_buf[0]) & (epos['z']<=z_roi_buf[1])
sub_epos = epos[is_roi]

bg_frac_roi = [120,150]
bg_frac = np.sum((sub_epos['m2q']>bg_frac_roi[0]) & (sub_epos['m2q']<bg_frac_roi[1])) \
                    / np.sum((epos['m2q']>bg_frac_roi[0]) & (epos['m2q']<bg_frac_roi[1]))

# Count the peaks, local bg, and global bg.  Ignore the local bg based info
cts, compositions, is_peak = GaN_fun.count_and_get_compositions(
        epos=sub_epos, 
        pk_data=pk_data, 
        pk_params=pk_params, 
        glob_bg_param=glob_bg_param, 
        bg_frac=bg_frac, 
        noise_threshhold=2)

ppd.pretty_print_compositions(compositions,pk_data)

# Plot the buffer spectrum
xs, ys_sm = GaN_fun.bin_and_smooth_spectrum(epos=sub_epos,
                                            user_roi=[0,150],
                                            bin_wid_mDa=30,
                                            smooth_wid_mDa=-1)

fig = plt.figure(num=3)
fig.set_size_inches(w=6.69, h=3)
fig.clear()
ax = fig.gca()

ax.plot(xs, ys_sm, lw=1, label='buffer spec',color='k')

glob_bg = ppd.physics_bg(xs,bg_frac*glob_bg_param)    
ax.plot(xs, glob_bg, lw=1, label='bg', alpha=1,color='r')

ax.set_xlim(0,120)
ax.set_ylim(1e0,1e4)
ax.grid(b=True)
ax.set(xlabel='m/z', ylabel='counts')
ax.set_yscale('log')    
ax.legend()
fig.tight_layout()

fig.savefig('InGaN_buffer_spectrum.pdf')
fig.savefig('InGaN_buffer_spectrum.jpg', dpi=300)

# Calculate the barrier composition
is_roi = (epos['z']>=z_roi_gan[0]) & (epos['z']<=z_roi_gan[1])
sub_epos = epos[is_roi]

bg_frac_roi = [120,150]
bg_frac = np.sum((sub_epos['m2q']>bg_frac_roi[0]) & (sub_epos['m2q']<bg_frac_roi[1])) \
                    / np.sum((epos['m2q']>bg_frac_roi[0]) & (epos['m2q']<bg_frac_roi[1]))

# Count the peaks, local bg, and global bg.  Ignore the local bg based info
cts, compositions, is_peak = GaN_fun.count_and_get_compositions(
        epos=sub_epos, 
        pk_data=pk_data, 
        pk_params=pk_params, 
        glob_bg_param=glob_bg_param, 
        bg_frac=bg_frac, 
        noise_threshhold=2)

ppd.pretty_print_compositions(compositions,pk_data)

# Plot the barrier spectrum
xs, ys_sm = GaN_fun.bin_and_smooth_spectrum(epos=sub_epos,
                                            user_roi=[0,150],
                                            bin_wid_mDa=30,
                                            smooth_wid_mDa=-1)

fig = plt.figure(num=4)
fig.set_size_inches(w=6.69, h=3)
fig.clear()
ax = fig.gca()

ax.plot(xs, ys_sm, lw=1, label='barrier spec',color='k')

glob_bg = ppd.physics_bg(xs,bg_frac*glob_bg_param)    
ax.plot(xs, glob_bg, lw=1, label='bg$\pm', alpha=1,color='r')

ax.set_xlim(0,120)
ax.set_ylim(1e0,1e4)
ax.grid(b=True)
ax.set(xlabel='m/z', ylabel='counts')
ax.set_yscale('log')    
ax.legend()
fig.tight_layout()

fig.savefig('InGaN_barrier_spectrum.pdf')
fig.savefig('InGaN_barrier_spectrum.jpg', dpi=300)



# Adding a section to plot the CSR and Ga+N vs reconstructed z-coordinate


chunk_centers, idxs_list = GaN_fun.chop_data_z(epos,chunk_edges_nm=np.arange(1,24,1))

csr = np.full([len(idxs_list)],-1.0)

avg_z = np.full([len(idxs_list)],-1.0)
avg_voltage = np.full([len(idxs_list)],-1.0)

# Gallium atomic % and standard deviation (no bg)
Ga_comp = np.full([len(idxs_list)],-1.0)
Ga_comp_std = np.full([len(idxs_list)],-1.0)

N_comp = np.full([len(idxs_list)],-1.0)
N_comp_std = np.full([len(idxs_list)],-1.0)


Ga_comp_glob = np.full([len(idxs_list)],-1.0)
Ga_comp_std_glob = np.full([len(idxs_list)],-1.0)

N_comp_glob = np.full([len(idxs_list)],-1.0)
N_comp_std_glob = np.full([len(idxs_list)],-1.0)

hit_multiplicity = np.full([len(idxs_list)],-1.0)

tot_cts = np.full([len(idxs_list)],-1.0)
    
keys = list(pk_data.dtype.fields.keys())
keys.remove('m2q')
Ga_idx = keys.index('Ga')
N_idx = keys.index('N')

for z_idx in range(len(idxs_list)):
    
    idxs = idxs_list[z_idx]
    sub_epos = epos[idxs]
    
    bg_frac_roi = [120,150]
    bg_frac = np.sum((sub_epos['m2q']>bg_frac_roi[0]) & (sub_epos['m2q']<bg_frac_roi[1])) \
                    / np.sum((epos['m2q']>bg_frac_roi[0]) & (epos['m2q']<bg_frac_roi[1]))      
    
    cts, compositions, is_peak = GaN_fun.count_and_get_compositions(
            epos=sub_epos, 
            pk_data=pk_data,
            pk_params=pk_params, 
            glob_bg_param=glob_bg_param, 
            bg_frac=bg_frac, 
            noise_threshhold=2)
    
    csr[z_idx] = np.sum(cts['total'][Ga2p_idxs]-cts['global_bg'][Ga2p_idxs])/np.sum(cts['total'][Ga1p_idxs]-cts['global_bg'][Ga1p_idxs])
    Ga_comp[z_idx] = ppd.do_composition(pk_data,cts)[0][0][Ga_idx]
    Ga_comp_std[z_idx] = ppd.do_composition(pk_data,cts)[0][1][Ga_idx]
    
    N_comp[z_idx] = ppd.do_composition(pk_data,cts)[0][0][N_idx]
    N_comp_std[z_idx] = ppd.do_composition(pk_data,cts)[0][1][N_idx]
    
    Ga_comp_glob[z_idx] = ppd.do_composition(pk_data,cts)[2][0][Ga_idx]
    Ga_comp_std_glob[z_idx] = ppd.do_composition(pk_data,cts)[2][1][Ga_idx]
    
    N_comp_glob[z_idx] = ppd.do_composition(pk_data,cts)[2][0][N_idx]
    N_comp_std_glob[z_idx] = ppd.do_composition(pk_data,cts)[2][1][N_idx]
    
    low_mz_idxs = np.where(sub_epos['m2q']<100)[0]
    hit_multiplicity[z_idx] = np.sum(sub_epos[low_mz_idxs]['ipp']!=1)/sub_epos[low_mz_idxs].size
    
    compositions = ppd.do_composition(pk_data,cts)
    #            ppd.pretty_print_compositions(compositions,pk_data)
    #            print('COUNTS IN CHUNK: ',np.sum(cts['total']))
    tot_cts[z_idx] = np.sum(cts['total'])
    
    avg_z[z_idx] = np.mean(sub_epos['z'])
    
    avg_voltage[z_idx] = np.mean(sub_epos['v_dc'])
    

fig = plt.figure(num=100)
fig.set_size_inches(w=6.69, h=3)
fig.clear()
ax = fig.gca()
ax.plot(avg_z,csr)
ax.set_xlabel('avg_z (nm)')
ax.set_ylabel('csr')
fig.tight_layout()

fig = plt.figure(num=101)
fig.set_size_inches(w=6.69, h=3)
fig.clear()
ax = fig.gca()
ax.plot(avg_z,avg_voltage)
ax.set_xlabel('avg_z (nm)')
ax.set_ylabel('avg voltage')
fig.tight_layout()

fig = plt.figure(num=102)
fig.set_size_inches(w=6.69, h=3)
fig.clear()
ax = fig.gca()
ax.plot(avg_z,csr/avg_voltage)
ax.set_xlabel('avg_z (nm)')
ax.set_ylabel('csr/avg_voltage')
fig.tight_layout()

fig = plt.figure(num=103)
fig.set_size_inches(w=6.69, h=3)
fig.clear()
ax = fig.gca()
ax.plot(avg_z,N_comp, label='N')
ax.plot(avg_z,Ga_comp_glob, label='Ga')
ax.plot(avg_z,Ga_comp_glob+N_comp_glob, label='Ga+N')
ax.set_xlabel('avg_z (nm)')
ax.legend()
fig.tight_layout()


fig = plt.figure(num=104)
fig.set_size_inches(w=6.69, h=3)
fig.clear()
ax = fig.gca()

ax.errorbar(csr.flatten(),Ga_comp_glob.flatten(),yerr=Ga_comp_std_glob.flatten(),fmt='.',capsize=4)


ax.set_xlabel('csr')
ax.set_ylabel('Ga')
ax.set_xscale('log')
ax.grid(b=True)
fig.tight_layout()





sel_idxs = np.where(avg_z>5)[0]


fig = plt.figure(num=1000)
fig.set_size_inches(w=6.69, h=3)
fig.clear()
ax = fig.gca()
ax.plot(avg_z[sel_idxs],csr[sel_idxs])
ax.set_xlabel('depth (nm)')
ax.set_ylabel('Ga CSR')

twin_ax = ax.twinx()
twin_ax.plot(avg_z[sel_idxs],avg_voltage[sel_idxs],'r')
twin_ax.set_ylabel('voltage (V)')

fig.tight_layout()














