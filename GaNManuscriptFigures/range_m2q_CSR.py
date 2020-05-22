# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Need to put the functions in the path
# Probably not necessary if I understood Python/Git/modules better
import os 
import sys
parent_directory = os.getcwd().rsplit(sep='\\',maxsplit=1)[0]
if parent_directory not in sys.path:
    sys.path.insert(1, parent_directory)

# standard imports 
import numpy as np
import matplotlib.pyplot as plt

# custom imports
import apt_fileio
import plotting_stuff
import initElements_P3

import peak_param_determination as ppd
from histogram_functions import bin_dat

#plt.close('all')

# Read in data
work_dir = r"C:\Users\capli\Google Drive\NIST\pos_and_epos_files\GaN_manuscript"

#fn = work_dir+"\\"+r"R20_07094-v03.epos" # template
#fn = work_dir+"\\"+r"R20_07148-v01.epos" # Mg doped
#fn = work_dir+"\\"+r"R20_07199-v03.epos" # InGaN QW
fn = work_dir+"\\"+r"R20_07247.epos" # CSR ~ 2
#fn = work_dir+"\\"+r"R20_07248-v01.epos" # CSR ~ 2
#fn = work_dir+"\\"+r"R20_07249-v01.epos" # CSR ~ 0.5
#fn = work_dir+"\\"+r"R20_07250-v01.epos" # CSR ~ 0.1


fn = fn[:-5]+'_vbm_corr.epos'
epos = apt_fileio.read_epos_numpy(fn)
#epos = epos[epos.size//2:-1]

# Plot m2q vs event index and show the current ROI selection
roi_event_idxs = np.arange(5000,epos.size-10000)
roi_event_idxs = np.arange(0,epos.size)

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

import peak_assignments
pk_data = peak_assignments.GaN()

# Define which peaks to use for CSR calcs
Ga1p_m2qs = [ed['Ga'].isotopes[69][0], ed['Ga'].isotopes[71][0]]
Ga2p_m2qs = [ed['Ga'].isotopes[69][0]/2, ed['Ga'].isotopes[71][0]/2]

Ga1p_idxs = [np.argmin(np.abs(m2q-pk_data['m2q'])) for m2q in Ga1p_m2qs]
Ga2p_idxs = [np.argmin(np.abs(m2q-pk_data['m2q'])) for m2q in Ga2p_m2qs]

# Range the peaks
pk_params = ppd.get_peak_ranges(epos,pk_data['m2q'],peak_height_fraction=0.1)
    
# Determine the global background
#glob_bg_param = ppd.fit_uncorr_bg(epos['m2q'],fit_roi=[3.5,6.5])
bg_rois=[[3.5,6.5],[90,110]]
bg_rois=[[0.5,0.9],[8,11],[90,110]]

glob_bg_param = ppd.get_glob_bg(epos['m2q'],rois=bg_rois)

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
xs, ys = bin_dat(epos['m2q'],user_roi=[0.5, 120],isBinAligned=True)
#ys_sm = ppd.do_smooth_with_gaussian(ys,30)
ys_sm = ppd.moving_average(ys,30)

glob_bg = ppd.physics_bg(xs,glob_bg_param)    

fig = plt.figure(num=2)
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
        
for roi in bg_rois:
    xbox = np.array([roi[0],roi[0],roi[1],roi[1]])
    ybox = np.array([0.1,np.max(ys_sm)/10,np.max(ys_sm)/10,0.1])
    
    ax.fill(xbox,ybox, 'b', alpha=0.2)

        
plt.pause(0.1)






#### END BASIC ANALYSIS ####

#sys.exit([0])

#### START BONUS ANALYSIS ####

# Import some GaN helper functions
import GaN_fun

# Plot some detector hitmaps
GaN_fun.create_det_hit_plots(epos,pk_data,pk_params,fig_idx = 10)

# Find the pole center and show it
ax = plt.gcf().get_axes()[0]
m2q_roi = [3, 100]
sel_idxs = np.where((epos['m2q']>m2q_roi[0]) & (epos['m2q']<m2q_roi[1]))
xc,yc = GaN_fun.mean_shift(epos['x_det'][sel_idxs],epos['y_det'][sel_idxs])
a_circle = plt.Circle((xc[-1],yc[-1]), 10, facecolor='none', edgecolor='k', lw=2, ls='-')
ax.add_artist(a_circle)



# Chop data up by Radius AND time
time_chunk_centers,r_centers,idxs_list = GaN_fun.chop_data_rad_and_time(epos,
                                                                        [xc[-1],
                                                                         yc[-1]],
                                                                         time_chunk_size=2**16,
                                                                         N_ann_chunks=3)

N_time_chunks = time_chunk_centers.size
N_ann_chunks = r_centers.size

# Charge state ratios
csr = np.full([N_time_chunks,N_ann_chunks],-1.0)

# Gallium atomic % and standard deviation (no bg)
Ga_comp = np.full([N_time_chunks,N_ann_chunks],-1.0)
Ga_comp_std = np.full([N_time_chunks,N_ann_chunks],-1.0)

# Gallium atomic % and standard deviation (global bg)
Ga_comp_glob = np.full([N_time_chunks,N_ann_chunks],-1.0)
Ga_comp_std_glob = np.full([N_time_chunks,N_ann_chunks],-1.0)

tot_cts = np.full([N_time_chunks,N_ann_chunks],-1.0)

keys = list(pk_data.dtype.fields.keys())
keys.remove('m2q')
Ga_idx = keys.index('Ga')

for t_idx in np.arange(N_time_chunks):
    for a_idx in np.arange(N_ann_chunks):
        idxs = idxs_list[t_idx][a_idx]
        sub_epos = epos[idxs]
        
        glob_bg_param_chunk = ppd.get_glob_bg(sub_epos['m2q'], rois=bg_rois)

        cts = ppd.do_counting(sub_epos,pk_params,glob_bg_param_chunk)
        
        csr[t_idx,a_idx] = np.sum(cts['total'][Ga2p_idxs]-cts['global_bg'][Ga2p_idxs])/np.sum(cts['total'][Ga1p_idxs]-cts['global_bg'][Ga1p_idxs])
        Ga_comp[t_idx,a_idx] = ppd.do_composition(pk_data,cts)[0][0][Ga_idx]
        Ga_comp_std[t_idx,a_idx] = ppd.do_composition(pk_data,cts)[0][1][Ga_idx]
    
        Ga_comp_glob[t_idx,a_idx] = ppd.do_composition(pk_data,cts)[2][0][Ga_idx]
        Ga_comp_std_glob[t_idx,a_idx] = ppd.do_composition(pk_data,cts)[2][1][Ga_idx]
        
        compositions = ppd.do_composition(pk_data,cts)
        ppd.pretty_print_compositions(compositions,pk_data)
        print('COUNTS IN CHUNK: ',np.sum(cts['total']))
        tot_cts[t_idx,a_idx] = np.sum(cts['total'])


fig = plt.figure(num=11)
#fig.clear()
ax = fig.gca()

#ax.errorbar(csr.flatten(),Ga_comp.flatten(),yerr=Ga_comp_std.flatten(),fmt='.',capsize=4,label='det based (radial)')
ax.errorbar(csr.flatten(),Ga_comp_glob.flatten(),yerr=Ga_comp_std_glob.flatten(),fmt='.',capsize=4)
ax.plot([np.min(csr),np.max(csr)],[0.5,0.5],'k--')

ax.set(xlabel='CSR', ylabel='Ga %', ylim=[0, 1], xlim=[5e-3,5])
ax.set_ylim(0.25,0.75)
ax.legend()
ax.set_title('det radius and time based chunking')
ax.set_xscale('log')
ax.grid(b=True)
fig.tight_layout()
fig.canvas.manager.window.raise_()

# Plot stuff
plt.close(12)
fig, axs = plt.subplots(2,3,num=12)

(c1,c2,csr1),(s1,s2,cts1) = axs

ax = c1
im, cbar = GaN_fun.heatmap(Ga_comp*100,
                           ['idxs~{:.0f}'.format(n) for n in time_chunk_centers],                            
                           ['r~{:.1f} mm'.format(n) for n in r_centers], 
                           ax=ax,
                           cmap="YlGn", cbarlabel=r"$\mu$ Ga % (no bg)")
texts = GaN_fun.annotate_heatmap(im, valfmt="{x:.1f}")

ax = c2
im, cbar = GaN_fun.heatmap(Ga_comp_glob*100,
                           ['idxs~{:.0f}'.format(n) for n in time_chunk_centers],                            
                           ['r~{:.1f} mm'.format(n) for n in r_centers], 
                           ax=ax,
                           cmap="YlGn", cbarlabel=r"$\mu$ Ga % (glob bg)")
texts = GaN_fun.annotate_heatmap(im, valfmt="{x:.1f}")

ax = s1
im, cbar = GaN_fun.heatmap(Ga_comp_std*100,
                           ['idxs~{:.0f}'.format(n) for n in time_chunk_centers],                            
                           ['r~{:.1f} mm'.format(n) for n in r_centers], 
                           ax=ax,
                           cmap="YlGn", cbarlabel=r"$\sigma$ Ga % (no bg)")
texts = GaN_fun.annotate_heatmap(im, valfmt="{x:.1f}")

ax = s2
im, cbar = GaN_fun.heatmap(Ga_comp_std_glob*100,
                           ['idxs~{:.0f}'.format(n) for n in time_chunk_centers],                            
                           ['r~{:.1f} mm'.format(n) for n in r_centers], 
                           ax=ax,
                           cmap="YlGn", cbarlabel=r"$\sigma$ Ga % (glob bg)")
texts = GaN_fun.annotate_heatmap(im, valfmt="{x:.1f}")

ax = csr1
im, cbar = GaN_fun.heatmap(csr,
                           ['idxs~{:.0f}'.format(n) for n in time_chunk_centers],                            
                           ['r~{:.1f} mm'.format(n) for n in r_centers], 
                           ax=ax,
                           cmap="YlGn", cbarlabel=r"$Ga^{2+}/Ga^{1+}$")
texts = GaN_fun.annotate_heatmap(im, valfmt="{x:.2f}")

ax = cts1
im, cbar = GaN_fun.heatmap(tot_cts,
                           ['idxs~{:.0f}'.format(n) for n in time_chunk_centers],                            
                           ['r~{:.1f} mm'.format(n) for n in r_centers], 
                           ax=ax,
                           cmap="binary", cbarlabel=r"total ranged cts")
texts = GaN_fun.annotate_heatmap(im, valfmt="{x:.0f}")
#fig.tight_layout()

# Write data to console for copy/pasting
print('CSR','\t','Ga comp (glob bg)','\t','Ga comp std (glob bg)')
for i in np.arange(csr.size):
    print(csr.flatten()[i],'\t',Ga_comp_glob.flatten()[i],'\t',Ga_comp_std_glob.flatten()[i])










# Chop data up by time
time_chunk_centers,r_centers,idxs_list = GaN_fun.chop_data_rad_and_time(epos,
                                                                        [xc[-1],
                                                                         yc[-1]],
                                                                         time_chunk_size=2**14,
                                                                         N_ann_chunks=1)

N_time_chunks = time_chunk_centers.size
N_ann_chunks = r_centers.size

avg_volt = np.full([N_time_chunks,N_ann_chunks],-1.0)

# Charge state ratios
csr = np.full([N_time_chunks,N_ann_chunks],-1.0)

# Gallium atomic % and standard deviation (no bg)
Ga_comp = np.full([N_time_chunks,N_ann_chunks],-1.0)
Ga_comp_std = np.full([N_time_chunks,N_ann_chunks],-1.0)

# Gallium atomic % and standard deviation (global bg)
Ga_comp_glob = np.full([N_time_chunks,N_ann_chunks],-1.0)
Ga_comp_std_glob = np.full([N_time_chunks,N_ann_chunks],-1.0)

tot_cts = np.full([N_time_chunks,N_ann_chunks],-1.0)

keys = list(pk_data.dtype.fields.keys())
keys.remove('m2q')
Ga_idx = keys.index('Ga')

for t_idx in np.arange(N_time_chunks):
    for a_idx in np.arange(N_ann_chunks):
        idxs = idxs_list[t_idx][a_idx]
        sub_epos = epos[idxs]
        
        glob_bg_param_chunk = ppd.get_glob_bg(sub_epos['m2q'], rois=bg_rois)

        cts = ppd.do_counting(sub_epos,pk_params,glob_bg_param_chunk)
        
        csr[t_idx,a_idx] = np.sum(cts['total'][Ga2p_idxs]-cts['global_bg'][Ga2p_idxs])/np.sum(cts['total'][Ga1p_idxs]-cts['global_bg'][Ga1p_idxs])
        Ga_comp[t_idx,a_idx] = ppd.do_composition(pk_data,cts)[0][0][Ga_idx]
        Ga_comp_std[t_idx,a_idx] = ppd.do_composition(pk_data,cts)[0][1][Ga_idx]
    
        Ga_comp_glob[t_idx,a_idx] = ppd.do_composition(pk_data,cts)[2][0][Ga_idx]
        Ga_comp_std_glob[t_idx,a_idx] = ppd.do_composition(pk_data,cts)[2][1][Ga_idx]
        
        avg_volt[t_idx,a_idx] = np.mean(sub_epos['v_dc'])
        
        compositions = ppd.do_composition(pk_data,cts)
        ppd.pretty_print_compositions(compositions,pk_data)
        print('COUNTS IN CHUNK: ',np.sum(cts['total']))
        tot_cts[t_idx,a_idx] = np.sum(cts['total'])


fig = plt.figure(num=13)
fig.clear()
ax = fig.gca()

#ax.errorbar(csr.flatten(),Ga_comp.flatten(),yerr=Ga_comp_std.flatten(),fmt='.',capsize=4,label='det based (radial)')
#ax.errorbar(csr.flatten(),Ga_comp_glob.flatten(),yerr=Ga_comp_std_glob.flatten(),fmt='.',capsize=4)
color = 'tab:blue'
ax.set_xlabel('wall time')
ax.set_ylabel('Ga %', color=color)

ax.errorbar(np.arange(Ga_comp_glob.size), Ga_comp_glob.flatten(), label='Ga %',yerr=Ga_comp_std_glob.flatten(),fmt='.',capsize=4,color=color)
ax.set_ylim(.25,.75)
ax.plot([0,Ga_comp_glob.size],[0.5,0.5],'--', color=color)

ax.tick_params(axis='y', labelcolor=color)

color = 'tab:red'
ax_twin = ax.twinx()
ax_twin.set_ylabel('CSR', color=color)

ax_twin.plot(csr.flatten(), label='CSR', color=color)
ax_twin.tick_params(axis='y', labelcolor=color)

#ax.set(xlabel='CSR', ylabel='Ga %', ylim=[0, 1], xlim=[5e-3,5])

ax.set_title('det radius and time based chunking')
#ax.set_xscale('log')
ax.grid()       
fig.tight_layout()



fig = plt.figure(num=14)
fig.clear()
ax = fig.gca()

#ax.errorbar(csr.flatten(),Ga_comp.flatten(),yerr=Ga_comp_std.flatten(),fmt='.',capsize=4,label='det based (radial)')
#ax.errorbar(csr.flatten(),Ga_comp_glob.flatten(),yerr=Ga_comp_std_glob.flatten(),fmt='.',capsize=4)
color = 'tab:blue'
ax.set_xlabel('wall time')
ax.set_ylabel('V_dc', color=color)

ax.plot(np.arange(Ga_comp_glob.size), avg_volt.flatten(), label='$V_{dc}$', color=color)


ax.tick_params(axis='y', labelcolor=color)

color = 'tab:red'
ax_twin = ax.twinx()
ax_twin.set_ylabel('CSR', color=color)

ax_twin.plot(csr.flatten(), label='CSR', color=color)
ax_twin.tick_params(axis='y', labelcolor=color)

#ax.set(xlabel='CSR', ylabel='Ga %', ylim=[0, 1], xlim=[5e-3,5])

ax.set_title('det radius and time based chunking')
#ax.set_xscale('log')
ax.grid()       
fig.tight_layout()





fig = plt.figure(num=24)
#fig.clear()
ax = fig.gca()
ax.errorbar(csr.flatten(), Ga_comp_glob.flatten(), label='Ga %',yerr=Ga_comp_std_glob.flatten(),fmt='.',capsize=4)



#ax.plot([0,r_centers[-1]],[0.5,0.5],'--', color=color)
ax.set(xlabel='CSR', ylabel='Ga %', ylim=[0, 1], xlim=[5e-3,10])
ax.set_ylim(.25,.75)
ax.set_title('time based chunking')
ax.set_xscale('log')
ax.grid(b=True)
fig.tight_layout()















# Chop data up by radius
time_chunk_centers,r_centers,idxs_list = GaN_fun.chop_data_rad_and_time(epos,
                                                                        [xc[-1],
                                                                         yc[-1]],
                                                                         time_chunk_size=epos.size,
                                                                         N_ann_chunks=9)

N_time_chunks = time_chunk_centers.size
N_ann_chunks = r_centers.size

avg_volt = np.full([N_time_chunks,N_ann_chunks],-1.0)

# Charge state ratios
csr = np.full([N_time_chunks,N_ann_chunks],-1.0)

# Gallium atomic % and standard deviation (no bg)
Ga_comp = np.full([N_time_chunks,N_ann_chunks],-1.0)
Ga_comp_std = np.full([N_time_chunks,N_ann_chunks],-1.0)

# Gallium atomic % and standard deviation (global bg)
Ga_comp_glob = np.full([N_time_chunks,N_ann_chunks],-1.0)
Ga_comp_std_glob = np.full([N_time_chunks,N_ann_chunks],-1.0)

tot_cts = np.full([N_time_chunks,N_ann_chunks],-1.0)

keys = list(pk_data.dtype.fields.keys())
keys.remove('m2q')
Ga_idx = keys.index('Ga')

for t_idx in np.arange(N_time_chunks):
    for a_idx in np.arange(N_ann_chunks):
        idxs = idxs_list[t_idx][a_idx]
        sub_epos = epos[idxs]
        
        glob_bg_param_chunk = ppd.get_glob_bg(sub_epos['m2q'], rois=bg_rois)

        cts = ppd.do_counting(sub_epos,pk_params,glob_bg_param_chunk)
        
        csr[t_idx,a_idx] = np.sum(cts['total'][Ga2p_idxs]-cts['global_bg'][Ga2p_idxs])/np.sum(cts['total'][Ga1p_idxs]-cts['global_bg'][Ga1p_idxs])
        
        Ga_comp[t_idx,a_idx] = ppd.do_composition(pk_data,cts)[0][0][Ga_idx]
        Ga_comp_std[t_idx,a_idx] = ppd.do_composition(pk_data,cts)[0][1][Ga_idx]
    
        Ga_comp_glob[t_idx,a_idx] = ppd.do_composition(pk_data,cts)[2][0][Ga_idx]
        Ga_comp_std_glob[t_idx,a_idx] = ppd.do_composition(pk_data,cts)[2][1][Ga_idx]
        
        avg_volt[t_idx,a_idx] = np.mean(sub_epos['v_dc'])
        
        compositions = ppd.do_composition(pk_data,cts)
        ppd.pretty_print_compositions(compositions,pk_data)
        print('COUNTS IN CHUNK: ',np.sum(cts['total']))
        tot_cts[t_idx,a_idx] = np.sum(cts['total'])


fig = plt.figure(num=15)
fig.clear()
ax = fig.gca()

#ax.errorbar(csr.flatten(),Ga_comp.flatten(),yerr=Ga_comp_std.flatten(),fmt='.',capsize=4,label='det based (radial)')
#ax.errorbar(csr.flatten(),Ga_comp_glob.flatten(),yerr=Ga_comp_std_glob.flatten(),fmt='.',capsize=4)
color = 'tab:blue'
ax.set_xlabel('det radius (mm)')
ax.set_ylabel('Ga %', color=color)

ax.errorbar(r_centers, Ga_comp_glob.flatten(), label='Ga %',yerr=Ga_comp_std_glob.flatten(),fmt='.',capsize=4,color=color)
ax.set_ylim(.25,.75)
ax.plot([0,r_centers[-1]],[0.5,0.5],'--', color=color)

ax.tick_params(axis='y', labelcolor=color)

color = 'tab:red'
ax_twin = ax.twinx()
ax_twin.set_ylabel('CSR', color=color)

ax_twin.plot(r_centers, csr.flatten(), label='CSR', color=color)
ax_twin.tick_params(axis='y', labelcolor=color)

#ax.set(xlabel='CSR', ylabel='Ga %', ylim=[0, 1], xlim=[5e-3,5])

ax.set_title('det radius and time based chunking')
#ax.set_xscale('log')
ax.grid()       
fig.tight_layout()




fig = plt.figure(num=16)
#fig.clear()
ax = fig.gca()
ax.errorbar(csr.flatten(), Ga_comp_glob.flatten(), label='Ga %',yerr=Ga_comp_std_glob.flatten(),fmt='.',capsize=4)



#ax.plot([0,r_centers[-1]],[0.5,0.5],'--', color=color)
ax.set(xlabel='CSR', ylabel='Ga %', ylim=[0, 1], xlim=[5e-3,10])
ax.set_ylim(.25,.75)
ax.set_title('det radius based chunking')
ax.set_xscale('log')
ax.grid(b=True)
fig.tight_layout()


# Make a 2D plot of CSR
fig = GaN_fun.create_csr_2d_plots(epos, pk_params, Ga1p_idxs, Ga2p_idxs, fig_idx=500)






