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
epos = GaN_fun.load_epos(run_number='R20_07209', 
                         epos_trim=[5000, 5000],
                         fig_idx=999)

pk_data = GaN_type_peak_assignments.AlGaN()
bg_rois=[[0.4,0.9]]

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

ax.set_xlim(0,80)
ax.set_ylim(5e0,5e4)
ax.grid(b=True)
ax.set(xlabel='m/z', ylabel='counts')
ax.set_yscale('log')    
ax.legend()
fig.tight_layout()

fig.savefig('AlGaN_full_spectrum.pdf')
fig.savefig('AlGaN_full_spectrum.jpg', dpi=300)








#### END BASIC ANALYSIS ####

#sys.exit([0])

#### START BONUS ANALYSIS ####

# Import some GaN helper functions

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

N_comp_glob = np.full([N_time_chunks,N_ann_chunks],-1.0)
Al_comp_glob = np.full([N_time_chunks,N_ann_chunks],-1.0)

N_comp_std_glob = np.full([N_time_chunks,N_ann_chunks],-1.0)
Al_comp_std_glob = np.full([N_time_chunks,N_ann_chunks],-1.0)

tot_cts = np.full([N_time_chunks,N_ann_chunks],-1.0)

keys = list(pk_data.dtype.fields.keys())
keys.remove('m2q')
Ga_idx = keys.index('Ga')
Al_idx = keys.index('Al')
N_idx = keys.index('N')

for t_idx in np.arange(N_time_chunks):
    for a_idx in np.arange(N_ann_chunks):
        idxs = idxs_list[t_idx][a_idx]
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
        
        
        csr[t_idx,a_idx] = np.sum(cts['total'][Ga2p_idxs]-cts['global_bg'][Ga2p_idxs])/np.sum(cts['total'][Ga1p_idxs]-cts['global_bg'][Ga1p_idxs])
        Ga_comp[t_idx,a_idx] = ppd.do_composition(pk_data,cts)[0][0][Ga_idx]
        Ga_comp_std[t_idx,a_idx] = ppd.do_composition(pk_data,cts)[0][1][Ga_idx]
    
        Ga_comp_glob[t_idx,a_idx] = ppd.do_composition(pk_data,cts)[2][0][Ga_idx]
        Ga_comp_std_glob[t_idx,a_idx] = ppd.do_composition(pk_data,cts)[2][1][Ga_idx]
        
        N_comp_glob[t_idx,a_idx] = ppd.do_composition(pk_data,cts)[2][0][N_idx]
        N_comp_std_glob[t_idx,a_idx] = ppd.do_composition(pk_data,cts)[2][1][N_idx]
        
        Al_comp_glob[t_idx,a_idx] = ppd.do_composition(pk_data,cts)[2][0][Al_idx]
        Al_comp_std_glob[t_idx,a_idx] = ppd.do_composition(pk_data,cts)[2][1][Al_idx]
        
        ppd.pretty_print_compositions(compositions,pk_data)
        print('COUNTS IN CHUNK: ',np.sum(cts['total']))
        tot_cts[t_idx,a_idx] = np.sum(cts['total'])


fig = plt.figure(num=11)
fig.clear()
ax = fig.gca()

#ax.errorbar(csr.flatten(),Ga_comp.flatten(),yerr=Ga_comp_std.flatten(),fmt='.',capsize=4,label='det based (radial)')
ax.errorbar(csr.flatten(),Ga_comp_glob.flatten(),yerr=Ga_comp_std_glob.flatten(),fmt='.',capsize=4, label='Ga %')
ax.errorbar(csr.flatten(),Al_comp_glob.flatten(),yerr=Al_comp_std_glob.flatten(),fmt='.',capsize=4, label='Al %')
ax.errorbar(csr.flatten(),N_comp_glob.flatten(),yerr=N_comp_std_glob.flatten(),fmt='.',capsize=4, label='N %')
ax.plot([np.min(csr),np.max(csr)],[0.25,0.25],'k--')
ax.plot([np.min(csr),np.max(csr)],[0.5,0.5],'k--')

ax.set(xlabel='CSR', ylabel='%', ylim=[0, 1], xlim=[5e-3,5])
ax.legend()
ax.set_title('det radius and time based chunking')
ax.set_xscale('log')
ax.grid()       
fig.tight_layout()
fig.canvas.manager.window.raise_()




fig = plt.figure(num=12)
fig.clear()
ax = fig.gca()

#ax.errorbar(csr.flatten(),Ga_comp.flatten(),yerr=Ga_comp_std.flatten(),fmt='.',capsize=4,label='det based (radial)')
ax.errorbar(csr.flatten(),
            Ga_comp_glob.flatten()+Al_comp_glob.flatten(),
            yerr=np.sqrt(Ga_comp_std_glob**2+Al_comp_std_glob**2).
            flatten(),
            fmt='.',
            capsize=4)

ax.plot([np.min(csr),np.max(csr)],[0.5,0.5],'k--')

ax.set(xlabel='CSR', ylabel='Ga + Al %', ylim=[0, 1], xlim=[5e-3,5])
ax.legend()
ax.set_title('det radius and time based chunking')
ax.set_xscale('log')
ax.grid()       
fig.tight_layout()
fig.canvas.manager.window.raise_()



# Plot stuff
plt.close(13)
fig, axs = plt.subplots(2,3,num=13)

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








# Make a 2D plot of CSR
fig = GaN_fun.create_csr_2d_plots(epos, pk_params, Ga1p_idxs, Ga2p_idxs, fig_idx=500)








