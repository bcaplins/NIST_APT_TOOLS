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

# Read in data R20_07148
epos = GaN_fun.load_epos(run_number='R20_07148', 
                         epos_trim=[5000, 1500000],
                         fig_idx=999)

pk_data = GaN_type_peak_assignments.Mg_doped_GaN()
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
                                            bin_wid_mDa=100,
                                            smooth_wid_mDa=-1)

fig = plt.figure(num=1)
fig.set_size_inches(w=3.5, h=2)
fig.clear()
ax = fig.gca()

ax.plot(xs, ys_sm, lw=1, label='full spec',color='k')

glob_bg = ppd.physics_bg(xs,glob_bg_param)    
ax.plot(xs, glob_bg, lw=1, label='bg', alpha=1,color='r')

ax.set_xlim(0,80)
ax.set_ylim(1e1,5e4)
ax.grid(b=True)
ax.set(xlabel='m/z', ylabel='counts')
ax.set_yscale('log')    
ax.legend()
fig.tight_layout()

#fig.savefig('AlGaN_full_spectrum.pdf')
#fig.savefig('AlGaN_full_spectrum.jpg', dpi=300)








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
                                                                         time_chunk_size=2**18,
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
Mg_comp_glob = np.full([N_time_chunks,N_ann_chunks],-1.0)

N_comp_std_glob = np.full([N_time_chunks,N_ann_chunks],-1.0)
Mg_comp_std_glob = np.full([N_time_chunks,N_ann_chunks],-1.0)

tot_cts = np.full([N_time_chunks,N_ann_chunks],-1.0)


glob_bg_cts_roi = np.full([N_time_chunks,N_ann_chunks],-1.0)
tot_cts_roi = np.full([N_time_chunks,N_ann_chunks],-1.0)


keys = list(pk_data.dtype.fields.keys())
keys.remove('m2q')
Ga_idx = keys.index('Ga')
Mg_idx = keys.index('Mg')
N_idx = keys.index('N')

for t_idx in np.arange(N_time_chunks):
    for a_idx in np.arange(N_ann_chunks):
        idxs = idxs_list[t_idx][a_idx]
        sub_epos = epos[idxs]

        bg_frac_roi = [120,150]
        bg_frac = np.sum((sub_epos['m2q']>bg_frac_roi[0]) & (sub_epos['m2q']<bg_frac_roi[1])) \
                            / np.sum((epos['m2q']>bg_frac_roi[0]) & (epos['m2q']<bg_frac_roi[1]))        
        
        
        
        
        xs, ys_sm = GaN_fun.bin_and_smooth_spectrum(epos=sub_epos,
                                            user_roi=[0,150],
                                            bin_wid_mDa=10,
                                            smooth_wid_mDa=-1)
        glob_bg = ppd.physics_bg(xs,glob_bg_param*bg_frac)    
        
        glob_bg_cts_roi[t_idx,a_idx] = np.sum(glob_bg)+np.sum(sub_epos['m2q']>150)
#        tot_cts_roi[t_idx,a_idx] = np.sum(ys_sm)
        tot_cts_roi[t_idx,a_idx] = sub_epos.size
        
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
        
        Mg_comp_glob[t_idx,a_idx] = ppd.do_composition(pk_data,cts)[2][0][Mg_idx]
        Mg_comp_std_glob[t_idx,a_idx] = ppd.do_composition(pk_data,cts)[2][1][Mg_idx]
        
        ppd.pretty_print_compositions(compositions,pk_data)
        print('COUNTS IN CHUNK: ',np.sum(cts['total']))
        tot_cts[t_idx,a_idx] = np.sum(cts['total'])
        
        
        


fig = plt.figure(num=11)
fig.clear()
#fig.set_size_inches(w=3.345, h=3.345)
ax = fig.gca()

#ax.errorbar(csr.flatten(),Ga_comp.flatten(),yerr=Ga_comp_std.flatten(),fmt='.',capsize=4,label='det based (radial)')
ax.errorbar(csr.flatten(),Ga_comp_glob.flatten(),yerr=Ga_comp_std_glob.flatten(),fmt='.',capsize=4, label='Ga %')
#ax.errorbar(csr.flatten(),Mg_comp_glob.flatten(),yerr=Mg_comp_std_glob.flatten(),fmt='.',capsize=4, label='Mg %')
ax.errorbar(csr.flatten(),N_comp_glob.flatten(),yerr=N_comp_std_glob.flatten(),fmt='.',capsize=4, label='N %')
#ax.plot([np.min(csr),np.max(csr)],[0.25,0.25],'k--')
ax.plot([np.min(csr),np.max(csr)],[0.5,0.5],'k--')

#ax.set(xlabel='CSR', ylabel='%', ylim=[0, 1], xlim=[5e-3,5])
ax.set(xlabel='CSR', ylabel='%', ylim=[0.35, 0.65], xlim=[0.01,10])
ax.legend()
ax.set_title('det radius and time based chunking')
ax.set_xscale('log')
ax.grid()       
fig.tight_layout()
fig.canvas.manager.window.raise_()

fig.savefig('MgdGaN_CSR.pdf')
fig.savefig('MgdGaN_CSR.jpg', dpi=300)


fig = plt.figure(num=12)
fig.clear()
ax = fig.gca()

#ax.errorbar(csr.flatten(),Ga_comp.flatten(),yerr=Ga_comp_std.flatten(),fmt='.',capsize=4,label='det based (radial)')
ax.errorbar(csr.flatten(),
            Ga_comp_glob.flatten()+Mg_comp_glob.flatten(),
            yerr=np.sqrt(Ga_comp_std_glob**2+Mg_comp_std_glob**2).
            flatten(),
            fmt='.',
            capsize=4, label='Ga+Al')
ax.errorbar(csr.flatten(),N_comp_glob.flatten(),yerr=N_comp_std_glob.flatten(),fmt='.',capsize=4, label='N')

ax.plot([np.min(csr),np.max(csr)],[0.5,0.5],'k--')

#ax.set(xlabel='CSR', ylabel='Ga + Al %', ylim=[0, 1], xlim=[5e-3,5])
ax.set(xlabel='CSR', ylabel='Ga + Mg %', ylim=[0.2, 0.6], xlim=[0.1,10])
ax.legend()
ax.set_title('det radius and time based chunking')
ax.set_xscale('log')
ax.grid()       
fig.tight_layout()
fig.canvas.manager.window.raise_()



#Plotting Al and Ga as if N=50%
fig = plt.figure(num=13)
fig.clear()
ax = fig.gca()

#ax.errorbar(csr.flatten(),Ga_comp.flatten(),yerr=Ga_comp_std.flatten(),fmt='.',capsize=4,label='det based (radial)')
ax.errorbar(csr.flatten(),Ga_comp_glob.flatten(),yerr=Ga_comp_std_glob.flatten(),fmt='.',capsize=4, label='Ga %')
ax.errorbar(csr.flatten(),Ga_comp_glob.flatten()/(2*(Ga_comp_glob.flatten()+Mg_comp_glob.flatten())),yerr=Ga_comp_std_glob.flatten(),fmt='.',capsize=4, label='Ga N=50%')
ax.errorbar(csr.flatten(),Mg_comp_glob.flatten(),yerr=Mg_comp_std_glob.flatten(),fmt='.',capsize=4, label='Al %')
ax.errorbar(csr.flatten(),Mg_comp_glob.flatten()/(2*(Ga_comp_glob.flatten()+Mg_comp_glob.flatten())),yerr=Ga_comp_std_glob.flatten(),fmt='.',capsize=4, label='Al N=50%')
ax.plot([np.min(csr),np.max(csr)],[0.25,0.25],'k--')
#ax.plot([np.min(csr),np.max(csr)],[0.5,0.5],'k--')

#ax.set(xlabel='CSR', ylabel='%', ylim=[0, 1], xlim=[5e-3,5])
ax.set(xlabel='CSR', ylabel='%', ylim=[0, 0.5], xlim=[0.1,10])
ax.legend()
ax.set_title('det radius and time based chunking')
ax.set_xscale('log')
ax.grid()       
fig.tight_layout()
fig.canvas.manager.window.raise_()


# Plot stuff
plt.close(14)
fig, axs = plt.subplots(2,3,num=14)

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











#Plotting Al and Ga as if N=50%
fig = plt.figure(num=99)
fig.clear()
ax = fig.gca()

#ax.errorbar(csr.flatten(),Ga_comp.flatten(),yerr=Ga_comp_std.flatten(),fmt='.',capsize=4,label='det based (radial)')
ax.plot(csr.flatten(),glob_bg_cts_roi.flatten()/tot_cts_roi.flatten(), 'o', label='tot_cts')
#ax.plot(csr.flatten(),tot_cts.flatten(), 'o', label='tot_cts')
#ax.plot(csr.flatten(),tot_cts_roi.flatten(), 'o', label='tot_cts_roi')
#ax.plot(csr.flatten(),glob_bg_cts_roi.flatten(), 'o', label='glob_bg_cts_roi')


ax.set(xlabel='CSR', ylabel='bg %', ylim=[0, 1], xlim=[5e-3,5])
#ax.set(xlabel='CSR', ylabel='%', ylim=[0, 0.5], xlim=[0.1,10])
ax.legend()
ax.set_title('det radius and time based chunking')
#ax.set_xscale('log')
ax.grid()       
fig.tight_layout()
fig.canvas.manager.window.raise_()





import colorcet as cc


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


def corrhist(epos, delta=0.1):
    # decide whether m2q or tof correlation histogram and set roi
    dat = epos['m2q']
    roi = [0, 150]
    
    # Determine number of bins and create histogram array
    N = int(np.ceil((roi[1]-roi[0])/delta))
    corrhist = np.zeros([N,N], dtype=int)
    
    # Get multihit indicies
    multi_idxs = np.where(epos['ipp']>1)[0]
    for multi_idx in multi_idxs:
        # Get the multiplicity of the multihit and retieve the m/z data
        n_hits = epos['ipp'][multi_idx]
        cluster = dat[multi_idx:multi_idx+n_hits]
        
        # Loop over each unique set of hit pairs in multihit event and add to histogram
        for i in range(n_hits):
            for j in range(i+1,n_hits):
                idx1 = int(np.floor(cluster[i]/delta))
                idx2 = int(np.floor(cluster[j]/delta))
                if idx1 < N and idx2 < N:
                    corrhist[idx1,idx2] += 1
    
    # Create bin edges
    edges = np.arange(roi[0],roi[1]+delta,delta)
    assert edges.size-1 == N
    
    # Symmetrize the correlation histogram during return
    return (edges, corrhist+corrhist.T-np.diag(np.diag(corrhist)))



# Create correlation histogram
edges, ch = corrhist(epos, delta=0.1)

# add one to get rid of zeros and logscale
ch_log = np.log2(1+ch)

fig = plt.figure(num=66, figsize=(8,8))
fig.clf()
ax = fig.gca()   
plot_2d_histo(ax, ch_log, edges, edges)
ax.set_aspect('equal', 'box')
ax.set_xlabel('ion 1 (m/z)')
ax.set_ylabel('ion 2 (m/z)')

# You can change the color limits if you want to blow out the colorscale
#for im in ax.get_images():
#    im.set_clim(0, 1)

lims = [0,80]
ax.set(xlim=lims, ylim=lims)
ax.set_title('Pairwise Correlation Histogram (log color scale)')
fig.savefig('AlGaN_m2q_corr_hist.png')




import initElements_P3
ed = initElements_P3.initElements()

m1 = ed['N'].isotopes[14][0]
m2 = ed['Al'].isotopes[27][0]
mp = (m1+m2)/2

Vd_V0 = np.linspace(0,1,2**5)

m1_eff = m1/(1-Vd_V0*(1-m1/mp))
m2_eff = m2/(1-Vd_V0*(1-m2/mp))

plt.plot(m1_eff,m2_eff,'w--', alpha=0.5)
















