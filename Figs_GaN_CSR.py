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

def counter_g():
    count = 0
    while True:
        yield count
        count += 1
counter = counter_g()
        
        

def CSR_plot(run_number, comp_csr_fig_idx, spec_fig_num, multi_fig_num):
    # Read in data
    epos = GaN_fun.load_epos(run_number=run_number, 
                             epos_trim=[5000, 5000],
                             fig_idx=plt.figure().number)
    
    pk_data = GaN_type_peak_assignments.GaN()
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
    
    ppd.pretty_print_compositions(compositions,pk_data)

    # Plot the full spectrum
    xs, ys_sm = GaN_fun.bin_and_smooth_spectrum(epos=epos,
                                                user_roi=[0,150],
                                                bin_wid_mDa=30,
                                                smooth_wid_mDa=-1)
    
    fig = plt.figure(num=spec_fig_num)
    ax = fig.gca()    
    
    scale_factor = 1E4**next(counter)
    
    ax.plot(xs, scale_factor*(ys_sm+1), lw=1, label=run_number)    
    glob_bg = ppd.physics_bg(xs,glob_bg_param)    
    ax.plot(xs, scale_factor*(glob_bg+1), lw=1, label=run_number+' (bg)', alpha=1)
    
    
    
    # Plot some detector hitmaps
    GaN_fun.create_det_hit_plots(epos,pk_data,pk_params,fig_idx = plt.figure().number)
    
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
    
    # Multiplicy
    hit_multiplicity = np.full([N_time_chunks,N_ann_chunks],-1.0)
    
    tot_cts = np.full([N_time_chunks,N_ann_chunks],-1.0)
    
    keys = list(pk_data.dtype.fields.keys())
    keys.remove('m2q')
    Ga_idx = keys.index('Ga')
    
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
            
            
            low_mz_idxs = np.where(sub_epos['m2q']<100)[0]
            hit_multiplicity[t_idx,a_idx] = np.sum(sub_epos[low_mz_idxs]['ipp']!=1)/sub_epos[low_mz_idxs].size
            
            compositions = ppd.do_composition(pk_data,cts)
#            ppd.pretty_print_compositions(compositions,pk_data)
#            print('COUNTS IN CHUNK: ',np.sum(cts['total']))
            tot_cts[t_idx,a_idx] = np.sum(cts['total'])
    
    
    fig = plt.figure(num=comp_csr_fig_idx)
    #fig.clear()
    ax = fig.gca()
    
    #ax.errorbar(csr.flatten(),Ga_comp.flatten(),yerr=Ga_comp_std.flatten(),fmt='.',capsize=4,label='det based (radial)')
    ax.errorbar(csr.flatten(),Ga_comp_glob.flatten(),yerr=Ga_comp_std_glob.flatten(),fmt='.',capsize=4, label=run_number)
    
    fig = plt.figure(num=multi_fig_num)
    #fig.clear()
    ax = fig.gca()
    ax.plot(csr.flatten(),hit_multiplicity.flatten(),'.', label=run_number)
    
    
    
    
    # Write data to console for copy/pasting
    print('CSR','\t','Ga comp (glob bg)','\t','Ga comp std (glob bg)')
    for i in np.arange(csr.size):
        print(csr.flatten()[i],'\t',Ga_comp_glob.flatten()[i],'\t',Ga_comp_std_glob.flatten()[i])

    
    return fig


csr_fig = plt.figure()
csr_fig.set_size_inches(w=3.345, h=3.345)

multi_fig = plt.figure()
multi_fig.set_size_inches(w=3.345, h=3.345)
 
 
spec_fig = plt.figure()
spec_fig.set_size_inches(w=6.69, h=3)
    
#    
#CSR_plot(run_number='R20_07094',
#         comp_csr_fig_idx=csr_fig.number,
#         spec_fig_num=spec_fig.number) # 'Template'
CSR_plot(run_number='R20_07247',
         comp_csr_fig_idx=csr_fig.number,
         spec_fig_num=spec_fig.number,
         multi_fig_num=multi_fig.number) # CSR ~ 2
CSR_plot(run_number='R20_07248',
         comp_csr_fig_idx=csr_fig.number,
         spec_fig_num=spec_fig.number,
         multi_fig_num=multi_fig.number)# CSR ~ 2
CSR_plot(run_number='R20_07249',
         comp_csr_fig_idx=csr_fig.number,
         spec_fig_num=spec_fig.number,
         multi_fig_num=multi_fig.number) # CSR ~ 0.5
CSR_plot(run_number='R20_07250',
         comp_csr_fig_idx=csr_fig.number,
         spec_fig_num=spec_fig.number,
         multi_fig_num=multi_fig.number) # CSR ~ 0.1

ax = csr_fig.gca()

#xlim = [5e-3, 5]
xlim = [1e-2, 1e1]

ax.set(xlabel='CSR', ylabel='Ga %', ylim=[0, 1], xlim=xlim)
ax.plot(xlim,[0.5,0.5],'k--', label='nominal')
    
ax.set_ylim(0.4,0.6)
#ax.legend()
ax.set_title('det radius and time based chunking')
ax.set_xscale('log')
ax.grid(b=True)
csr_fig.tight_layout()

csr_fig.savefig('GaN_CSR_plot.pdf')
csr_fig.savefig('GaN_CSR_plot.jpg', dpi=300)

    

ax = multi_fig.gca()

#xlim = [5e-3, 5]
xlim = [1e-2, 1e1]

ax.set(xlabel='CSR', ylabel='multihit frac.', ylim=[0, 1], xlim=xlim)
    
ax.set_ylim(0,0.65)
#ax.legend()
ax.set_title('det radius and time based chunking')
ax.set_xscale('log')
ax.grid(b=True)
multi_fig.tight_layout()

multi_fig.savefig('GaN_CSR_multi_plot.pdf')
multi_fig.savefig('GaN_CSR_multi_plot.jpg', dpi=300)



ax = spec_fig.gca()


ax.set_xlim(0,120)
ax.set_ylim(1,5e16)
ax.grid(b=True)
ax.set(xlabel='m/z', ylabel='counts')
ax.set_yscale('log')    
ax.legend()
spec_fig.tight_layout()

spec_fig.savefig('GaN_full_CSR_spectrum.pdf')
spec_fig.savefig('GaN_full_CSR_spectrum.jpg', dpi=300)



