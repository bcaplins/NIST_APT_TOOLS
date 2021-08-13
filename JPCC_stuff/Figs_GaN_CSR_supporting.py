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


narrow_fig = plt.figure()
narrow_fig.set_size_inches(w=3.345, h=3.345)
 
wide_fig = plt.figure()
wide_fig.set_size_inches(w=6.69, h=3)

run_number='R20_07094'
comp_csr_fig_idx=narrow_fig.number
spec_fig_num=wide_fig.number





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
ax.plot(xs, ys_sm, lw=1, label=run_number)    
glob_bg = ppd.physics_bg(xs,glob_bg_param)    
ax.plot(xs, glob_bg, lw=1, label=run_number+' (bg)', alpha=1)



# Find indexes for Ga1p and Ga2p
HW = 0.3        
Ga1p = np.array([68.92, 70.92])
Ga2p = Ga1p/2
    





# Compute the plot of CSR vs time
wall_time = np.cumsum(np.int64(epos['pslep']))/1e4/60/60 # in hours

N_time = 2**7

time_edges = np.linspace(np.min(wall_time), np.max(wall_time), N_time+1)
time_centers = 0.5*(time_edges[:-1]+time_edges[1:])

Ga1p_cts = np.zeros(N_time)
Ga2p_cts = np.zeros(N_time)

for time_idx in range(N_time):
    sub_epos = epos[(wall_time >= time_edges[time_idx]) & (wall_time < time_edges[time_idx+1])]

    sel_idxs1 = np.where(((sub_epos['m2q']>(Ga1p[0]-HW)) & (sub_epos['m2q']<(Ga1p[0]+HW))) \
                         | ((sub_epos['m2q']>(Ga1p[1]-HW)) & (sub_epos['m2q']<(Ga1p[1]+HW))))[0]
    sel_idxs2 = np.where(((sub_epos['m2q']>(Ga2p[0]-HW)) & (sub_epos['m2q']<(Ga2p[0]+HW))) \
                         | ((sub_epos['m2q']>(Ga2p[1]-HW)) & (sub_epos['m2q']<(Ga2p[1]+HW))))[0]

    Ga1p_cts[time_idx] = sel_idxs1.size
    Ga2p_cts[time_idx] = sel_idxs2.size

    
fig = plt.figure(321321)
plt.clf()
ax = fig.gca()


from matplotlib import cm

cm.tab10(0)

ax.plot(wall_time,epos['v_dc'], color=cm.tab10(0))
ax.yaxis.label.set_color(cm.tab10(0))
ax.spines['left'].set_color(cm.tab10(0))    
ax.tick_params(axis='y', colors=cm.tab10(0))
ax.set_ylabel('DC voltage')
ax.set_xlabel('elapsed time (hours)')




twin_ax = ax.twinx()
twin_ax.step(time_centers, Ga2p_cts/(Ga1p_cts+1),'r',where='mid')
twin_ax.set_ylabel('CSR  $\\left( \\frac{Ga^{2+}}{Ga^{1+}} \\right)$')
twin_ax.yaxis.label.set_color('red')
twin_ax.spines['right'].set_color('red')    
twin_ax.tick_params(axis='y', colors='red')
twin_ax.spines['left'].set_color(cm.tab10(0))    


sel_idxs1 = np.where(((epos['m2q']>(Ga1p[0]-HW)) & (epos['m2q']<(Ga1p[0]+HW))) \
                         | ((epos['m2q']>(Ga1p[1]-HW)) & (epos['m2q']<(Ga1p[1]+HW))))[0]
sel_idxs2 = np.where(((epos['m2q']>(Ga2p[0]-HW)) & (epos['m2q']<(Ga2p[0]+HW))) \
                         | ((epos['m2q']>(Ga2p[1]-HW)) & (epos['m2q']<(Ga2p[1]+HW))))[0]

AVG_CSR = sel_idxs2.size/sel_idxs1.size

fig.tight_layout()






# Plot some detector hitmaps
GaN_fun.create_det_hit_plots_SI(epos,pk_data,pk_params,fig_idx = plt.figure().number)

# Find the pole center and show it
ax = plt.gcf().get_axes()[0]
m2q_roi = [3, 100]
sel_idxs = np.where((epos['m2q']>m2q_roi[0]) & (epos['m2q']<m2q_roi[1]))
xc,yc = GaN_fun.mean_shift(epos['x_det'][sel_idxs],epos['y_det'][sel_idxs])

r_edges = np.sqrt(np.linspace(0, 28**2, 3+1))


a_circle = plt.Circle((xc[-1],yc[-1]), 0.5, facecolor='none', edgecolor='k', lw=2, ls='-')
ax.add_artist(a_circle)

for r in r_edges[1:]:
    a_circle = plt.Circle((xc[-1],yc[-1]), r, facecolor='none', edgecolor='k', lw=2, ls='-')
    ax.add_artist(a_circle)


rs = np.sqrt((epos['x_det']-xc[-1])**2+(epos['y_det']-yc[-1])**2)

NNN = 2**6
r_edges = np.sqrt(np.linspace(0, 28**2, NNN+1))

csr_rad = np.zeros(NNN)

for r_idx in range(NNN):
    sub_epos = epos[(rs>=r_edges[r_idx]) & (rs<r_edges[r_idx+1])]

    sel_idxs1 = np.where(((sub_epos['m2q']>(Ga1p[0]-HW)) & (sub_epos['m2q']<(Ga1p[0]+HW))) \
                         | ((sub_epos['m2q']>(Ga1p[1]-HW)) & (sub_epos['m2q']<(Ga1p[1]+HW))))[0]
    sel_idxs2 = np.where(((sub_epos['m2q']>(Ga2p[0]-HW)) & (sub_epos['m2q']<(Ga2p[0]+HW))) \
                         | ((sub_epos['m2q']>(Ga2p[1]-HW)) & (sub_epos['m2q']<(Ga2p[1]+HW))))[0]

    csr_rad[r_idx] = sel_idxs2.size/sel_idxs1.size

plt.gcf().tight_layout()
    

fig = plt.figure(3231)
plt.clf()
ax = fig.gca()
#ax.step(r_edges[:-1], csr_rad, where='post', lw=2)

ax.plot(0.5*(r_edges[:-1]+r_edges[1:]), csr_rad, '-o')

ax.set_ylabel('CSR  $\\left( \\frac{Ga^{2+}}{Ga^{1+}} \\right)$')
ax.set_xlabel('$r_{detector}$ (mm)')
ax.set_xlim(0,r_edges[-1])
ax.set_yscale('log')
ax.set_ylim(.5,10)





fig.tight_layout()




import sys
sys.exit()












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
        
        compositions = ppd.do_composition(pk_data,cts)
#            ppd.pretty_print_compositions(compositions,pk_data)
#            print('COUNTS IN CHUNK: ',np.sum(cts['total']))
        tot_cts[t_idx,a_idx] = np.sum(cts['total'])


fig = plt.figure(num=comp_csr_fig_idx)
#fig.clear()
ax = fig.gca()

#ax.errorbar(csr.flatten(),Ga_comp.flatten(),yerr=Ga_comp_std.flatten(),fmt='.',capsize=4,label='det based (radial)')
ax.errorbar(csr.flatten(),Ga_comp_glob.flatten(),yerr=Ga_comp_std_glob.flatten(),fmt='.',capsize=4, label=run_number)
# Write data to console for copy/pasting
print('CSR','\t','Ga comp (glob bg)','\t','Ga comp std (glob bg)')
for i in np.arange(csr.size):
    print(csr.flatten()[i],'\t',Ga_comp_glob.flatten()[i],'\t',Ga_comp_std_glob.flatten()[i])












import sys
sys.exit()


















def CSR_plot(run_number, comp_csr_fig_idx, spec_fig_num):
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
    ax.plot(xs, ys_sm, lw=1, label=run_number)    
    glob_bg = ppd.physics_bg(xs,glob_bg_param)    
    ax.plot(xs, glob_bg, lw=1, label=run_number+' (bg)', alpha=1)
    
    
    
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
            
            compositions = ppd.do_composition(pk_data,cts)
#            ppd.pretty_print_compositions(compositions,pk_data)
#            print('COUNTS IN CHUNK: ',np.sum(cts['total']))
            tot_cts[t_idx,a_idx] = np.sum(cts['total'])
    
    
    fig = plt.figure(num=comp_csr_fig_idx)
    #fig.clear()
    ax = fig.gca()
    
    #ax.errorbar(csr.flatten(),Ga_comp.flatten(),yerr=Ga_comp_std.flatten(),fmt='.',capsize=4,label='det based (radial)')
    ax.errorbar(csr.flatten(),Ga_comp_glob.flatten(),yerr=Ga_comp_std_glob.flatten(),fmt='.',capsize=4, label=run_number)
    # Write data to console for copy/pasting
    print('CSR','\t','Ga comp (glob bg)','\t','Ga comp std (glob bg)')
    for i in np.arange(csr.size):
        print(csr.flatten()[i],'\t',Ga_comp_glob.flatten()[i],'\t',Ga_comp_std_glob.flatten()[i])

    
    return fig


csr_fig = plt.figure()
csr_fig.set_size_inches(w=3.345, h=3.345)
 
spec_fig = plt.figure()
spec_fig.set_size_inches(w=6.69, h=3)
    
#    
#CSR_plot(run_number='R20_07094',
#         comp_csr_fig_idx=csr_fig.number,
#         spec_fig_num=spec_fig.number) # 'Template'
CSR_plot(run_number='R20_07247',
         comp_csr_fig_idx=csr_fig.number,
         spec_fig_num=spec_fig.number) # CSR ~ 2
CSR_plot(run_number='R20_07248',
         comp_csr_fig_idx=csr_fig.number,
         spec_fig_num=spec_fig.number) # CSR ~ 2
CSR_plot(run_number='R20_07249',
         comp_csr_fig_idx=csr_fig.number,
         spec_fig_num=spec_fig.number) # CSR ~ 0.5
CSR_plot(run_number='R20_07250',
         comp_csr_fig_idx=csr_fig.number,
         spec_fig_num=spec_fig.number) # CSR ~ 0.1

ax = csr_fig.gca()

#xlim = [5e-3, 5]
xlim = [1e-2, 1e1]

ax.set(xlabel='CSR', ylabel='Ga %', ylim=[0, 1], xlim=xlim)
ax.plot(xlim,[0.5,0.5],'k--', label='nominal')
    
ax.set_ylim(0.35,0.65)
#ax.legend()
ax.set_title('det radius and time based chunking')
ax.set_xscale('log')
ax.grid(b=True)
csr_fig.tight_layout()

csr_fig.savefig('GaN_CSR_plot.pdf')
csr_fig.savefig('GaN_CSR_plot.jpg', dpi=300)

    

ax = spec_fig.gca()


ax.set_xlim(0,120)
ax.set_ylim(1,5e4)
ax.grid(b=True)
ax.set(xlabel='m/z', ylabel='counts')
ax.set_yscale('log')    
ax.legend()
spec_fig.tight_layout()

spec_fig.savefig('GaN_full_CSR_spectrum.pdf')
spec_fig.savefig('GaN_full_CSR_spectrum.jpg', dpi=300)



