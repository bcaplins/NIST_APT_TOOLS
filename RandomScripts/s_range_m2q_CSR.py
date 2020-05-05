# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Simple background function.  Assumes background has form approx 1/sqrt(m2q).
# Note. bg_param assumed a 1 dalton binning.  Mult by binsize for other sizes.
#



# standard imports 
import numpy as np
import matplotlib.pyplot as plt

# custom imports
import apt_fileio
import plotting_stuff
import initElements_P3

import peak_param_determination as ppd

from histogram_functions import bin_dat

plt.close('all')

# Read in data
#fn = r"\\cfs2w.nist.gov\647\NIST_Projects\EUV_APT_IMS\BWC\180821_GaN_A71\R20_07094-v03.epos"
#fn = r"\\cfs2w.nist.gov\647\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\R20_07148-v01.epos" # Mg doped
#fn = r"\\cfs2w.nist.gov\647\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\R20_07148-v01_vbmq_corr.epos"
#fn = r"\\cfs2w.nist.gov\647\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\R20_07247.epos"
#fn = r"\\cfs2w.nist.gov\647\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\R20_07248-v01.epos"
#fn = r"\\cfs2w.nist.gov\647\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\R20_07249-v01.epos"
#fn = r"\\cfs2w.nist.gov\647\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\R20_07250-v01.epos"
#fn = r"\\cfs2w.nist.gov\647\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\190421_AlGaN50p7_A83\R20_07209-v01.epos"
#
fn = r"\\cfs2w.nist.gov\647\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\181210_D315_A74\R20_07167-v03.epos"
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

#                            N      Ga       Da
# Define possible peaks
pk_data =   np.array(    [  (1,     0,        ed['N'].isotopes[14][0]/2),
                            (1,     0,        ed['N'].isotopes[14][0]/1),
                            (1,     0,        ed['N'].isotopes[15][0]/1),
                            (1,     0,        ed['N'].isotopes[15][0]+ed['H'].isotopes[1][0]),
                            (0,     1,        ed['Ga'].isotopes[69][0]/3),
                            (0,     1,        ed['Ga'].isotopes[71][0]/3),
                            (2,     0,        ed['N'].isotopes[14][0]*2),
                            (2,     0,        ed['N'].isotopes[14][0]+ed['N'].isotopes[15][0]),
                            (2,     0,        ed['N'].isotopes[14][0]+ed['N'].isotopes[15][0]+ed['H'].isotopes[1][0]),
                            (0,     1,        ed['Ga'].isotopes[69][0]/2),
                            (0,     1,        ed['Ga'].isotopes[71][0]/2),
                            (1,     1,        (ed['N'].isotopes[14][0] + ed['Ga'].isotopes[69][0])/2),
                            (3,     0,        ed['N'].isotopes[14][0]*3),
                            (1,     1,        (ed['N'].isotopes[14][0] + ed['Ga'].isotopes[71][0])/2),
                            (3,     1,        (ed['Ga'].isotopes[69][0]+3*ed['N'].isotopes[14][0])/2),
                            (3,     1,        (ed['Ga'].isotopes[71][0]+3*ed['N'].isotopes[14][0])/2),
                            (0,     1,        ed['Ga'].isotopes[69][0]),
                            (0,     1,        ed['Ga'].isotopes[71][0]),
                            (0,     1,        ed['Ga'].isotopes[71][0]+ed['H'].isotopes[1][0]),
                            ],
                            dtype=[('N','i4'),('Ga','i4'),('m2q','f4')] )


#                            N      Ga      In  Da
## Define possible peaks
#
#pk_data =   np.array(    [  (1,     0,      0,  ed['N'].isotopes[14][0]/2),
#                            (1,     0,      0,  ed['N'].isotopes[14][0]/1),
#                            (1,     0,      0,  ed['N'].isotopes[15][0]/1),
#                            (1,     0,      0,  ed['N'].isotopes[15][0]+ed['H'].isotopes[1][0]),
#                            (0,     1,      0,  ed['Ga'].isotopes[69][0]/3),
#                            (0,     1,      0,  ed['Ga'].isotopes[71][0]/3),
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
#                            (0,     1,      0,  ed['Ga'].isotopes[71][0]+ed['H'].isotopes[1][0]),
#                            (0,     0,      1,  ed['In'].isotopes[113][0]/2),
#                            (0,     0,      1,  ed['In'].isotopes[115][0]/2)
#                            ],
#                            dtype=[('N','i4'),('Ga','i4'),('In','i4'),('m2q','f4')] )

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
#glob_bg_param = ppd.fit_uncorr_bg(epos['m2q'],fit_roi=[3.5,6.5])
bg_rois=[[3.5,6.5],[90,110]]
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
        
for roi in bg_rois:
    xbox = np.array([roi[0],roi[0],roi[1],roi[1]])
    ybox = np.array([0.1,np.max(ys_sm)/10,np.max(ys_sm)/10,0.1])
    
    ax.fill(xbox,ybox, 'b', alpha=0.2)

        
plt.pause(0.1)




#### END BASIC ANALYSIS ####



#### START EXPLORATORY ANALYSIS ####




# Slice and dice the data in wall_time
idxs_list = []
CHUNK_SIZE = 32000
s_idx = 0
while s_idx<epos.size:
    e_idx = np.min([epos.size,s_idx+CHUNK_SIZE])
    idxs_list.append([s_idx,e_idx])
    s_idx = e_idx
    
# Count and compositions
csr = np.full(len(idxs_list),-1.0)
Ga_comp = np.full(len(idxs_list),-1.0)
Ga_comp_std = np.full(len(idxs_list),-1.0)
Ga_comp_glob = np.full(len(idxs_list),-1.0)
Ga_comp_std_glob = np.full(len(idxs_list),-1.0)

keys = list(pk_data.dtype.fields.keys())
keys.remove('m2q')
Ga_idx = keys.index('Ga')
for loop_idx, idxs in enumerate(idxs_list):

    sub_epos = epos[idxs[0]:idxs[1]]
    
    plotting_stuff.plot_histo(sub_epos['m2q'],321,user_xlim=[0,275],user_bin_width=0.1,user_label=loop_idx)
    plt.waitforbuttonpress()    
#    r = np.sqrt(sub_epos['x_det']**2+sub_epos['y_det']**2)
#    sub_idxs = np.nonzero(r>=0)
#    cts = ppd.do_counting(sub_epos[sub_idxs],pk_params,glob_bg_param)
#    
#    tot_bg_ct = epos['m2q'][(epos['m2q']>=80) & (epos['m2q']<=120)].size
#    sub_bg_ct = sub_epos['m2q'][(sub_epos['m2q']>=80) & (sub_epos['m2q']<=120)].size
#    
#    loc_bg_param = glob_bg_param*sub_bg_ct/tot_bg_ct
    glob_bg_param_chunk = ppd.get_glob_bg(sub_epos['m2q'],rois=bg_rois)
    
    cts = ppd.do_counting(sub_epos,pk_params,glob_bg_param_chunk)
    Ga_comp[loop_idx] = ppd.do_composition(pk_data,cts)[0][0][Ga_idx]
    Ga_comp_std[loop_idx] = ppd.do_composition(pk_data,cts)[0][1][Ga_idx]
    
    Ga_comp_glob[loop_idx] = ppd.do_composition(pk_data,cts)[2][0][Ga_idx]
    Ga_comp_std_glob[loop_idx] = ppd.do_composition(pk_data,cts)[2][1][Ga_idx]
    
    
    
    csr[loop_idx] = np.sum(cts['total'][Ga2p_idxs])/np.sum(cts['total'][Ga1p_idxs])
    print('Total Ranged Ions: '+str(np.sum(cts['total'])))
    print('Total Ranged Global Background Ions: '+str(np.sum(cts['global_bg'])))
    print('Total Ions: '+str(sub_epos.size))

    compositions = ppd.do_composition(pk_data,cts)
    ppd.pretty_print_compositions(compositions,pk_data)
#    
#    
#    print(sub_epos['m2q'][(sub_epos['m2q']>=80) & (sub_epos['m2q']<=120)].size)
#    print(np.sum(cts['total']))
#    print(sub_epos['m2q'][(sub_epos['m2q']>=80) & (sub_epos['m2q']<=120)].size/np.sum(cts['total']))
#    
    
    

fig = plt.figure(num=101)
fig.clear()
ax = fig.gca()
ax.errorbar((np.arange(Ga_comp.size)+0.5)*CHUNK_SIZE,Ga_comp,yerr=Ga_comp_std,fmt='.',capsize=4,label='chunk based (wall time)')
ax.errorbar((np.arange(Ga_comp.size)+0.5)*CHUNK_SIZE,Ga_comp_glob,yerr=Ga_comp_std_glob,fmt='.',capsize=4,label='chunk based (wall time)_glob')
ax.set(xlabel='ion idx', ylabel='Ga %')
ax.legend()





fig = plt.figure(num=102)
fig.clear()
ax = fig.gca()
ax.errorbar(csr,Ga_comp,yerr=Ga_comp_std,fmt='.',capsize=4,label='by wall time')
    
ax.plot([np.min(csr),np.max(csr)],[0.5,0.5],'k--')

ax.set(xlabel='CSR', ylabel='Ga %', ylim=[0, 1], xlim=[1e-2,2])

ax.set_title('by wall time')
ax.set_xscale('log')

ax.legend()
ax.grid()
fig.tight_layout()
fig.canvas.manager.window.raise_()









import colorcet as cc

def extents(f):
    delta = f[1] - f[0]
    return [f[0] - delta/2, f[-1] + delta/2]

def create_histogram(xs,ys,x_roi=None,y_roi=None):
    num_x = 128
    num_y = num_x
    N,x_edges,y_edges = np.histogram2d(xs,ys,bins=[num_x,num_y],range=[x_roi,y_roi],density=False)
    return (N,x_edges,y_edges)


fig = plt.figure(num=301)
plt.clf()
ax = fig.gca()


sel_idxs = np.arange(epos.size)
sel_idxs = np.where((epos['m2q']>(0.9+0)) & (epos['m2q']<(1.2+0)))


pk_data

N,x_edges,y_edges = create_histogram(epos['x_det'][sel_idxs],epos['y_det'][sel_idxs],
                                     x_roi=[-35,35],y_roi=[-35,35])
#ax.imshow(np.log10(1+1*np.transpose(N)), aspect='auto', 
ax.imshow(np.transpose(N), aspect='auto', 
           extent=extents(x_edges) + extents(y_edges), origin='lower', cmap=cc.cm.CET_L8,
           interpolation='nearest')
ax.set_aspect('equal', 'box')

ax.set(xlabel='det_x')
ax.set(ylabel='det_y')




keys = list(pk_data.dtype.fields.keys())
keys.remove('m2q')

for k in keys:
    k_pks = np.where(pk_data[k]>0)[0]    
    sel_idxs = np.zeros(0,dtype='int64')
    for pk in k_pks:
        ev_idxs = np.where((epos['m2q']>pk_params['pre_rng'][pk]) & (epos['m2q']<pk_params['post_rng'][pk]))[0]
        sel_idxs = np.concatenate((sel_idxs,ev_idxs))
    
    fig = plt.figure()
    plt.clf()
    ax = fig.gca()
        
    N,x_edges,y_edges = create_histogram(epos['x_det'][sel_idxs],epos['y_det'][sel_idxs],
                                         x_roi=[-35,35],y_roi=[-35,35])
    #ax.imshow(np.log10(1+1*np.transpose(N)), aspect='auto', 
    ax.imshow(np.transpose(N), aspect='auto', 
               extent=extents(x_edges) + extents(y_edges), origin='lower', cmap=cc.cm.CET_L8,
               interpolation='nearest')
    ax.set_aspect('equal', 'box')
    
    ax.set(xlabel='det_x')
    ax.set(ylabel='det_y')
    ax.set_title(k)
        
    




# All ranged ions

keys = list(pk_data.dtype.fields.keys())
keys.remove('m2q')

sel_idxs = np.zeros(0,dtype='int64')    

for k in keys:
    k_pks = np.where(pk_data[k]>0)[0]    
    for pk in k_pks:
        ev_idxs = np.where((epos['m2q']>pk_params['pre_rng'][pk]) & (epos['m2q']<pk_params['post_rng'][pk]))[0]
        sel_idxs = np.concatenate((sel_idxs,ev_idxs))
    
fig = plt.figure(321)
plt.clf()
ax = fig.gca()
    
N,x_edges,y_edges = create_histogram(0+epos['x_det'][sel_idxs],0+epos['y_det'][sel_idxs],
                                     x_roi=[-35,35],y_roi=[-35,35])
#ax.imshow(np.log10(1+1*np.transpose(N)), aspect='auto', 
met = N

met[met==0] = 100

met = 1/(met+3)


ax.imshow(np.transpose(met), aspect='auto', 
           extent=extents(x_edges) + extents(y_edges), origin='lower', cmap=cc.cm.CET_R2,
           interpolation='nearest')
ax.set_aspect('equal', 'box')

ax.set(xlabel='det_x')
ax.set(ylabel='det_y')
    

cx = 0.5*(x_edges[1:]+x_edges[:-1])
cy = 0.5*(y_edges[1:]+y_edges[:-1])

CX,CY = np.meshgrid(cx,cy)




  



def mean_shift(xs,ys):
    radius = 5
    
    x_curr = 0
    y_curr = 0
    
    N_LOOPS = 64
    
    xi = np.zeros(N_LOOPS)
    yi = np.zeros(N_LOOPS)
    
    for i in np.arange(N_LOOPS):
        x_prev = x_curr
        y_prev = y_curr
        
        idxs = np.where(((xs-x_curr)**2+(ys-y_curr)**2) <= radius**2)
#        print(idxs)
        
        x_q = np.mean(xs[idxs])
        y_q = np.mean(ys[idxs])
           
        dx = x_q-x_prev
        dy = y_q-y_prev
        
        x_curr = x_prev-dx
        y_curr = y_prev-dy
        
        if np.sqrt((x_curr-x_prev)**2 + (y_curr-y_prev)**2) < (radius*1e-2):
#            print('iter  B #',i,'    ',x_curr,y_curr)
#            print(i)
            break
#        else:
#            print('iter NB #',i,'    ',x_curr,y_curr)
#        print('iter #',i,'    ',x_curr,y_curr)
        xi[i] = x_curr
        yi[i] = y_curr
              

    return xi[:i], yi[:i]



xc,yc = mean_shift(epos['x_det'][sel_idxs],epos['y_det'][sel_idxs])

ax.plot(xc,yc,'-o')








# Slice and dice the data in detector space (polar)
idxs_list = []
#STEP = 2

r = np.sqrt(np.square(epos['x_det']-xc[-1])+np.square(epos['y_det']-yc[-1]))

R_DET_MAX = 28
R_C = np.sqrt(xc[-1]**2+yc[-1]**2)

R_MAX = R_DET_MAX-R_C

r_edges = np.sqrt(np.linspace(0, R_MAX**2, 6))
r_centers = (r_edges[:-1]+r_edges[1:])/2

for i in np.arange(r_edges.size-1):
    idxs = np.where((r>r_edges[i]) & (r<=r_edges[i+1]))[0]
    idxs_list.append(idxs)    



#for rq in np.arange(2,R_MAX,STEP):
#    idxs = np.where((r>rq) & (r<=(rq+STEP)))[0]
#    idxs_list.append(idxs)    

# Count and compositions
csr = np.full(len(idxs_list),-1.0)
Ga_comp = np.full(len(idxs_list),-1.0)
Ga_comp_std = np.full(len(idxs_list),-1.0)
Ga_comp_glob = np.full(len(idxs_list),-1.0)
Ga_comp_std_glob = np.full(len(idxs_list),-1.0)


Ga_idx = keys.index('Ga')
for loop_idx, idxs in enumerate(idxs_list):
    sub_epos = epos[idxs]
    
#    tot_bg_ct = epos['m2q'][(epos['m2q']>=80) & (epos['m2q']<=120)].size
#    sub_bg_ct = sub_epos['m2q'][(sub_epos['m2q']>=80) & (sub_epos['m2q']<=120)].size
    
#    loc_bg_param = glob_bg_param*sub_bg_ct/tot_bg_ct
    
    glob_bg_param_chunk = ppd.get_glob_bg(sub_epos['m2q'],rois=bg_rois)

    
    cts = ppd.do_counting(sub_epos,pk_params,glob_bg_param_chunk)
    
    csr[loop_idx] = np.sum(cts['total'][Ga2p_idxs])/np.sum(cts['total'][Ga1p_idxs])
    Ga_comp[loop_idx] = ppd.do_composition(pk_data,cts)[0][0][Ga_idx]
    Ga_comp_std[loop_idx] = ppd.do_composition(pk_data,cts)[0][1][Ga_idx]

    Ga_comp_glob[loop_idx] = ppd.do_composition(pk_data,cts)[2][0][Ga_idx]
    Ga_comp_std_glob[loop_idx] = ppd.do_composition(pk_data,cts)[2][1][Ga_idx]
    
    compositions = ppd.do_composition(pk_data,cts)
    ppd.pretty_print_compositions(compositions,pk_data)
    print('COUNTS IN CHUNK: ',np.sum(cts['total']))



fig = plt.figure(num=201)
fig.clear()
ax = fig.gca()
#ax.errorbar((np.arange(Ga_comp.size)+0.5)*STEP,Ga_comp,yerr=Ga_comp_std,fmt='.',capsize=4,label='Ga %')
#ax.errorbar((np.arange(Ga_comp.size)+0.5)*STEP,Ga_comp_glob,yerr=Ga_comp_std_glob,fmt='.',capsize=4,label='glob')
ax.errorbar(r_centers,Ga_comp,yerr=Ga_comp_std,fmt='.',capsize=4,label='Ga %')
ax.errorbar(r_centers,Ga_comp_glob,yerr=Ga_comp_std_glob,fmt='.',capsize=4,label='glob')




ax.set(xlabel='radius', ylabel='Ga %')

ax_twin= ax.twinx()
#ax_twin.plot((np.arange(Ga_comp.size)+0.5)*STEP,csr,'s',color='r',label='Ga CSR')
ax_twin.plot(r_centers,csr,'s',color='r',label='Ga CSR')

ax.legend()
ax_twin.legend(loc=7)
ax_twin.set(ylabel='Ga CSR')


fig = plt.figure(num=202)
fig.clear()
ax = fig.gca()

ax.errorbar(csr,Ga_comp,yerr=Ga_comp_std,fmt='.',capsize=4,label='det based (radial)')
ax.plot([np.min(csr),np.max(csr)],[0.5,0.5],'k--')

ax.set(xlabel='CSR', ylabel='Ga %', ylim=[0, 1], xlim=[1e-2,2])
ax.legend()
ax.set_title('by radius')
ax.set_xscale('log')
ax.grid()
fig.tight_layout()
fig.canvas.manager.window.raise_()


for i in np.arange(csr.size):
    print(csr[i],'\t',Ga_comp[i],'\t',Ga_comp_std[i])










# chop data up by Radius AND time
# Aim for 50 k ions per time chunk
# Aim for 5 radial chunks


es2cs = lambda es : (es[:-1]+es[1:])/2.0
    
#idxs_list = []
    
N_time_chunks = int(np.floor(epos.size/256000))
#N_time_chunks = 4

N_events_per_time_chunk = epos.size//N_time_chunks
time_chunk_edges = np.arange(N_time_chunks+1)*N_events_per_time_chunk
time_chunk_centers = es2cs(time_chunk_edges)


R_DET_MAX = 28
R_C = np.sqrt(xc[-1]**2+yc[-1]**2)
R_MAX = R_DET_MAX-R_C

N_ann_chunks = 3
r_edges = np.sqrt(np.linspace(0, R_MAX**2, N_ann_chunks+1))
r_centers = es2cs(r_edges)




csr = np.full([N_time_chunks,N_ann_chunks],-1.0)
Ga_comp = np.full([N_time_chunks,N_ann_chunks],-1.0)
Ga_comp_std = np.full([N_time_chunks,N_ann_chunks],-1.0)
Ga_comp_glob = np.full([N_time_chunks,N_ann_chunks],-1.0)
Ga_comp_std_glob = np.full([N_time_chunks,N_ann_chunks],-1.0)
tot_cts = np.full([N_time_chunks,N_ann_chunks],-1.0)

for t_idx in np.arange(N_time_chunks):
    sub_epos = epos[time_chunk_edges[t_idx]:time_chunk_edges[t_idx+1]]
    r = np.sqrt(np.square(sub_epos['x_det']-xc[-1])+np.square(sub_epos['y_det']-yc[-1]))
    for a_idx in np.arange(N_ann_chunks):
        
        idxs = np.where((r>r_edges[a_idx]) & (r<=r_edges[a_idx+1]))[0]
        
        subsubepos = sub_epos[idxs]
        
        glob_bg_param_chunk = ppd.get_glob_bg(subsubepos['m2q'],rois=bg_rois)

        cts = ppd.do_counting(subsubepos,pk_params,glob_bg_param_chunk)
        
        csr[t_idx,a_idx] = np.sum(cts['total'][Ga2p_idxs])/np.sum(cts['total'][Ga1p_idxs])
        Ga_comp[t_idx,a_idx] = ppd.do_composition(pk_data,cts)[0][0][Ga_idx]
        Ga_comp_std[t_idx,a_idx] = ppd.do_composition(pk_data,cts)[0][1][Ga_idx]
    
        Ga_comp_glob[t_idx,a_idx] = ppd.do_composition(pk_data,cts)[2][0][Ga_idx]
        Ga_comp_std_glob[t_idx,a_idx] = ppd.do_composition(pk_data,cts)[2][1][Ga_idx]
        
        compositions = ppd.do_composition(pk_data,cts)
        ppd.pretty_print_compositions(compositions,pk_data)
        print('COUNTS IN CHUNK: ',np.sum(cts['total']))
        tot_cts[t_idx,a_idx] = np.sum(cts['total'])
        


fig = plt.figure(num=402)
fig.clear()
ax = fig.gca()

#ax.errorbar(csr.flatten(),Ga_comp.flatten(),yerr=Ga_comp_std.flatten(),fmt='.',capsize=4,label='det based (radial)')
ax.errorbar(csr.flatten(),Ga_comp_glob.flatten(),yerr=Ga_comp_std_glob.flatten(),fmt='.',capsize=4,label='det based (radial)')
ax.plot([np.min(csr),np.max(csr)],[0.5,0.5],'k--')

ax.set(xlabel='CSR', ylabel='Ga %', ylim=[0, 1], xlim=[5e-3,5])
ax.legend()
ax.set_title('by radius')
ax.set_xscale('log')
ax.grid()       
fig.tight_layout()
fig.canvas.manager.window.raise_()




fig = plt.figure(num=405)
fig.clear()
ax = fig.gca()
ax.imshow(csr.T,
          extent=extents(time_chunk_edges) + [r_edges[0], r_edges[-1]+np.diff(r_edges[-2:])],
          aspect='auto',
           origin='lower')
ax.set(xlabel='ev idx')
ax.set(ylabel='rad')



for i in np.arange(csr.size):
    print(csr.flatten()[i],'\t',Ga_comp_glob.flatten()[i],'\t',Ga_comp_std_glob.flatten()[i])





