# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# standard imports 
import numpy as np
import matplotlib.pyplot as plt

# custom imports
import apt_fileio
import plotting_stuff
import initElements_P3

import peak_param_determination as ppd

from histogram_functions import bin_dat

# Read in data
#fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\180821_GaN_A71\R20_07094-v03.epos"
fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\R20_07148-v01.epos"
#fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\R20_07248-v01.epos"
#fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\R20_07249-v01.epos"
#fn = r"Q:\NIST_Projects\EUV_APT_IMS\BWC\GaN epos files\R20_07250-v01.epos"
new_fn = fn[:-5]+'_vbm_corr.epos'
epos = apt_fileio.read_epos_numpy(new_fn)
#epos = epos[epos.size//2:-1]

# Plot m2q vs event index and show the current ROI selection
#roi_event_idxs = np.arange(1000,epos.size-1000)
roi_event_idxs = np.arange(epos.size)
ax = plotting_stuff.plot_m2q_vs_time(epos['m2q'],epos,1)
ax.plot(roi_event_idxs[0]*np.ones(2),[0,1200],'--k')
ax.plot(roi_event_idxs[-1]*np.ones(2),[0,1200],'--k')
ax.set_title('roi selected to start analysis')
epos = epos[roi_event_idxs]

# Compute some extra information from epos information
wall_time = np.cumsum(epos['pslep'])/10000.0
pulse_idx = np.arange(0,epos.size)
isSingle = np.nonzero(epos['ipp'] == 1)

# Define peaks to range
ed = initElements_P3.initElements()
#                            N      Ga  Da

# ADD 16, 30, 71.9, also add GaN3^2+
pk_data =   np.array(    [  (1,     0,  ed['N'].isotopes[14][0]/2),
                            (1,     0,  ed['N'].isotopes[14][0]/1),
                            (1,     0,  ed['N'].isotopes[15][0]/1),
                            (1,     0,  ed['N'].isotopes[15][0]+ed['H'].isotopes[1][0]),
                            (0,     1,  ed['Ga'].isotopes[69][0]/3),
                            (0,     1,  ed['Ga'].isotopes[71][0]/3),
                            (2,     0,  ed['N'].isotopes[14][0]*2),
                            (2,     0,  ed['N'].isotopes[14][0]+ed['N'].isotopes[15][0]),
                            (2,     0,  ed['N'].isotopes[14][0]+ed['N'].isotopes[15][0]+ed['H'].isotopes[1][0]),
                            (0,     1,  ed['Ga'].isotopes[69][0]/2),
                            (0,     1,  ed['Ga'].isotopes[71][0]/2),
                            (1,     1,  (ed['N'].isotopes[14][0] + ed['Ga'].isotopes[69][0])/2),
                            (3,     0,  ed['N'].isotopes[14][0]*3),
                            (1,     1,  (ed['N'].isotopes[14][0] + ed['Ga'].isotopes[71][0])/2),
                            (0,     1,  ed['Ga'].isotopes[69][0]),
                            (0,     1,  ed['Ga'].isotopes[71][0]),
                            (0,     1,  ed['Ga'].isotopes[71][0]+ed['H'].isotopes[1][0]),
                            (3,     1,  (ed['Ga'].isotopes[69][0]+3*ed['N'].isotopes[14][0])/2),
                            (3,     1,  (ed['Ga'].isotopes[71][0]+3*ed['N'].isotopes[14][0])/2),
                            (0,     0,  ed['Mg'].isotopes[24][0]/2)
                            ],
                            dtype=[('N_at_ct','i4'), 
                                   ('Ga_at_ct','i4'),
                                   ('m2q','f4')] )

#
#pk_data =   np.array(    [  (1,     0,  ed['N'].isotopes[14][0]/2),
#                            (1,     0,  ed['N'].isotopes[14][0]/1),
#                            (1,     0,  ed['N'].isotopes[15][0]/1),
#                            (0,     1,  ed['Ga'].isotopes[69][0]/3),
#                            (0,     1,  ed['Ga'].isotopes[71][0]/3),
#                            (2,     0,  ed['N'].isotopes[14][0]*2),
#                            (2,     0,  ed['N'].isotopes[14][0]+ed['N'].isotopes[15][0]),
#                            (0,     1,  ed['Ga'].isotopes[69][0]/2),
#                            (0,     1,  ed['Ga'].isotopes[71][0]/2),
#                            (1,     1,  (ed['N'].isotopes[14][0] + ed['Ga'].isotopes[69][0])/2),
#                            (3,     0,  ed['N'].isotopes[14][0]*3),
#                            (1,     1,  (ed['N'].isotopes[14][0] + ed['Ga'].isotopes[71][0])/2),
#                            (0,     1,  ed['Ga'].isotopes[69][0]),
#                            (0,     1,  ed['Ga'].isotopes[71][0])],
#                            dtype=[('N_at_ct','i4'), 
#                                   ('Ga_at_ct','i4'),
#                                   ('m2q','f4')] )

# Range the peaks
pk_params = ppd.get_peak_ranges(epos,pk_data['m2q'])
    
# Determine the global background
glob_bg_param = ppd.fit_uncorr_bg(epos['m2q'])


# Count the peaks, local bg, and global bg
cts = ppd.do_counting(epos,pk_params,glob_bg_param)

# Calculate compositions
compositions = ppd.do_composition(pk_data,cts)
print('compositions [N, Ga]: (no bg corr, local bg corr, global bg corr)')
print(compositions)
print('Total Ranged Ions: '+str(np.sum(cts['total'])))
print('Total Ions: '+str(epos.size))
print('Overall CSR: '+str((cts['total'][9]+cts['total'][10])/(cts['total'][14]+cts['total'][15])))


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

for pk_param in pk_params:
    ax.plot(np.array([1,1])*pk_param['pre_rng'] ,np.array([0.5,(pk_param['amp']+pk_param['off'])]),'k--')
    ax.plot(np.array([1,1])*pk_param['post_rng'] ,np.array([0.5,(pk_param['amp']+pk_param['off'])]),'k--')
    ax.plot(np.array([1,1])*pk_param['pre_bg_rng'] ,np.array([0.5,(pk_param['amp']+pk_param['off'])]),'r--')
    ax.plot(np.array([1,1])*pk_param['post_bg_rng'] ,np.array([0.5,(pk_param['amp']+pk_param['off'])]),'r--')
    
    ax.plot(np.array([pk_param['pre_bg_rng'],pk_param['post_bg_rng']]) ,np.ones(2)*pk_param['loc_bg'],'g--')
    
plt.pause(0.1)



# Slice and dice the data in wall_time
idxs_list = []
STEP = 30000
s_idx = 0
while s_idx<epos.size:
    e_idx = np.min([epos.size,s_idx+STEP])
    idxs_list.append(np.arange(s_idx,e_idx))
    s_idx = e_idx
    
# Count and compositions
csr = np.full(len(idxs_list),-1.0)
Ga_comp = np.full(len(idxs_list),-1.0)

for loop_idx, idxs in enumerate(idxs_list):
    cts = ppd.do_counting(epos[idxs],pk_params,glob_bg_param)
    csr[loop_idx] = (cts['total'][9]+cts['total'][10])/(cts['total'][14]+cts['total'][15])

    Ga_comp[loop_idx] = ppd.do_composition(pk_data,cts)[0][1]


fig = plt.figure(num=101)
fig.clear()
ax = fig.gca()

ax.plot(csr,Ga_comp,'.',label='time based')
#ax.plot([0,2.25],[0.5,0.5],'k--')

ax.set(xlabel='CSR', ylabel='Ga %', ylim=[0, 1])
#ax.grid()
#fig.tight_layout()
#fig.canvas.manager.window.raise_()
#ax.legend()





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

for loop_idx, idxs in enumerate(idxs_list):
    cts = ppd.do_counting(epos[idxs],pk_params,glob_bg_param)
    csr[loop_idx] = (cts['total'][9]+cts['total'][10])/(cts['total'][14]+cts['total'][15])
    Ga_comp[loop_idx] = ppd.do_composition(pk_data,cts)[0][1]


ax.plot(csr,Ga_comp,'.',label='det based (radial)')
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

