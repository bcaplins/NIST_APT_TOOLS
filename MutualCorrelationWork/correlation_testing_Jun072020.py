# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 14:53:20 2020

@author: capli
"""




import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc



def create_histogram(xs, ys, x_roi=None, delta_x=0.1, y_roi=None, delta_y=0.1):
    """Create a 2d histogram of the data, specifying the bin intensity, region
    of interest (on the y-axis), and the spacing of the y bins"""
    # even number
    num_x = int(np.ceil((x_roi[1]-x_roi[0])/delta_x))
    num_y = int(np.ceil((y_roi[1]-y_roi[0])/delta_y))

    return np.histogram2d(xs, ys, bins=[num_x, num_y],
                          range=[x_roi, y_roi],
                          density=False)

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


def corrhist(epos):
    dat = epos['tof']
    roi = [0, 2500]
    delta = 1
    
    N = int(np.ceil((roi[1]-roi[0])/delta))
    
    corrhist = np.zeros([N,N], dtype=int)
    
    multi_idxs = np.where(epos['ipp']>1)[0]
    
    for multi_idx in multi_idxs:
        n_hits = epos['ipp'][multi_idx]
        cluster = dat[multi_idx:multi_idx+n_hits]
        
        idx1 = -1
        idx2 = -1
        for i in range(n_hits):
            for j in range(i+1,n_hits):
                idx1 = int(np.floor(cluster[i]/delta))
                idx2 = int(np.floor(cluster[j]/delta))
                if idx1 < N and idx2 < N:
                    corrhist[idx1,idx2] += 1
    
    edges = np.arange(roi[0],roi[1]+delta,delta)
    assert edges.size-1 == N
    
    return (edges, corrhist+corrhist.T-np.diag(np.diag(corrhist)))
                
def orthocorrhist(epos):
    dat = epos['tof']
    roi_x = [0, 10000]
    roi_y = [0, 500]
    delta_x = 2
    delta_y = 0.25
    
    Nx = int(np.ceil((roi_x[1]-roi_x[0])/delta_x))
    Ny = int(np.ceil((roi_y[1]-roi_y[0])/delta_y))
    
    corrhist = np.zeros([Nx,Ny], dtype=int)
    
    multi_idxs = np.where(epos['ipp']>1)[0]
    
    for multi_idx in multi_idxs:
        n_hits = epos['ipp'][multi_idx]
        cluster = dat[multi_idx:multi_idx+n_hits]
        
        idx1 = -1
        idx2 = -1
        for i in range(n_hits):
            for j in range(i+1,n_hits):
                new_y = np.abs((cluster[i]-cluster[j])/np.sqrt(2))
                new_x = np.abs((cluster[i]+cluster[j])/np.sqrt(2))
                
                idx1 = int(np.floor(new_x/delta_x))
                idx2 = int(np.floor(new_y/delta_y))
                
                if idx1 < Nx and idx2 < Ny:
                    corrhist[idx1,idx2] += 1
    
    x_edges = np.arange(roi_x[0],roi_x[1]+delta_x,delta_x)
    y_edges = np.arange(roi_y[0],roi_y[1]+delta_y,delta_y)
    
    return (x_edges, y_edges, corrhist)


# standard imports 
import numpy as np
import matplotlib.pyplot as plt


import peak_param_determination as ppd
from histogram_functions import bin_dat

plt.close('all')

fn = r'C:\Users\capli\Google Drive\NIST\pos_and_epos_files\GaN_manuscript\R20_07148-v01_vbm_corr.epos'
#fn = r'C:\Users\capli\Google Drive\NIST\pos_and_epos_files\R45_00504-v56_mod.epos'
#fn = r'C:\Users\capli\Downloads\R45_04472-v03.epos'
#fn = r'C:\Users\capli\Downloads\JAVASS_R44_03187-v01.epos'
#fn =  r'C:\Users\capli\Downloads\R45_04472-v03.epos'

import apt_fileio
    
epos = apt_fileio.read_epos_numpy(fn)

#tof_corr = np.sqrt(epos['m2q'])
#
#tof_corr = tof_corr*np.mean(epos['tof'])/np.mean(tof_corr)
#
#epos['tof'] = tof_corr


#epos = epos[2550000:]

# Voltage and bowl correct ToF data
from voltage_and_bowl import do_voltage_and_bowl
import voltage_and_bowl

p_volt = np.array([])
p_bowl = np.array([])
tof_corr, p_volt, p_bowl = do_voltage_and_bowl(epos,p_volt,p_bowl)        



tof_vcorr = voltage_and_bowl.mod_full_voltage_correction(p_volt,epos['tof'],epos['v_dc'])
tof_corr = voltage_and_bowl.mod_geometric_bowl_correction(p_bowl,tof_vcorr,epos['x_det'],epos['y_det'])


epos_vb = epos.copy()
epos_vb['tof'] = tof_corr.copy()



#def bah_humbug(p_in,tof,x_det,y_det,DEL):
#    SCALES = np.array([1e2, 1e-4, 1e-4, 1e-3])
#    
#    p_in = p_in*SCALES
#
#    r2 = x_det**2+y_det**2
#    
#    
#    Lr_hat = 1/     \
#        (np.sqrt(1+r2/p_in[0]**2)   \
#        *(1+p_in[1]*x_det+p_in[2]*y_det+p_in[3]*(r2/30**2)**2))
#    
#    
##    DEL = 500
#    new_tof = (Lr_hat-1)*DEL+tof
##
##    com0 = np.mean(tof)
##    com = np.mean(new_tof)
##    new_tof = (com0/com)*new_tof   
#            
#    return new_tof

#for D in np.arange(0,1200,100):
#    
#
DIFFS = np.arange(380,421,20.0)
ERRS = np.zeros_like(DIFFS)

for loop_idx in np.arange(DIFFS.size):
        
    
    def calc_deltas(epos,tof):
        mult_idxs = np.where(epos['ipp']>1)[0]
        
        deltas = np.zeros_like(tof)
        
        import scipy.stats.mstats
    
        for idx in mult_idxs:
            num_ions = epos['ipp'][idx]
            deltas[idx:idx+num_ions] = np.mean(tof[idx:idx+num_ions])-360
#            deltas[idx:idx+num_ions] = np.sqrt(np.sum(tof[idx:idx+num_ions]**2))/np.sqrt(num_ions)-410
    #        deltas[idx:idx+num_ions] = 2*np.sum(tof[idx:idx+num_ions])/np.prod(tof[idx:idx+num_ions])-360
    #        deltas[idx:idx+num_ions] = scipy.stats.mstats.gmean(tof[idx:idx+num_ions])-300
    #        deltas[idx:idx+num_ions] = np.sqrt(tof[idx:idx+num_ions]**2)/num_ions
    #        
    #        if num_ions == 2:
    #            deltas[idx:idx+num_ions] = np.mean(tof[idx:idx+num_ions])
    #        else:
    #            deltas[idx:idx+num_ions] = np.median(tof[idx:idx+num_ions])
    #        
            
        return deltas
    
    DEL = calc_deltas(epos,epos['tof'])
    tof_vcorr = voltage_and_bowl.mod_full_voltage_correction(p_volt,epos['tof']-DEL,epos['v_dc'])+DEL
    #tof_vcorr = voltage_and_bowl.mod_full_voltage_correction(p_volt,epos['tof'],epos['v_dc'])
    #tof_corr = bah_humbug(p_bowl,tof_vcorr,epos['x_det'],epos['y_det'],D)
    #DEL = 2500
    DEL = calc_deltas(epos,tof_vcorr)
    tof_corr = voltage_and_bowl.mod_geometric_bowl_correction(p_bowl,tof_vcorr-DEL,epos['x_det'],epos['y_det'])+DEL
    
    epos_vb_mod = epos.copy()
    epos_vb_mod['tof'] = tof_corr.copy()
    
    epos_v_mod = epos.copy()
    epos_v_mod['tof'] = tof_vcorr.copy()
    
    
    import histogram_functions
    
    
#    edges, ch = corrhist(epos)
#    fig1 = plt.figure(num=1)
#    fig1.clf()
#    ax1 = fig1.gca()    
#    plot_2d_histo(ax1, ch, edges, edges)
#    ax1.axis('equal')
#    ax1.set_xlabel('ns')
#    ax1.set_ylabel('ns')
#    #
#    
#    edges, ch = corrhist(epos_vb)
#    fig2 = plt.figure(num=2)
#    fig2.clf()
#    ax2 = fig2.gca()   
#    plot_2d_histo(ax2, ch, edges, edges)
#    ax2.axis('equal')
#    ax2.set_xlabel('ns')
#    ax2.set_ylabel('ns')
#    
    edges, ch = corrhist(epos_vb_mod)
    fig3 = plt.figure(num=333)
    fig3.clf()
    ax3 = fig3.gca()   
    plot_2d_histo(ax3, ch, edges, edges)
    ax3.axis('equal')
    ax3.set_xlabel('ns')
    ax3.set_ylabel('ns')
        
    
    fig4 = plt.figure(num=4)
    #    plt.clf()    
    ax4 = fig4.gca()
    
    edgesx, edgesy, h = orthocorrhist(epos_vb_mod)
       
    delta = int(np.round(np.mean(np.diff(edgesx))))
    coord = histogram_functions.edges_to_centers(edgesy)[0]
    
#    roi = [2000,5000]
    roi = [0000,10000]
    #roi = [3900,4100]
#    yyy = np.sum(h[int(roi[0]/delta):int(roi[1]/delta),:],axis=0)/np.diff(roi)
    yyy = np.sum(h[int(roi[0]/delta):int(roi[1]/delta),:],axis=0)/np.diff(roi)
    ax4.plot(coord,yyy,
             label='roi='+str(roi),
             lw=1)
        
    plt.pause(1)

    ERRS[loop_idx] = np.sum(yyy**2)




#    
#    
#    #h, edgesx, edgesy = create_histogram(sus, dts, x_roi=[0,10000], delta_x=2, y_roi=[0,1000], delta_y=1)
fig5 = plt.figure(num=5)    
plot_2d_histo(fig5.gca(), h, edgesx, edgesy)
##    
##    ax4.set_xlabel('(t1+t2)/sqrt(2) in ns')
##    ax4.set_ylabel('(t1-t2)/sqrt(2) in ns')
#    ax4.set_xlim(0,500)
##    ax4.set_ylim(0,50)
#    ax4.set_title(str(D))
#    plt.pause(1)
#
#





fig66 = plt.figure(num=66)
plt.clf()    
dts = np.abs(epos['tof'][idxs]-epos['tof'][idxs+1])
#    sus = np.sqrt(tof_corr[idxs]**2+tof_corr[idxs+1]**2)
#    sus = np.fmax(tof_corr[idxs],tof_corr[idxs+1])
sus = (epos['tof'][idxs]+epos['tof'][idxs+1])/np.sqrt(2)
plt.plot(sus,dts,'.',ms=1,alpha=.5)
#    fig66.gca().axis('equal')
fig66.gca().set_xlim(0,7000)
fig66.gca().set_ylim(-100, 800)
    






    


#
#ref_fn = r"C:\Users\capli\Google Drive\NIST\pos_and_epos_files\GaN_manuscript\R20_07094-v03.epos"
#ref_epos = apt_fileio.read_epos_numpy(ref_fn)
#
#import m2q_calib
#
#m2q_corr, p_m2q = m2q_calib.align_m2q_to_ref_m2q(ref_epos['m2q'],tof_corr)














epos_vb = epos.copy()

epos_vb['tof'] = tof_corr.copy()

import voltage_and_bowl

tof_vcorr = voltage_and_bowl.mod_full_voltage_correction(p_volt,epos['tof'],epos['v_dc'])
epos_v = epos.copy()
epos_v['tof'] = tof_vcorr.copy()

tof_bcorr = voltage_and_bowl.mod_geometric_bowl_correction(p_bowl,epos['tof'],epos['x_det'],epos['y_det'])
epos_b = epos.copy()
epos_b['tof'] = tof_bcorr.copy()

ROI = [0, None]

import histogram_functions

edges, ch = corrhist(epos)


fig1 = plt.figure(num=1)
plt.clf()    
ax1 = fig1.gca()

plot_2d_histo(ax1, ch, edges, edges)
ax1.axis('equal')

ax1.set_xlabel('ns')
ax1.set_ylabel('ns')

ax1.set_xlim(edges.min(),edges.max())
ax1.set_ylim(edges.min(),edges.max())

ax1.set_xlim(0,5000)
ax1.set_ylim(0,5000)



idxs = np.where(epos['ipp'] == 2)[0]

dts = np.abs(epos['tof'][idxs]-epos['tof'][idxs+1])/np.sqrt(2)
sus = (epos['tof'][idxs]+epos['tof'][idxs+1])/np.sqrt(2)



fig2 = plt.figure(num=2)
plt.clf()    
ax2 = fig2.gca()

edgesx, edgesy, h = orthocorrhist(epos)

#h, edgesx, edgesy = create_histogram(sus, dts, x_roi=[0,10000], delta_x=2, y_roi=[0,1000], delta_y=1)
plot_2d_histo(ax2, h, edgesx, edgesy)

ax2.set_xlabel('(t1+t2)/sqrt(2) in ns')
ax2.set_ylabel('(t1-t2)/sqrt(2) in ns')
ax2.set_xlim(0,7000)
ax2.set_ylim(0,700)




fig3 = plt.figure(33)
plt.clf()
ax3 = fig3.gca()

coord = histogram_functions.edges_to_centers(edgesy)[0]

delta = int(np.round(np.mean(np.diff(edgesx))))


myedges = [500, 550, 600, 700, 800, 900, 1000, 1200, 1500, 2000, 2500, 3000, 4000, 5000, 6000]

myedges = [300,500,700,1000,2000,3000,4000,5000]

for idx in np.arange(len(myedges)-1):
    roi = [myedges[idx],myedges[idx+1]]
    ax3.plot(coord,np.sum(h[int(roi[0]/delta):int(roi[1]/delta),:],axis=0)/np.diff(roi),
             label='roi='+str(roi),
             color=cc.cm.CET_L8( float(idx/(len(myedges)-1)) ) ,
             lw=3)
    
    
fig3.legend()
ax3.set_xlabel('(t1+t2)/sqrt(2) in ns')
ax3.set_ylabel('counts/time')



#plt.title('raw')
#fig1.gca().set_xlim(ROI[0],ROI[1])
#fig1.gca().set_ylim(ROI[0],ROI[1])
#
#ch = histogram_functions.corrhist(epos_v)
#fig2 = plt.figure(num=2)
#plt.clf()    
#plt.imshow(np.log2(1+ch))
#plt.title('volt')
#fig2.gca().set_xlim(ROI[0],ROI[1])
#fig2.gca().set_ylim(ROI[0],ROI[1])
#
#ch = histogram_functions.corrhist(epos_b)
#fig3 = plt.figure(num=3)
#plt.clf()    
#plt.imshow(np.log2(1+ch))
#plt.title('bowl')
#fig3.gca().set_xlim(ROI[0],ROI[1])
#fig3.gca().set_ylim(ROI[0],ROI[1])
#
#ch = histogram_functions.corrhist(epos_vb)
#fig4 = plt.figure(num=4)
#plt.clf()    
#plt.imshow(np.log10(1+ch))
#plt.title('v+b')
##    fig4.gca().set_xlim(ROI[0],ROI[1])
##    fig4.gca().set_ylim(ROI[0],ROI[1])



fig5 = plt.figure(num=5)
plt.clf()    
dts = np.abs(tof_corr[idxs]-tof_corr[idxs+1])
plt.hist(dts,bins=np.arange(0,2000,.5),label='deltaT')
plt.hist(tof_corr[np.r_[idxs,idxs+1]],bins=np.arange(0,2000,.5),label='since t0')









fig66 = plt.figure(num=66)
plt.clf()    
dts = np.abs(epos['tof'][idxs]-epos['tof'][idxs+1])
#    sus = np.sqrt(tof_corr[idxs]**2+tof_corr[idxs+1]**2)
#    sus = np.fmax(tof_corr[idxs],tof_corr[idxs+1])
sus = (epos['tof'][idxs]+epos['tof'][idxs+1])/np.sqrt(2)
plt.plot(sus,dts,'.',ms=1,alpha=.5)
#    fig66.gca().axis('equal')
fig66.gca().set_xlim(0,7000)
fig66.gca().set_ylim(-100, 800)
    




ER = epos.size/epos['pslep'].cumsum()[-1]

n = np.arange(15)
num_ev = np.zeros_like(n)
for idx in np.arange(1,n.size):
    num_ev[idx] = (epos['ipp']==n[idx]).sum()
    
num_ev[0] = epos['pslep'].cumsum()[-1]-np.sum(num_ev)

num_ev = num_ev/np.sum(num_ev)

def poisson(n, rate):
    return (rate**n)*np.exp(-rate)/scipy.special.factorial(n)
    
est_ev = poisson(n,ER)


fig321 = plt.figure(321)
fig321.clf()
ax321 = fig321.gca()

#est_ev[est_ev<1e-6] = 0

ax321.bar(n[1:6], n[1:6]*num_ev[1:6],  label='meas', alpha=0.5)
ax321.bar(n[1:6], n[1:6]*est_ev[1:6], label='est from ER', alpha=0.5)

ax321.legend()





















