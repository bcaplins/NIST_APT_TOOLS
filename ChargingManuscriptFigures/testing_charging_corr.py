import scaling_correction

# standard imports 
import numpy as np
import matplotlib.pyplot as plt

# custom imports
import apt_fileio
import plotting_stuff

from histogram_functions import bin_dat
import scipy.interpolate

def moving_average(a, n=3) :    
    # Moving average with reflection at the boundaries
    if 2*n+1>a.size:
        raise Exception('The kernel is too big!')
    kern = np.ones(2*n+1)/(2*n+1)
    return np.convolve(np.r_[a[n:0:-1],a,a[-2:-n-2:-1]],kern,'valid')


    



        
num_ions = 2**18

pk_m2qs = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 1, 14, 16, 14, 16, 14, 16, 14, 16, 14, 16, 32, 44, 60.])

pk_tofs = np.sqrt(pk_m2qs)*80
pk_idx = np.random.randint(0,pk_tofs.size,size=num_ions)
sigma = 0.25
tof_mu = pk_tofs[pk_idx]
tof_sigma = 0.25

q_tof = np.random.randn(tof_mu.size)*tof_sigma+tof_mu


bg_idxs = np.nonzero(tof_mu<0.01)[0]
num_bg = bg_idxs.size


q_tof[bg_idxs] = np.random.rand(num_bg)*np.max(q_tof)*1.3

noise = moving_average(np.random.randn(tof_mu.size),n=5000)

varying_tof = q_tof*(1+noise-np.mean(noise))




def mytofplot(tof,fig_idx):
        
    fig = plt.figure(num=fig_idx)
    fig.clf()
    ax = fig.gca()
    
    event_idx = np.arange(0,tof.size)    
    ax.plot(event_idx,tof,'.', 
            markersize=.1,
            marker=',',
            markeredgecolor='#1f77b4aa')
    ax.set(xlabel='event index', ylabel='ToF (ns)')
    
    ax.grid()
    
    fig.tight_layout()
    fig.canvas.manager.window.raise_()
    
    
    plt.pause(0.1)



mytofplot(q_tof,1)
mytofplot(varying_tof,2)



#import colorcet as cc
#      
#import matplotlib as mpl
#mpl.rc('image', cmap='gray')


def chi2(dat):
    n = dat.size
    f = np.sum(dat)    
    f_n = f/n    
    chi2 = np.sum(np.square(dat-f_n)/f_n)
    return chi2




tof_roi = [50,700]







def do_chi2(tof,tof_roi,N_lower,N_upper):
                
    tof1 = tof[0::2]
    tof2 = tof[1::2]
    
    N = N_upper-N_lower+1
    slicings = np.logspace(N_lower,N_upper,N,base=2)
    
    opt_res = np.zeros(N)
    time_res = np.zeros(N)

    for idx,cts_per_slice in enumerate(slicings):
        
        
        t_start = time.time()
        pointwise_scales,piecewise_scales = scaling_correction.get_all_scale_coeffs(tof1,
                                                                 m2q_roi=tof_roi ,
                                                                 cts_per_slice=cts_per_slice,
                                                                 max_scale=1.075,
                                                                 delta_ly=1e-4)
        t_end = time.time()
        time_res[idx] = t_end-t_start
        print('Total Time = ',time_res[idx])
                    
        # Compute corrected data
        tof_corr = tof2/pointwise_scales
            
        _, ys = bin_dat(tof_corr,isBinAligned=True,bin_width=0.1,user_roi=tof_roi)
    
        opt_res[idx] = chi2(ys)
        print(opt_res[idx])
        
    print(slicings)
    print(opt_res/np.max(opt_res))
    print(time_res)
    
    return (slicings,opt_res)


x,y = do_chi2(varying_tof,tof_roi,3,10)
plt.figure(32123)
plt.clf()
plt.plot(x,y,label='notinterp')





def do_chi2v2(tof,tof_roi,N_lower,N_upper):
                
    tof1 = tof[0::2]
    tof1_idxs = np.array(range(0,tof1.size))*2
    
    tof2 = tof[1::2]
    tof2_idxs = np.array(range(0,tof2.size))*2+1
    
    N = N_upper-N_lower+1
    slicings = np.logspace(N_lower,N_upper,N,base=2)
    
    opt_res = np.zeros(N)
    time_res = np.zeros(N)

    for idx,cts_per_slice in enumerate(slicings):
        
        
        t_start = time.time()
        pointwise_scales,piecewise_scales = scaling_correction.get_all_scale_coeffs(tof1,
                                                                 m2q_roi=tof_roi ,
                                                                 cts_per_slice=cts_per_slice,
                                                                 max_scale=1.075,
                                                                 delta_ly=1e-4)
        t_end = time.time()
        time_res[idx] = t_end-t_start
        print('Total Time = ',time_res[idx])
                    
        f = scipy.interpolate.interp1d(tof1_idxs,pointwise_scales,fill_value='extrapolate')
        
        # Compute corrected data
        tof_corr = tof2/f(tof2_idxs)
            
        _, ys = bin_dat(tof_corr,isBinAligned=True,bin_width=0.1,user_roi=tof_roi)
    
        opt_res[idx] = chi2(ys)
        print(opt_res[idx])
        
    print(slicings)
    print(opt_res/np.max(opt_res))
    print(time_res)
    
    return (slicings,opt_res)


x,y = do_chi2v2(varying_tof,tof_roi,3,10)
plt.plot(x,y,label='interp')







def do_chi2v3(tof,tof_roi,N_lower,N_upper):
                
    tof1 = tof[0::2]
    tof1_idxs = np.array(range(0,tof1.size))*2
    
    tof2 = tof[1::2]
    tof2_idxs = np.array(range(0,tof2.size))*2+1
    
    N = N_upper-N_lower+1
    slicings = np.logspace(N_lower,N_upper,N,base=2)
    
    opt_res = np.zeros(N)
    time_res = np.zeros(N)

    for idx,cts_per_slice in enumerate(slicings):
        
        
        t_start = time.time()
        pointwise_scales,piecewise_scales = scaling_correction.get_all_scale_coeffs(tof1,
                                                                 m2q_roi=tof_roi ,
                                                                 cts_per_slice=cts_per_slice,
                                                                 max_scale=1.075,
                                                                 delta_ly=1e-4)
        t_end = time.time()
        time_res[idx] = t_end-t_start
        print('Total Time = ',time_res[idx])
                    
        f = scipy.interpolate.interp1d(tof1_idxs,pointwise_scales,fill_value='extrapolate')
        
        # Compute corrected data
        tof_corr = tof1/f(tof1_idxs)
            
        _, ys = bin_dat(tof_corr,isBinAligned=True,bin_width=0.1,user_roi=tof_roi)
    
        opt_res[idx] = chi2(ys)
        print(opt_res[idx])
        
    print(slicings)
    print(opt_res/np.max(opt_res))
    print(time_res)
    
    return (slicings,opt_res)


x,y = do_chi2v3(varying_tof,tof_roi,3,10)
plt.plot(x,y,label='same')






x = [2**3, 2**10]
y = np.array([1,1])*chi2(bin_dat(q_tof[0::2],isBinAligned=True,bin_width=0.1,user_roi=tof_roi)[1])
plt.plot(x,y,label='ref1')

x = [2**3, 2**10]
y = np.array([1,1])*chi2(bin_dat(q_tof[1::2],isBinAligned=True,bin_width=0.1,user_roi=tof_roi)[1])
plt.plot(x,y,label='ref2')




plt.figure(32132313)


pointwise_scales,piecewise_scales = scaling_correction.get_all_scale_coeffs(varying_tof,
                                                         m2q_roi=tof_roi ,
                                                         cts_per_slice=2**6,
                                                         max_scale=1.075,
                                                         delta_ly=1e-4)


        





ax = plotting_stuff.plot_histo(q_tof,4,user_xlim=[0, 800],user_bin_width=0.1,user_label='orig')
ax = plotting_stuff.plot_histo(varying_tof,4,clearFigure=False,user_xlim=[0, 800],user_bin_width=0.1,user_label='messed')
ax = plotting_stuff.plot_histo(varying_tof/pointwise_scales,4,clearFigure=False,user_xlim=[0, 800],user_bin_width=0.1,user_label='messed')










#
#
#
#
#
#
#
#for expon in np.arange(3,10):
#    cts_per_slice=2**expon
#    #cts_per_slice=2**3
#    #m2q_roi = [0.9,190]
#    #    tof_roi = [0, 1000]
#    
#    import time
#    t_start = time.time()
#    pointwise_scales,piecewise_scales = scaling_correction.get_all_scale_coeffs(varying_tof,
#                                                             m2q_roi=tof_roi,
#                                                             cts_per_slice=cts_per_slice,
#                                                             max_scale=1.075)
#    t_end = time.time()
#    print('Total Time = ',t_end-t_start)
#    
#    #    fake_tof_corr = fake_tof/np.sqrt(pointwise_scales)
#    q_tof_corr = varying_tof/pointwise_scales
#    
#    
#    
##    mytofplot(q_tof_corr,3)
#    
#    ax = plotting_stuff.plot_histo(q_tof_corr,4,clearFigure=False,user_xlim=[0, 800],user_bin_width=0.05,user_label='fixed')
#    
#    
#    
#    
#    
#    
#    
#    
