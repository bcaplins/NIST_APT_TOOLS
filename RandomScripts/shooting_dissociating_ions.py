# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 10:55:04 2020

@author: bwc
"""

import numpy as np
import numpy.random

def draw_theta():
#    return numpy.random.uniform(0,np.pi)
#    return numpy.random.uniform(0,np.pi/4)
    return numpy.random.normal(loc=0,scale=np.pi/4)

def draw_time(tau=10e-9):
    return numpy.random.exponential(scale=tau)

def draw_kinetic_release(x0=6, sigma=.1):
    joule_per_eV = 1.60218e-19
    ker = numpy.random.normal(loc=joule_per_eV*x0,scale=joule_per_eV*sigma)
    if ker < 0:
        ker=0
    return ker
    
def calc_diss_vel(ker,mA,mB,theta):
    vA = np.sqrt(2*ker/((mA/mB)*(mA+mB)))
    vB = np.sqrt(2*ker/((mB/mA)*(mA+mB)))
    
    vxA = np.sin(theta)*vA
    vzA = np.cos(theta)*vA
    
    vxB = -np.sin(theta)*vB
    vzB = -np.cos(theta)*vB
    
    return (vxA,vzA,vxB,vzB)
    

def prop_ion_to_det(x0,z0,vx,vz,det_z):
    a = vx**2+vz**2
    b = 2*(x0*vx+z0*vz)
    c = x0**2+z0**2-det_z**2    
    
    dt = (-b+np.sqrt(b**2-4*a*c))/(2*a)    
    
    x = vx*dt + x0
    z = vz*dt + z0
    
    vx = vx
    vz = vz
    
    return (dt,x,z,vx,vz)

def prop_ion_to_plane(x0,z0,vx,vz,az,plane_z):
    a = 0.5*az
    b = vz
    c = z0-plane_z
    dt = (-b+np.sqrt(b**2-4*a*c))/(2*a)
       
    x = vx*dt + x0
    z = a*dt**2 + b*dt + c
    
    vx = vx
    vz = az*dt + vz
    
    return (dt,x,z,vx,vz)

def prop_ion_to_time(x0,z0,vx,vz,az,dt):
    x = vx*dt + x0
    z = 0.5*az*dt**2 + vz*dt + z0
    
    vx = vx
    vz = az*dt + vz
        
    return (dt,x,z,vx,vz)


dd = 90e-3
d1 = 40e-6
d2 = dd-d1



amu2kg = 1.66054e-27
chargestate2coulomb = 1.60217662e-19

mAB = 2*12*amu2kg
mA = 1*12*amu2kg
mB = 1*12*amu2kg

qAB = 2*chargestate2coulomb
qA = 1*chargestate2coulomb
qB = 1*chargestate2coulomb

volt = 6000

aAB = qAB*volt/mAB/d1    
aA = qA*volt/mA/d1    
aB = qB*volt/mB/d1  

N = 2**16

tofA = np.zeros(N)
tofB = np.zeros(N)
delta_r = np.zeros(N)

for i in np.arange(N):
    theta = draw_theta()
    t_diss = draw_time()
    ker = draw_kinetic_release()
    
    (t_1_max,_,_,_,_) = prop_ion_to_plane(x0=0,z0=0,vx=0,vz=0,az=aAB,plane_z=d1)
    
    t = 0
    if(t_diss < t_1_max):
        
        
        # dissociates in field

        # Propagate AB to point of dissociation         
        (dt_pre,x,zAB,vxAB,vzAB) = prop_ion_to_time(x0=0,z0=0,
                                                vx=0,vz=0,
                                                az=aAB,
                                                dt=t_diss)
        
        
        # dissociate ion
        (vxA,vzA,vxB,vzB) = calc_diss_vel(ker,mA,mB,theta)
        
        # Right after dissociation these are positions and velocities
        zA = zAB
        zB = zAB
        vzA = vzA+vzAB
        vzB = vzB+vzAB
        vxA = vxA+vxAB
        vxB = vxB+vxAB

        # Propagate to LE plane
        (dtA_post,xA,zA,vxA,vzA) = prop_ion_to_plane(x0=0,z0=zA,
                                                vx=vxA,vz=vzA,
                                                az=aA,
                                                plane_z=d1)
        
        (dtB_post,xB,zB,vxB,vzB) = prop_ion_to_plane(x0=0,z0=zB,
                                                vx=vxB,vz=vzB,
                                                az=aB,
                                                plane_z=d1)
        
        # Propagate to detector
        (dtA_ff,xA,zA,vxA,vzA) = prop_ion_to_det(x0=xA,z0=zA,
                                                vx=vxA,vz=vzA,
                                                det_z=dd)
        (dtB_ff,xB,zB,vxB,vzB) = prop_ion_to_det(x0=xB,z0=zB,
                                                vx=vxB,vz=vzB,
                                                det_z=dd)
        
        # final
        tofA[i] = dt_pre + dtA_post + dtA_ff
        tofB[i] = dt_pre + dtB_post + dtB_ff
        
        delta_r[i] = np.sqrt((xA-xB)**2 + (zA-zB)**2)
        
    else:
        # dissociates in field free
        
        # Propagate to LE
        (dtAB_f,xAB,zAB,vxAB,vzAB) = prop_ion_to_plane(x0=0,z0=0,
                                                    vx=0,vz=0,
                                                    az=aAB,
                                                    plane_z=d1)
       
        # Propagate to point of dissociation
        (dt_ff,xAB,zAB,vxAB,vzAB) = prop_ion_to_time(x0=xAB,z0=zAB,
                                            vx=vxAB,vz=vzAB,
                                            az=0,
                                            dt=t_diss-t_1_max)
        # dissociate ion
        (vxA,vzA,vxB,vzB) = calc_diss_vel(ker,mA,mB,theta)

        # Right after dissociation these are positions and velocities
        zA = zAB
        zB = zAB
        vzA = vzA+vzAB
        vzB = vzB+vzAB
        vxA = vxA+vxAB
        vxB = vxB+vxAB

        # Propagate to detector
        (dtA_ff,xA,zA,vxA,vzA) = prop_ion_to_det(x0=0,z0=zA,
                                                vx=vxA,vz=vzA,
                                                det_z=dd)
        (dtB_ff,xB,zB,vxB,vzB) = prop_ion_to_det(x0=0,z0=zB,
                                                vx=vxB,vz=vzB,
                                                det_z=dd)
        
        # final
        tofA[i] = dtAB_f + dt_ff + dtA_ff
        tofB[i] = dtAB_f + dt_ff + dtB_ff
        
        delta_r[i] = np.sqrt((xA-xB)**2 + (zA-zB)**2)
    
    
    
    
    
    
    
(dt1,x,z,vx,vz) = prop_ion_to_plane(x0=0,z0=0,vx=0,vz=0,az=aAB,plane_z=d1)
(dt2,x,z,vx,vz) = prop_ion_to_det(x0=x,z0=z,vx=vx,vz=vz,det_z=dd)
    
    
tAB = dt1+dt2


m2qA = np.square(tofA/tAB)*(mAB/qAB)/amu2kg*chargestate2coulomb
m2qB = np.square(tofB/tAB)*(mAB/qAB)/amu2kg*chargestate2coulomb

# Dither
sigma = 0.05
m2qA = m2qA + numpy.random.normal(loc=0,scale=sigma,size=N)
m2qB = m2qB + numpy.random.normal(loc=0,scale=sigma,size=N)




    
import matplotlib.pyplot as plt

    
    
 
fig = plt.figure(num=1)
plt.clf()
ax = fig.gca()

#ax.scatter(, tofB/1e-9)


lhs = np.min([np.min(m2qA),np.min(m2qB)])
rhs = np.max([np.max(m2qA),np.max(m2qB)])
n_bin = 128
step = (rhs-lhs)/n_bin


edges = np.arange(lhs,rhs,step)


# Small bins
plt.hist2d(m2qA, 
            m2qB, 
            bins=(edges, edges), cmap=plt.cm.Reds)
plt.plot(edges,edges,color='w',ls='--')

#plt.show()

    
fig = plt.figure(num=2)
plt.clf()
ax = fig.gca()
    

edges_m2q = np.arange(0,1,0.025)
edges_r = np.arange(0,10,0.05)

plt.hist2d(delta_r*1000,
           np.abs(m2qA-m2qB),
           bins=(edges_r,edges_m2q), cmap=plt.cm.Reds)
plt.plot(edges,edges,color='w',ls='--')









