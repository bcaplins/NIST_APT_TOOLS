# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 10:21:03 2019

@author: bwc

Numpy versions of functions by https://github.com/MHC03
Original functions by https://github.com/oscarbranson

"""

import numpy as np




def read_pos_numpy(file_path):
    """ Loads an APT .pos file as np struct array
    Columns:
        x: Reconstructed x position
        y: Reconstructed y position
        z: Reconstructed z position
        m2q: mass/charge ratio of ion"""
    if not file_path.lower().endswith('.pos') :
        raise ValueError("File Path does not end with .pos.")
    with open(file_path, 'rb') as f:
        # >f4 is big endian 4 byte float
        pos_d_type = np.dtype({'names': ['x', 'y', 'z', 'm2q'],
                             'formats': ['>f4', '>f4', '>f4', '>f4']})
        # returns a numpy array, where you can, for example, access all 'x' by pos_arr['x'] and so on
        pos_arr = np.fromfile(f, pos_d_type, -1)
    return pos_arr


def read_epos_numpy(file_path):
    """Loads an APT .epos file as a pandas dataframe.
    Columns:
        x: Reconstructed x position
        y: Reconstructed y position
        z: Reconstructed z position
        m2q: Mass/charge ratio of ion
        tof: Ion Time Of Flight
        v_dc: Potential
        v_pulse: Size of voltage pulse (voltage pulsing mode only)
        x_det: Detector x position
        y_det: Detector y position
        pslep: Pulses since last event pulse (i.e. ionisation rate)
        ipp: Ions per pulse (multihits)
     [x,y,z,m2q,tof,v_dc,v_pulse,x_det,y_det,pslep,ipp].
        pslep = pulses since last event pulse
        ipp = ions per pulse
    When more than one ion is recorded for a given pulse, only the
    first event will have an entry in the "Pulses since last evenT
    pulse" column. Each subsequent event for that pulse will have
    an entry of zero because no additional pulser firings occurred
    before that event was recorded. Likewise, the "Ions Per Pulse"
    column will contain the total number of recorded ion events for
    a given pulse. This is normally one, but for a sequence of records
    a pulse with multiply recorded ions, the first ion record will
    have the total number of ions measured in that pulse, while the
    remaining records for that pulse will have 0 for the Ions Per
    Pulse value.
        ~ Appendix A of 'Atom Probe tomography: A Users Guide',
          notes on ePOS format."""
    if not file_path.lower().endswith('.epos'):
        raise ValueError("File Path does not end with .epos.")
    with open(file_path, 'rb') as f:
        # >f4 is big endian 4 byte float
        epos_d_type = np.dtype({'names': ['x', 'y', 'z', 'm2q', 'tof', 'v_dc', 'v_pulse', 'x_det', 'y_det', 'pslep', 'ipp'],
                                'formats': ['>f4', '>f4', '>f4', '>f4', '>f4', '>f4', '>f4', '>f4', '>f4', '>i4', '>i4']})

        # returns a numpy array, where you can, for example, access all 'x' by pos_arr['x'] and so on
        pos_arr = np.fromfile(f, epos_d_type, -1)
    return pos_arr

def write_epos_numpy(epos, fn):
    if not fn.lower().endswith('.epos'):
        fn += '.epos'
    with open(fn,'wb') as f:
        bytes_written = f.write(epos.tobytes())
    return (bytes_written == epos.nbytes), fn

def write_pos_numpy(pos, fn):
    if not fn.lower().endswith('.pos'):
        fn += '.pos'
    with open(fn,'wb') as f:
        bytes_written = f.write(pos.tobytes())
    return (bytes_written == pos.nbytes), fn 


#def read_rrng(f):
#    """Loads a .rrng file produced by IVAS. Returns two dataframes of 'ions'
#    and 'ranges'."""
#    import re
#
#    rf = open(f,'r').readlines()
#
#    patterns = re.compile(r'Ion([0-9]+)=([A-Za-z0-9]+).*|Range([0-9]+)=(\d+.\d+) +(\d+.\d+) +Vol:(\d+.\d+) +([A-Za-z:0-9 ]+) +Color:([A-Z0-9]{6})')
#
#    ions = []
#    rrngs = []
#    for line in rf:
#        m = patterns.search(line)
#        if m:
#            if m.groups()[0] is not None:
#                ions.append(m.groups()[:2])
#            else:
#                rrngs.append(m.groups()[2:])
#
#    ions = pd.DataFrame(ions, columns=['number','name'])
#    ions.set_index('number',inplace=True)
#    rrngs = pd.DataFrame(rrngs, columns=['number','lower','upper','vol','comp','colour'])
#    rrngs.set_index('number',inplace=True)
#
#    rrngs[['lower','upper','vol']] = rrngs[['lower','upper','vol']].astype(float)
#    rrngs[['comp','colour']] = rrngs[['comp','colour']].astype(str)
#
#    return ions,rrngs
#
#
#def label_ions(pos,rrngs):
#    """labels ions in a .pos or .epos dataframe (anything with a 'Da' column)
#    with composition and colour, based on an imported .rrng file."""
#
#    pos['comp'] = ''
#    pos['colour'] = '#FFFFFF'
#
#    for n,r in rrngs.iterrows():
#        pos.loc[(pos.Da >= r.lower) & (pos.Da <= r.upper),['comp','colour']] = [r['comp'],'#' + r['colour']]
#
#    return pos
#
#
#def deconvolve(lpos):
#    """Takes a composition-labelled pos file, and deconvolves
#    the complex ions. Produces a dataframe of the same input format
#    with the extra columns:
#       'element': element name
#       'n': stoichiometry
#    For complex ions, the location of the different components is not
#    altered - i.e. xyz position will be the same for several elements."""
#
#    import re
#
#    out = []
#    pattern = re.compile(r'([A-Za-z]+):([0-9]+)')
#
#    for g,d in lpos.groupby('comp'):
#        if g is not '':
#            for i in range(len(g.split(' '))):
#                tmp = d.copy()
#                cn = pattern.search(g.split(' ')[i]).groups()
#                tmp['element'] = cn[0]
#                tmp['n'] = cn[1]
#                out.append(tmp.copy())
#    return pd.concat(out)