# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Need to put the functions in the path
# Probably not necessary if I understood Python/Git/modules better
import os 
import sys
parent_directory = os.getcwd().rsplit(sep='\\',maxsplit=1)[0]
if parent_directory not in sys.path:
    sys.path.insert(1, parent_directory)
    

import initElements_P3
ed = initElements_P3.initElements()

import numpy as np

def In_doped_GaN():
    #                           N      Ga      In  Da
    # Define possible peaks
    pk_data =   np.array(    [  (1,     0,      0,  ed['N'].isotopes[14][0]/2),
                                (1,     0,      0,  ed['N'].isotopes[14][0]/1),
                                (1,     0,      0,  ed['N'].isotopes[15][0]/1),
                                (1,     0,      0,  ed['N'].isotopes[15][0]+ed['H'].isotopes[1][0]),
                                (0,     1,      0,  ed['Ga'].isotopes[69][0]/3),
                                (0,     1,      0,  ed['Ga'].isotopes[71][0]/3),
                                (2,     0,      0,  ed['N'].isotopes[14][0]*2),
                                (2,     0,      0,  ed['N'].isotopes[14][0]+ed['N'].isotopes[15][0]),
                                (2,     0,      0,  ed['N'].isotopes[14][0]+ed['N'].isotopes[15][0]+ed['H'].isotopes[1][0]),
                                (0,     1,      0,  ed['Ga'].isotopes[69][0]/2),
                                (0,     1,      0,  ed['Ga'].isotopes[71][0]/2),
                                (1,     1,      0,  (ed['N'].isotopes[14][0] + ed['Ga'].isotopes[69][0])/2),
                                (3,     0,      0,  ed['N'].isotopes[14][0]*3),
                                (1,     1,      0,  (ed['N'].isotopes[14][0] + ed['Ga'].isotopes[71][0])/2),
                                (3,     1,      0,  (ed['Ga'].isotopes[69][0]+3*ed['N'].isotopes[14][0])/2),
                                (3,     1,      0,  (ed['Ga'].isotopes[71][0]+3*ed['N'].isotopes[14][0])/2),
                                (0,     1,      0,  ed['Ga'].isotopes[69][0]),
                                (0,     1,      0,  ed['Ga'].isotopes[71][0]),
                                (0,     1,      0,  ed['Ga'].isotopes[71][0]+ed['H'].isotopes[1][0]),
                                (0,     0,      1,  ed['In'].isotopes[113][0]/2),
                                (0,     0,      1,  ed['In'].isotopes[115][0]/2),
                                (0,     0,      1,  ed['In'].isotopes[113][0]),
                                (0,     0,      1,  ed['In'].isotopes[115][0]),
                                ],
                                dtype=[('N','i4'),('Ga','i4'),('In','i4'),('m2q','f4')] )
    return pk_data

def Mg_doped_GaN():
    #                            N      Ga      Mg  Da
    # Define possible peaks
    pk_data =   np.array(    [  (1,     0,      0,  ed['N'].isotopes[14][0]/2),
                                (1,     0,      0,  ed['N'].isotopes[14][0]/1),
                                (1,     0,      0,  ed['N'].isotopes[15][0]/1),
                                (1,     0,      0,  ed['N'].isotopes[15][0]+ed['H'].isotopes[1][0]),
                                (0,     1,      0,  ed['Ga'].isotopes[69][0]/3),
                                (0,     1,      0,  ed['Ga'].isotopes[71][0]/3),
                                (2,     0,      0,  ed['N'].isotopes[14][0]*2),
                                (2,     0,      0,  ed['N'].isotopes[14][0]+ed['N'].isotopes[15][0]),
                                (2,     0,      0,  ed['N'].isotopes[14][0]+ed['N'].isotopes[15][0]+ed['H'].isotopes[1][0]),
                                (0,     1,      0,  ed['Ga'].isotopes[69][0]/2),
                                (0,     1,      0,  ed['Ga'].isotopes[71][0]/2),
                                (1,     1,      0,  (ed['N'].isotopes[14][0] + ed['Ga'].isotopes[69][0])/2),
                                (3,     0,      0,  ed['N'].isotopes[14][0]*3),
                                (1,     1,      0,  (ed['N'].isotopes[14][0] + ed['Ga'].isotopes[71][0])/2),
                                (3,     1,      0,  (ed['Ga'].isotopes[69][0]+3*ed['N'].isotopes[14][0])/2),
                                (3,     1,      0,  (ed['Ga'].isotopes[71][0]+3*ed['N'].isotopes[14][0])/2),
                                (0,     1,      0,  ed['Ga'].isotopes[69][0]),
                                (0,     1,      0,  ed['Ga'].isotopes[71][0]),
                                (0,     1,      0,  ed['Ga'].isotopes[71][0]+ed['H'].isotopes[1][0]),
                                (0,     0,      1,  ed['Mg'].isotopes[24][0]/2),
                                (0,     0,      1,  ed['Mg'].isotopes[24][0])
                                ],
                                dtype=[('N','i4'),('Ga','i4'),('Mg','i4'),('m2q','f4')] )
    return pk_data
    

def AlGaN():
    #                           N      Ga      Al  Da
    # Define possible peaks
    pk_data =   np.array(    [  (1,     0,      0,  ed['N'].isotopes[14][0]/2),
                                (1,     0,      0,  ed['N'].isotopes[14][0]/1),
                                (1,     0,      0,  ed['N'].isotopes[15][0]/1),
                                (1,     0,      0,  ed['N'].isotopes[15][0]+ed['H'].isotopes[1][0]),
                                (0,     1,      0,  ed['Ga'].isotopes[69][0]/3),
                                (0,     1,      0,  ed['Ga'].isotopes[71][0]/3),
                                (2,     0,      0,  ed['N'].isotopes[14][0]*2),
                                (2,     0,      0,  ed['N'].isotopes[14][0]+ed['N'].isotopes[15][0]),
                                (2,     0,      0,  ed['N'].isotopes[14][0]+ed['N'].isotopes[15][0]+ed['H'].isotopes[1][0]),
                                (0,     1,      0,  ed['Ga'].isotopes[69][0]/2),
                                (0,     1,      0,  ed['Ga'].isotopes[71][0]/2),
                                (1,     1,      0,  (ed['N'].isotopes[14][0] + ed['Ga'].isotopes[69][0])/2),
                                (3,     0,      0,  ed['N'].isotopes[14][0]*3),
                                (1,     1,      0,  (ed['N'].isotopes[14][0] + ed['Ga'].isotopes[71][0])/2),
                                (3,     1,      0,  (ed['Ga'].isotopes[69][0]+3*ed['N'].isotopes[14][0])/2),
                                (3,     1,      0,  (ed['Ga'].isotopes[71][0]+3*ed['N'].isotopes[14][0])/2),
                                (0,     1,      0,  ed['Ga'].isotopes[69][0]),
                                (0,     1,      0,  ed['Ga'].isotopes[71][0]),
                                (0,     1,      0,  ed['Ga'].isotopes[71][0]+ed['H'].isotopes[1][0]),
                                (0,     0,      1,  ed['Al'].isotopes[27][0]/3),
                                (0,     0,      1,  ed['Al'].isotopes[27][0]/2),
                                (1,     0,      1,  (ed['Al'].isotopes[27][0]+ed['N'].isotopes[14][0])/2),
                                (0,     0,      1,  ed['Al'].isotopes[27][0]/1)
                                ],
                                dtype=[('N','i4'),('Ga','i4'),('Al','i4'),('m2q','f4')] )
    return pk_data

def GaN():
    #                           N      Ga  Da
    # Define possible peaks
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
                                (3,     1,  (ed['Ga'].isotopes[69][0]+3*ed['N'].isotopes[14][0])/2),
                                (3,     1,  (ed['Ga'].isotopes[71][0]+3*ed['N'].isotopes[14][0])/2),
                                (0,     1,  ed['Ga'].isotopes[69][0]),
                                (0,     1,  ed['Ga'].isotopes[71][0]),
                                (0,     1,  ed['Ga'].isotopes[71][0]+ed['H'].isotopes[1][0])
                                ],
                                dtype=[('N','i4'),('Ga','i4'),('m2q','f4')] )
    return pk_data

def GaN_with_H():
    #                           N      Ga  Da
    # Define possible peaks
    pk_data =   np.array(    [  (0,     0,  ed['H'].isotopes[1][0]),
                                (0,     0,  ed['H'].isotopes[1][0]*2),
                                (0,     0,  ed['H'].isotopes[1][0]*3),
                                (1,     0,  ed['N'].isotopes[14][0]/2),
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
                                (3,     1,  (ed['Ga'].isotopes[69][0]+3*ed['N'].isotopes[14][0])/2),
                                (3,     1,  (ed['Ga'].isotopes[71][0]+3*ed['N'].isotopes[14][0])/2),
                                (0,     1,  ed['Ga'].isotopes[69][0]),
                                (0,     1,  ed['Ga'].isotopes[71][0]),
                                (0,     1,  ed['Ga'].isotopes[71][0]+ed['H'].isotopes[1][0])
                                ],
                                dtype=[('N','i4'),('Ga','i4'),('m2q','f4')] )
    return pk_data
