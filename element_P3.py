"""
.. This software was developed by employees of the National Institute of
.. Standards and Technology (NIST), an agency of the Federal Government. Pursuant
.. to title 17 United States Code Section 105, works of NIST employees are not
.. subject to copyright protection in the United States and are considered to be
.. in the public domain. Permission to freely use, copy, modify, and distribute
.. this software and its documentation without fee is hereby granted, provided
.. that this notice and disclaimer of warranty appears in all copies.

.. THE SOFTWARE IS PROVIDED 'AS IS' WITHOUT ANY WARRANTY OF ANY KIND, EITHER
.. EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY
.. THAT THE SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF
.. MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND FREEDOM FROM INFRINGEMENT,
.. AND ANY WARRANTY THAT THE DOCUMENTATION WILL CONFORM TO THE SOFTWARE, OR ANY
.. WARRANTY THAT THE SOFTWARE WILL BE ERROR FREE. IN NO EVENT SHALL NASA BE LIABLE
.. FOR ANY DAMAGES, INCLUDING, BUT NOT LIMITED TO, DIRECT, INDIRECT, SPECIAL OR
.. CONSEQUENTIAL DAMAGES, ARISING OUT OF, RESULTING FROM, OR IN ANY WAY CONNECTED
.. WITH THIS SOFTWARE, WHETHER OR NOT BASED UPON WARRANTY, CONTRACT, TORT, OR
.. OTHERWISE, WHETHER OR NOT INJURY WAS SUSTAINED BY PERSONS OR PROPERTY OR
.. OTHERWISE, AND WHETHER OR NOT LOSS WAS SUSTAINED FROM, OR AROSE OUT OF THE
.. RESULTS OF, OR USE OF, THE SOFTWARE OR SERVICES PROVIDED HEREUNDER.

.. Distributions of NIST software should also include copyright and licensing
.. statements of any third-party software that are legally bundled with the code
.. in compliance with the conditions of those licenses.
"""

'''
.. Original version of 'element' class defined in this file was copied from
.. mmass source code, version 5.5.0, "blocks.py" file.  mmass licence and copyright information
.. is as follows:

# -------------------------------------------------------------------------
#     Copyright (C) 2005-2013 Martin Strohalm <www.mmass.org>

#     This program is free software; you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation; either version 3 of the License, or
#     (at your option) any later version.

#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#     GNU General Public License for more details.

# -------------------------------------------------------------------------    

.. Modifications made by Paul Blanchard <paul.blanchard@nist.gov>
'''


class element:
    """Element object definition.
        name: (str) name
        symbol: (str) symbol
        atomicNumber: (int) atomic number
        isotopes: (dict) dict of isotopes {mass number:(mass, abundance),...}
        valence: (int)
        maxValence: (int)
    """
    
    def __init__(self, name, symbol, atomicNumber, isotopes={}, valence=None, maxValence=None):
        
        self.name = name
        self.symbol = symbol
        self.atomicNumber = int(atomicNumber)
        self.isotopes = isotopes
        self.valence = valence
        self.maxValence = maxValence
        
        # init masses
        massMo = 0
        massAv = 0
        maxAbundance = 0
        for isotop in list(self.isotopes.values()):
            massAv += isotop[0]*isotop[1]
            if maxAbundance < isotop[1]:
                massMo = isotop[0]
                maxAbundance = isotop[1]
        if massMo == 0 or massAv == 0:
            massMo = isotopes[0][0]
            massAv = isotopes[0][0]
        
        self.mass = (massMo, massAv)
        self.q1PeakDict = {}
        for mass, abundance in list(self.isotopes.values()):
            if abundance > 0.0:
                if mass not in list(self.q1PeakDict.keys()):
                    self.q1PeakDict[mass] = abundance
                else:
                    self.q1PeakDict[mass] += abundance


                        