#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: HSC Weak Lensing collaboration
"""

import os
import sys
import math
import numpy as np
import pyfits
import matplotlib.pyplot as plt
import healpy as hp

def removeDuplicatedVisits(d):
    # Remove visits with the same pntgid. Those visits were retaken because 
    # transparency was bad. They should be counted as a single visit rather than 
    # multiple visits
    print("# of visits before removing duplicates in pntgid:", str(len(d)))
    sel = list()
    for i in range(len(d)):
        eq = (d["pntgid"] == d["pntgid"][i])
        if np.sum(eq) > 1:
            indices = np.where(eq)[0]
            if indices[0] == i:
                sel.append(i)
        else:
            sel.append(i)
    d = d[sel]
    print("# of visits after removing duplicates in pntgid:"+str(len(d)))
    return d

def getFullAreaHealPixels(d):
    # Count up the number of visits and define full-color full-depth region in 
    # healpix. This part is originally from Yasuda-san. 
    bands = ['g', 'r', 'i', 'z', 'y']
    Nexp = {'g': 4, 'r': 4, 'i': 6, 'z': 6, 'y': 6}

    NSIDE = 1024
    m = dict()
    for b in bands:
        m[b] = np.zeros(hp.nside2npix(NSIDE))

    for dd in d:
        b = dd["filter"][0:1]

        phi = math.radians(dd["ra"])
        theta = math.radians(90.-dd["dec"])
    
        vec = hp.ang2vec(theta, phi)
        healpix_bins = hp.query_disc(NSIDE, vec, math.radians(0.75))

        for index in healpix_bins:
            m[b][index] += 1

    fill = dict()
    full = np.ones(hp.nside2npix(NSIDE))
    for b in bands:
        fill[b] = m[b] >= Nexp[b]
        print(str(NSIDE), str(b), str(129600/math.pi * sum(fill[b]) / len(fill[b])))
        full = np.logical_and(full, fill[b])

    print(str(NSIDE), str(5), str(129600/math.pi * sum(full) / len(full)))
    return full

def getContiguousPixels(m):
    # Get contiguous regions to remove isolated pixels around each field.
    nside =hp.get_nside(m)
    m_out = np.zeros(m.shape, dtype = np.bool)

    for field in ["AEGIS", "HECTOMAP", "GAMA09H", "WIDE12H", "GAMA15H", "VVDS", "XMM"]:
        print(field)
        if field == "AEGIS":
            alpha0 = 215.
            delta0 = 52.5
        if field == "HECTOMAP":
            alpha0 = 240.
            delta0 = 43.
        if field == "GAMA09H":
            alpha0 = 137.
            delta0 = 1.
        if field == "WIDE12H":
            alpha0 = 180.
            delta0 = 0.
        if field == "GAMA15H":
            alpha0 = 220.
            delta0 = 0.
        if field == "VVDS":
            alpha0 = 336.
            delta0 = 1.
        if field == "XMM":
            alpha0 = 34.
            delta0 = -4.
        
        phi = math.radians(alpha0)
        theta = math.radians(90-delta0)

        to_be_visited = list()
        visited = list()
        ipix = hp.ang2pix(nside, theta, phi)

        if m[ipix] != True:
            print("central pixel is not true")
            os.exit(1)
        m_out[ipix] = True

        flag = True
        ipix_in = ipix
        i = 0
        # grow from center
        while(flag):
            ipixs = hp.get_all_neighbours(nside, ipix_in)[[0,2,4,6]]
            visited.append(ipix_in)
            m_out[ipixs] = m[ipixs]
            ipixs_true = ipixs[m[ipixs]]
            ipixs_true_not_visited = list()
            for item in ipixs_true:
                if not (item in visited):
                    if not (item in to_be_visited):
                        ipixs_true_not_visited.append(item)
            to_be_visited += ipixs_true_not_visited
            ipix_in = np.min(to_be_visited)
            to_be_visited.remove(ipix_in)
            if len(to_be_visited) == 0:
                flag = False
            i += 1
            if i % 100 == 0:
                print(str(i), str(len(visited)), str(len(set(visited))),
                      str(len(to_be_visited)), str(len(set(to_be_visited))))
    return m_out
                    
def plot_mask(ra, dec, nside=2**10):
    pix = hp.pixelfunc.ang2pix(nside, dec, dec, lonlat=True)
    hp.visufunc.mollview(map=np.bincount(pix,minlength=hp.nside2npix(nside)),
                         cbar=False, notext=True, max=1, min=0, title="",
                         cmap="binary")
    hp.graticule()
    plt.show()
    
if __name__ == "__main__":
    
    location = '/scratch/csh4/tools/147279.fits'
    
    d = pyfits.getdata(location)#sys.argv[1])

    # I do not know why, but a visit in s16a_wide.mosaicframe__deepcoadd is missing in s16a_wide.frame.
    d = d[~d["visit_isnull"]]

    d = removeDuplicatedVisits(d)
    m = getFullAreaHealPixels(d)
    m = getContiguousPixels(m)
    
#    os.remove("/scratch/csh4/tools/S16A_fdfc_hp_map.fits")
#    hp.write_map("/scratch/csh4/tools/S16A_fdfc_hp_map.fits", m,
#                 dtype=np.bool, nest = False)
    
    hp.visufunc.mollview(map=m,
                         cbar=False, notext=True, max=1, min=0, title="",
                         cmap="binary")
    hp.graticule()
    plt.show()
    
    
    
    