#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 08:11:22 2017

@author: csh4
"""

import pickle
import numpy as np
from mpi4py import MPI
import sys

dir_path = "/scr/depot0/csh4/"
mpi_path=  dir_path + "cats/mpi/holder/h"

    
def two_point_angular_corr_part(ra1, dec1, tree2, bins):
    """
    Does the pair counting for the correlation. 
    
    @params
        ra1, dec1 - One side of the pair counting ra and dec
        tree2     - The other side of the pair counting, as a 
                    sklearn.neighbors.KDTree object
        bins      - The cartesian bins for correlating with
    
    @returns 
        The un-normalized pair counts. Remember to divide by (len(ra1)*len(ra2))
    """
    key1 = np.asarray(ra_dec_to_xyz(ra1, dec1), order='F').T
    return tree2.two_point_correlation(key1, bins)

def ra_dec_to_xyz(ra, dec):
    """
    Convert ra & dec to Euclidean points projected on a unit sphere.
    
    dependencies: numpy
    @params
        ra, dec - ndarrays
    @returns
        x, y, z - ndarrays
    """
    ra_to_use = np.array(ra, dtype=float)
    dec_to_use = np.array(dec, dtype=float)
    
    sin_ra = np.sin(ra_to_use * np.pi / 180.)
    cos_ra = np.cos(ra_to_use * np.pi / 180.)

    sin_dec = np.sin(np.pi / 2 - dec_to_use * np.pi / 180.)
    cos_dec = np.cos(np.pi / 2 - dec_to_use * np.pi / 180.)

    return  [cos_ra * sin_dec, sin_ra * sin_dec, cos_dec]

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    comm.Disconnect()
    
    print("Starting rank " + str(rank) + ". loading from " + mpi_path + str(rank))
    sys.stdout.flush()
    
    f = open(mpi_path + str(rank), 'rb')
    args = pickle.load(f)
    
    coords = args['coords']
    tree   = args['tree']
    bins   = args['bins']
    lenra2 = args['lenra2']      
    lenra1 = args['lenra1']
    path   = args['path']
    
    result = two_point_angular_corr_part(coords[0], coords[1], tree, bins)
    result = np.array(np.diff(result), dtype=float)/(lenra1*lenra2)
    np.save(path, result)
    
    print("Finished rank " + str(rank) + ". saved result " +
          str(result) + " to " + path)
    sys.stdout.flush()
    f.close()
    
    print("code:exit" + str(rank))
    sys.stdout.flush()


