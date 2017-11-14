#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:18:57 2017

Currently:
    
Goes through each catalog in chunks and gets
    -ra
    -dec
    -location (healpix or patch/tract)
    -type (if AGN)
    -redshift (if AGN or galaxy)
but only if the objects pass certain quality requirements, which are
    -not wise flagged or arcturus flagged
    -not in any bad patches specified beforehand
    -other quality cuts to come 

@author: csh4
"""

import pandas as pd
import numpy as np
import healpy as hp
import psutil
import sys
import time
import gc

dir_path = "/scr/depot0/csh4/"

p_rand_f = dir_path + "cats/processed/rand_f.hdf5"
p_rand_q = dir_path + "cats/processed/rand_q.hdf5"

p_agn_f =  dir_path + "cats/processed/agn_f.hdf5"
p_agn_q =  dir_path + "cats/processed/agn_q.hdf5"

p_hsc_f =  dir_path + "cats/processed/hsc_f.hdf5"
p_hsc_q =  dir_path + "cats/processed/hsc_q.hdf5"

#bad_patches of form = [[ra_min, ra_max, dec_min, dec_max], [], ... ]

bad_patches = [[0.0, 0.1, 0.0, 0.1]]

hsc_csize =  1500000
rand_csize = 1500000
agn_csize = 100000

start_time = int(time.time())
 

def in_bad_patches(ra, dec):
    is_bad = np.zeros(np.shape(ra), dtype=bool)
    for patch in bad_patches:
        ra_inside = np.logical_and(np.greater(ra, patch[0]), np.less(ra, patch[1]))
        dec_inside = np.logical_and(np.greater(dec, patch[2]), np.less(dec, patch[3]))
        ra_dec_bad = np.logical_and(ra_inside, dec_inside)
        is_bad = np.logical_or(is_bad, ra_dec_bad)
    return is_bad

def get_pix(ra, dec):
    """
    Uses healpy to mark each coordinate pair by heal pixel
    
    Returns the healpix number for each coordinate pair.
    """
    return hp.ang2pix(8, ra, dec, lonlat=True)

def rand_qc():
    """
    Iterates through the random catalog, saving the quality cut entries
    """
    good_ra = []
    good_dec = []
    pix = []
    
    chunks = pd.read_hdf(p_rand_f, chunksize=rand_csize, key="primary")
    
    for chunk in chunks:
        chunk_ra = chunk['ra'].tolist()
        chunk_dec = chunk['dec'].tolist()
        
        patches_ok = np.logical_not(in_bad_patches(chunk_ra, chunk_dec))
        wise_mask_ok = np.greater(chunk['wise_flag'].tolist(), 0)
        arcturus_ok = np.greater(chunk['flag'].tolist(), 0)
    
        all_ok = np.logical_and(np.logical_and(patches_ok, wise_mask_ok), arcturus_ok)
        indices_ok = np.where(all_ok)[0]
        
        ra_to_add = [chunk_ra[i] for i in indices_ok]
        dec_to_add = [chunk_dec[i] for i in indices_ok]
        
        good_ra = good_ra + ra_to_add
        good_dec = good_dec + dec_to_add
    
    good_ra = np.array(good_ra, dtype=float)
    good_dec = np.array(good_dec, dtype=float)
    pix = get_pix(good_ra, good_dec)
    
    dct = {"ra":good_ra, "dec":good_dec, "pix":pix}
    frame = pd.DataFrame.from_dict(dct)
    frame.to_hdf(p_rand_q, key="primary",format="table")

def hsc_qc():
    """
    HSC Quality cut
    """
    good_ra = []
    good_dec = []
    pix = []
    redshift = []
    
    chunks = pd.read_hdf(p_hsc_f, chunksize=hsc_csize, key="primary")
    
    for chunk in chunks:
        chunk_ra = chunk['ra'].tolist()
        chunk_dec = chunk['dec'].tolist()
        chunk_z = chunk['frankenz_best'].tolist()
        
        patches_ok = np.logical_not(in_bad_patches(chunk_ra, chunk_dec))
        wise_mask_ok = np.greater(chunk['wise_flag'].tolist(), 0)
        arcturus_ok = np.greater(chunk['flag'].tolist(), 0)
    
        all_ok = np.logical_and(np.logical_and(patches_ok, wise_mask_ok), arcturus_ok)
        indices_ok = np.where(all_ok)[0]
        
        good_ra = good_ra + [chunk_ra[i] for i in indices_ok]
        good_dec = good_dec + [chunk_dec[i] for i in indices_ok]
        redshift = redshift + [chunk_z[i] for i in indices_ok]
    
    good_ra = np.array(good_ra, dtype=float)
    good_dec = np.array(good_dec, dtype=float)
    redshift = np.array(redshift, dtype=float)
    pix = get_pix(good_ra, good_dec)
    
    dct = {"ra":good_ra, "dec":good_dec, "pix":pix, "redshift":redshift}
    frame = pd.DataFrame.from_dict(dct)
    frame.to_hdf(p_hsc_q, key="primary",format="table")

def agn_qc():
    """
    AGN quality cut
    """
    good_ra = []
    good_dec = []
    pix = []
    redshift = []
    type_2 = []
    
    chunks = pd.read_hdf(p_agn_f, chunksize=agn_csize, key="primary")
    
    for chunk in chunks:
        chunk_ra = chunk['RA'].tolist()
        chunk_dec = chunk['DEC'].tolist()
        hsc_object_id = chunk['object_id_hsc1'].tolist()
        
        rmag = chunk['rmag_kron_hsc1'].tolist()
        w2 = chunk['W2'].tolist()
        diff = (np.array(rmag) - np.array(w2))
        t2 = np.greater(diff,  6)
        
        #Make sure we're using the best available redshift.
        #That is, spec-z if available and frankenz if not.
        chunk_fz = chunk['fraknenz_best_hsc1'].tolist()
        chunk_sz = chunk['redshift_sz'].tolist()
        has_sz = np.logical_not(np.isnan(chunk_sz))
        for i in range(len(chunk_sz)):
            if has_sz[i]:
                chunk_fz[i] = chunk_sz[i]
        
        has_hsc_match = np.logical_not(np.isnan(hsc_object_id))
        patches_ok = np.logical_not(in_bad_patches(chunk_ra, chunk_dec))
        arcturus_ok = np.greater(chunk['flag'].tolist(), 0)
        
        all_ok = np.logical_and(np.logical_and(patches_ok, arcturus_ok),
                                has_hsc_match)
        indices_ok = np.where(all_ok)[0]
        
        type_2 = type_2 + [t2[i] for i in indices_ok]
        good_ra = good_ra + [chunk_ra[i] for i in indices_ok]
        good_dec = good_dec + [chunk_dec[i] for i in indices_ok]
        redshift = redshift + [chunk_fz[i] for i in indices_ok]
    
    good_ra = np.array(good_ra, dtype=float)
    good_dec = np.array(good_dec, dtype=float)
    redshift = np.array(redshift, dtype=float)
    pix = get_pix(good_ra, good_dec)
    
    dct = {"ra":good_ra, "dec":good_dec, "pix":pix, "redshift":redshift}
    frame = pd.DataFrame.from_dict(dct)
    frame.to_hdf(p_agn_q, key="primary", format="table")

def current_time():
    return (int(time.time()) - start_time)

def report(report_string):
    sys.stdout.flush()
    time = current_time()
    print("")
    print("--- qualifier.py reporting ---")
    print(report_string)
    print("Time is " + str(time) + " seconds from start. ", end="")
    print("Memory use is " + str(psutil.virtual_memory().percent) + "%")
    print("")
    sys.stdout.flush()
    gc.collect()

def main():
    report("Doing the random quality cut.")
    rand_qc()
    report("Doing the HSC quality cut.")
    hsc_qc()
    report("Doing the AGN quality cut.")
    agn_qc()
    report("Finished.")

if __name__ == "__main__":
    main()