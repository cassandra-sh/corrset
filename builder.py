#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 13:14:35 2017
builder.py
-Constructs the catalogs from scratch, step by step

Trying to intelligently use astropy where it makes sense and pandas where it 
does not. 

@author: csh4
"""

import os
import numpy as np
import pymangle
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
import gc
import time
import sys
import warnings
import subprocess
import psutil
import pandas as pd
import multiprocessing as mp
import cross_match as cm


"""
Catalog/file object nomenclature
name_x_fits - fits file type
p_name  - path to file
name_u  - unprocessed at all (typically a fits)
name_v  - venice mask applied (fits in and out)
name_vw - venice and wise mask applied
name_p  - partially processed (cross match started but not finished)
name_f  - fully processed/finished catalog
name_q  - quality cut catalog (whatever is decided on)
"""

"""Input parameters"""
hsc_csize =  1500000
rand_csize = 1500000
sdss_csize = 100000
agn_csize = 100000

#Some extra functionality for predicting how many rows there are
n_hsc_rows = 38130959
n_rand_rows = 52094203
get_row_nums = False

dir_path = "/scr/depot0/csh4/"

"""Catalogs info"""
#note: this version assumes the random is already assembled
p_rand_u_fits =  dir_path + "cats/unprocessed/rand_u.fits"
p_rand_v_fits = dir_path + "cats/partial/rand_v.fits"
p_rand_v =      dir_path + "cats/partial/rand_v.hdf5"
p_rand_vw =     dir_path + "cats/partial/rand_vw.hdf5"
p_rand_f =      dir_path + "cats/processed/rand_f.hdf5"

p_agn_u_fits =  dir_path + "cats/unprocessed/agn_u.fits"
p_agn_u =       dir_path + "cats/unprocessed/agn_u.hdf5"
p_agn_sz =      dir_path + "cats/partial/agn_sz.hdf5"    #1/3 cross match (specz)
p_agn_p =       dir_path + "cats/partial/agn_p.hdf5"     #1/2 cross matches done (hsc)
p_agn_f =       dir_path + "cats/processed/agn_f.hdf5"   #2/2 cross matches done (sdss)

p_hsc_u_fits =  dir_path + "cats/unprocessed/hsc_u.fits"
p_hsc_v_fits =  dir_path + "cats/partial/hsc_v.fits"
p_hsc_v =       dir_path + "cats/partial/hsc_v.hdf5"
p_hsc_vw =      dir_path + "cats/partial/hsc_vw.hdf5"
p_hsc_f =       dir_path + "cats/processed/hsc_f.hdf5"  #specz cross match done

p_specz_fits = dir_path + "cats/reference/DR1_specz_catalog.fits"
p_specz = dir_path + "cats/reference/DR1_specz_catalog.hdf5"
specz_raname   = 'ra2000'
specz_decname  = 'decl2000'
specz_specname = 'redshift'

p_sdss_fits = dir_path + "cats/reference/sdss_quasar_cat.fits"
p_sdss = dir_path + "cats/reference/sdss_quasar_cat.hdf5"
mgII_name = 'FWHM_MGII'
cIV_name  = 'FWHM_CIV'

"""Masks Info"""
wise_mangles = ["/u/csh4/Dropbox/research/agn/polygons/wise_mask_allwise_stars.ply",
                "/u/csh4/Dropbox/research/agn/polygons/wise_mask_allsky_pix.ply"]

"""Venice info"""
venice = dir_path + "tools/HSC-SSP_brightStarMask_Arcturus/venice-4.0.3/bin/venice"
venice_masks = " -m /scr/depot0/csh4/tools/HSC-SSP_brightStarMask_Arcturus/reg/masks_all.reg"
file_in = dir_path + "cats/unprocessed/rand_d.fits"
file_out = dir_path + "cats/unprocessed/rand_f.fits"

sample_in = (venice + venice_masks + " -f all -cat " + file_in + " -xcol RA -ycol DEC -o " + file_out)

"""Stilts info"""
stilts = dir_path + "tools/topcat/stilts"

"""Flag info"""
flag_names = ["iflags_pixel_saturated_any",
              "iflags_pixel_edge",
              "iflags_pixel_interpolated_any",
              "iflags_pixel_interpolated_center",
              "iflags_pixel_saturated_any",
              "iflags_pixel_saturated_center",
              "iflags_pixel_cr_any",
              "iflags_pixel_cr_center",
              "iflags_pixel_bad",
              "iflags_pixel_suspect_any",
              "iflags_pixel_suspect_center",
              "iflags_pixel_offimage",
              "iflags_pixel_bright_object_center",
              "iflags_pixel_bright_object_any",
              "icmodel_flags_badcentroid"]


def main():
    """
    Build everything from scratch. Comment out the finished parts.
    """
    start_time = int(time.time())
    def current_time():
        return (int(time.time()) - start_time)
    def report(report_string):
        sys.stdout.flush()
        time = current_time()
        print(report_string)
        print("Time is " + str(float(time/60.)) + " minutes from start. ", end="")
        print("Memory use is " + str(psutil.virtual_memory().percent) + "%")
        sys.stdout.flush()
        gc.collect()
        print("")
    report("Beginning builder.main()")
    
    if get_row_nums:
        pass
    
    """
    Step 1: Venice flags for
                -rand_u_fits -> rand_v_fits
                -hsc_u_fits  -> hsc_v_fits
    Already works/done
    """
    report("Flagging each relevant catalog with Venice")
    venice_mask(p_rand_u_fits, p_rand_v_fits, overwrite=True)
    venice_mask(p_hsc_u_fits, p_hsc_v_fits, overwrite=True)
    
    """
    Step 2: FITS to hdf for
                -sdss_fits   -> sdss
                -specz_fits  -> specz
                -hsc_v_fits  -> hsc_v
                -rand_v_fits -> rand_v
                -agn_u_fits  -> agn_u
       
    """
    report("File 1")
    fits_to_hdf(p_sdss_fits, p_sdss, overwrite=True)
    report("File 2")
    fits_to_hdf(p_specz_fits, p_specz, overwrite=True)
    report("File 3")
    fits_to_hdf(p_hsc_v_fits, p_hsc_v, overwrite=True)
    report("File 4")
    fits_to_hdf(p_rand_v_fits, p_rand_v, overwrite=True)
    report("File 5")
    fits_to_hdf(p_agn_u_fits, p_agn_u, overwrite=True)
    """
    Step 3: WISE flags for
                -rand_v -> rand_vw
                -hsc_v  -> hsc_vw
    """
    report("Flagging hsc catalog with wise mask")
    add_wise_mask_column(p_hsc_v, p_hsc_vw, chunksize=hsc_csize, verbose=True,
                         n_rows=n_hsc_rows, overwrite=True)
    report("Flagging random catalog with wise mask")
    add_wise_mask_column(p_rand_v, p_rand_f, chunksize=rand_csize, verbose=True,
                         n_rows=n_rand_rows, overwrite=True)

    """
    Step 4: Cross matches
                -specz,  agn_u  -> agn_sz (right join)
                -agn_sz, hsc_vw -> agn_p (left  join)
                -agn_p,  sdss   -> agn_f (left  join)
                -specz,  hsc_vw -> hsc_f (right join)
    Remember - the right value for cross match should be the larger hdf
    """
    
    report("Cross matching WISE to Spec-Z")
    cm.file_cross_match(p_agn_u, p_specz, p_agn_sz, '_sz', agn_csize,
                        algorithm = "closest", append='left',
                        left_ran = "RA", left_decn = "DEC",
                        right_ran = specz_raname, right_decn = specz_decname)
    
    report("Cross matching WISE to HSC")
    cm.file_cross_match(p_agn_sz, p_hsc_vw, p_agn_p, '_hsc1', hsc_csize, 
                        right_metric="imag_kron", algorithm = "lowest",
                        left_ran = "RA",left_decn = "DEC")
   
    report("Cross matching WISE to SDSS")
    cm.file_cross_match(p_agn_p, p_sdss, p_agn_f, "_sdss1", sdss_csize,
                                 algorithm = "closest",
                                 left_ran = "RA", left_decn = "DEC",
                                 right_ran = "RA", right_decn = "DEC")
        
    report("Cross matching HSC to Spec-Z")
    cm.file_cross_match(p_specz, p_hsc_vw, p_hsc_f, '_sz', agn_csize,
                        algorithm = "closest", append='right',
                        left_ran = specz_raname, left_decn = specz_decname)
    
    report("Done")


def venice_mask(unprocessed, masked, ra_name='ra', dec_name='dec', overwrite=False):
    """
    Venice masks the fits at the given path. 
    
    Adds a column called 'flag' which is 1 if the object is not masked and 
    0 if it is masked.
    
    dependencies: subprocess, Venice
    
    @params
        unprocessed       - name of the file to mask
        masked            - where the new file will be saved
        ra_name, dec_name - the name of the columns
    """
    o = ""
    if overwrite == True:
        o = "!"
    command = (venice + venice_masks + " -f all -cat " + unprocessed +
                     " -xcol " + ra_name + " -ycol " + dec_name + " -o " + o + masked)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()


def file_len(path):
    #Figure out the number of lines in a file
    with open(path) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def fits_to_hdf(path, new_path, overwrite=False):
    """
    Take a fits file and use astropy to retrieve the data. Save the data in
    an hdf5 table file.
    
    performance: low (astropy.io.fits memory limited)
    
    @params
        path
            Path to get the fits file from
        new_path
            Path to save the hdf5 table to
        overwrite=False
            If false, raise an error if something is already at new_path
    """
    if overwrite == False and os.path.isfile(new_path):
        raise IOError("Overwrite = False but there is a file at " + new_path)
            
#    Possible fitsio implementation:
#
#    #Make sure to populate names with the non-list column names first
#
#    #Get the fits file with fitsio
#    fits=fitsio.FITS('data.fits')
#        
#    #Figure out a smart way to take into account the end of the file
#    range1 = np.arange(0, filelimit, chunksize)
#    range2 = np.arange(chunksize, filelimit, chunksize)
#    range2[-1] = filelimit
#    for i in range(len(range1)):
#       dat = fits[1][range1[i]:range2[i]]
#       dct_to_add = {}
#        for j in range(len(names)):
#            dct_to_add[names[j]] = dat[names[j]]
#        frame = pd.DataFrame.from_dict(dct_to_add)
#        if i == 0:
#            frame.to_hdf(new_path, mode='w', format="table", key="primary")
#        else:
#            frame.to_hdf(new_path, mode='a', append=True, format="table",
#                         key="primary")
            
    
    f = fits.open(path)
    dat = f[1].data
    names = f[1].columns.names
    types = [f[1].columns.dtype[i] for i in range(len(names))]
    
    shapes = [len(np.shape(types[i])) for i in range(len(names))]
    not_list_columns = np.where(np.less(shapes, 1))[0]
    
    if len(not_list_columns) == len(names):
        frame = pd.DataFrame(data = dat)
        frame.to_hdf(new_path, key="primary", format="table")
    else:
        dropped_names = [names[i] for i in np.where(np.greater(shapes,0))[0]]
        warnings.warn(path + " has lists stored in columns " +
                      str(dropped_names) + ", which will be dropped.")
        dct = {}
        for i in not_list_columns:
            dct[names[i]] = dat[names[i]]
        frame = pd.DataFrame.from_dict(dct)
        frame.to_hdf(new_path, key="primary",format="table")
    
def add_wise_mask_column(unprocessed, masked, ra_name='ra', dec_name='dec',
                         chunksize=10000, verbose=False, n_rows=None,
                         overwrite=False):
    """
    Apply the WISE mangle mask to a file. 
    
    Adds a column called "wise_flag" which is 1 if the object is not masked
    and 0 if it is masked.
    
    For this one, higher chunksize is much better and will improve performance
    if memory is available.
    
    performance: Medium (multiprocessing enabled)
    
    @params
        unprocessed       - name of the file to mask
        masked            - where the new masked file will be saved
        ra_name, dec_name - the name of the columns
        chunksize         - size of chunks to read and write the file in
        verbose           - whether or not to print a progress bar
        n_rows            - number of rows in the file, to be used in printing a 
                            progress bar. If verbose=True but no n_rows is
                            passed, it will be generated (which can be slow and
                            not worth it). 
        overwrite=False   - Whether or not to overwrite if there is a file there
                            already.
    """
    #-1. Check to make sure we're not overwriting on accident...
    if overwrite == False and os.path.isfile(masked):
        raise IOError("Overwrite = False but there is a file at " + masked)
    elif overwrite == True and os.path.isfile(masked):
        os.remove(masked)
        
        
    #0.  Prepare some things
    if verbose:
        if n_rows == None:
            n_rows = (int(file_len(unprocessed)/chunksize)+1)
        else:
            n_rows = int(n_rows/chunksize)+1
        print("╠", end="")
        for i in range(2*n_rows): 
            print("═", end="")
        print("╣")
        print("╠", end="")
    
    #1. Get the unprocessed hdf in chunks
    frame = pd.read_hdf(unprocessed, chunksize=chunksize, key="primary")
    if chunksize == None: 
        frame = [frame]
    
    #Define the helper function which will be fed into multiprocessing.Process()
    def chunk_process(chunk, n, curr_write, done):
        #i.   Get ra and dec
        ra, dec = chunk[ra_name], chunk[dec_name]
        #ii.  Feed the RA and DEC into the wise mask function
        unmasked_indices = wise_mask(ra, dec)
        #iii. Use the array returned to construct an array to add to the DataFrame
        array_to_add = np.zeros(len(ra), dtype=int)
        for i in unmasked_indices:
            array_to_add[i] = 1
        #iv.  Add it to the DataFrame
        chunk['wise_flag']  = array_to_add
        #v.   Append to the hdf, in order, one at a time
        written = False
        while not written:
            #Check to see if it is this chunk's turn
            if curr_write.value == n:
                if n == 0: chunk.to_hdf(masked, mode='w', format="table",
                                        key="primary")
                else: chunk.to_hdf(masked, mode='a',  format="table",
                                   key="primary", append=True)
                fin.value = 1
                curr_write.value = n+1
                written = True
                if verbose: print("█", end="")
            else: #If it isn't, wait for its turn.
                time.sleep(3)
    
    processes = []
    status = []
    cw = mp.Value('i', 0)
    ncores = mp.cpu_count()
    n = 0    
    #2. Iterate through the chunks, giving over to multiprocessing.Process()
    for chunk in frame:
        if verbose: print("█", end="")
        #i.   Load up a process and start it. Keep track of when they finish.
        fin = mp.Value("i", 0)
        status.append(fin)
        processes.append(mp.Process(target=chunk_process,
                                    args=(chunk, n, cw, fin)))
        processes[-1].start()
        
        
        #ii.  If there are more than ncores processes, wait for there to be fewer
        while len(processes) >= ncores:
            #Check to see if any processes are finished. If they are, remove
            processes_in_progress, processes_in_progress_status = [], []
            for i in range(len(status)):
                if status[i].value == 0:
                    processes_in_progress.append(processes[i])
                    processes_in_progress_status.append(status[i])
            processes, status = processes_in_progress, processes_in_progress_status
            time.sleep(3)
        
        n = n + 1
        #ii.  Clean up, continue
        chunk = None
        gc.collect()
    
    #3. Wait for any processes to finish up.
    for p in processes:
        p.join()
        
    if verbose: print("╣")

def wise_mask(ra, dec, mangle_files=wise_mangles):
    """
    Takes a list of RA and DEC and returns the indices which are permissible
    
    Uses equatorial coordinates, converts them to galactic, does the thing, and
    returns the indices.
    
    Dependencies: numpy, astropy.coordinates.SkyCoord, mangle
    
    performance: medium (probably max)
    
    @params
        ra_list, dec_list
            The coordinate pairs to be compared to the footprint
        mangle_files = wise_mangles
            The iterable of full filepaths to the .ply files to be used for
            masking
        galactic = True
        
    @returns indices
            The indices of ra and dec which are not masked 
    """
    coordinates = SkyCoord(ra=ra*u.degree,
                           dec=dec*u.degree,
                           frame='icrs')
    coordinates = coordinates.transform_to("galactic")
    ga_ra_list = coordinates.l.deg
    ga_dec_list = coordinates.b.deg
    m = []
    for path in mangle_files:
        m.append(pymangle.Mangle(path))
    ins = []
    for j in range(0, len(m)):
        ins.append(m[j].contains(ga_ra_list, ga_dec_list))
    ins_true = ins[0]
    for j in range(1, len(m)):
        ins_true = np.logical_or(ins_true, ins[j])
    return np.where(ins_true == False)[0]

    
def ra_dec_to_xyz(ra, dec):
    """
    Convert ra & dec to Euclidean points projected on a unit sphere
    
    dependencies: numpy
    @params
        ra, dec - ndarrays
    @returns
        x, y, z - ndarrays
    """
    sin_ra = np.sin(ra * np.pi / 180.)
    cos_ra = np.cos(ra * np.pi / 180.)

    sin_dec = np.sin(np.pi / 2 - dec * np.pi / 180.)
    cos_dec = np.cos(np.pi / 2 - dec * np.pi / 180.)

    return  [cos_ra * sin_dec, sin_ra * sin_dec, cos_dec]


def xyz_to_ra_dec(x, y, z):
    """
    Convert back to RA and DEC
    
    dependencies: numpy
    @params
        ra, dec - ndarrays
    @returns
        x, y, z - ndarrays
    """
    xyz = [x, y, z]
    xy = xyz[:,0]**2 + xyz[:,1]**2
    dec = np.arctan2(xyz[:,2], np.sqrt(xy))
    ra = np.arctan2(xyz[:,1], xyz[:,0])
    return ra, dec

    
if __name__ == "__main__":
    main()
