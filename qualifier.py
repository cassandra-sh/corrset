#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qualifier.py
@author Cassandra Henderson
cassandra.s.henderson@gmail.com

* qualifier * -> mpi_prep -> mpi_run -> mpi_read -> jackknifer

Program Use Instructions:
    Instructions are found in qualifier.ini, as well as space for user input.
    
Program Description:
    Prepares the random and data catalogs for use by mpi_prep.
        -sorts data catalog by pixel, redshift
        -gets out pixel and redshift distribution of data
        -normalizes random catalog to large scale distribution of data
        -sorts random catalog by pixel
        
Directory info:

         parent
         │
         ├── corrset
         │   ├── qualifier.py     #This python program
         │   └── ... 
         └── cats
             ├── raw
             ├── matrices
             └── ready            #Where outputs are saved along with meta hdf 
                 ├──meta.hdf
                 ├──C0_D0_zbin1_pix1
                 ├──C0_D0_zbin2_pix1
                 └── ...
 
meta.hdf looks like this:
    hdfs:  'abins' :  (float 'edges'),
           'zbins' :  (float 'edges'),
           'terms' :  (str   'name',  str 'path', int 'num', str 'type'),
           'other' :  (int   'nside')
           'pops'  :  (ints...  terms...)
    
    terms format:
        
    index  name    path        num  type
    1.     'D1D2'  '/path...'  0    'cross'
    2.     'D1R2'  '/path...'  0    'cross'
    3.     'D0D0'  '/path...'  1    'auto'               
    ...    ...     ...         ...  ...
    
    
    corrs format   [indexing by correlation number]
    
    index  type     D0          D1          D2           R0    R1    R2 
    1.     'cross'  ''          '/path...'  '/path...'   ...   ...   ...
    2.     'auto'   '/path...'  ''          ''           ...   ...   ...
    ...    ...      ...         ...         ...          ...   ...   ...
    
"""

import file_manager
import mp_manager
import numpy as np
import os
import time
import hdf_sorter
import pandas as pd

def current_time(start_time):
    return (int(time.time()) - start_time)

def str2bins(string):
    """
    Parses string as a list of floats, comma separated
    
    as in: "1.2,3,6,7" -> [1.2, 3.0, 6.0, 7.0]
    """
    return [float(s) for s in string.split(",")]

def corr_cfg_read(n, cfg):
    """
    Given a corr number and the cfg dictionary from
    file_manager.read_config_file(), read the type of corr, the terms, and the
    file paths of the catalogs used for this corr
    
    @returns
        corr type    ('auto' or 'cross')
        corr terms   (e.g. 'C1D1',   'C1R1',    'C1D2',   'C1R2' for n=1 cross)
        term files   (e.g. /D1/path, /R1/path,  /D2/path, /R2/path)
    """
    #Get current directory information
    directory = os.path.dirname(os.path.realpath(__file__))
    parent = os.path.dirname(directory)
    
    #Figure out what the correlation type is and prepare to read it
    c = str("C" + str(n))
    corr_type  = cfg[c + "_type"].strip(' ')
    terms, dat = [], []
    
    #Read the file paths to the terms specified by the correlation type
    if corr_type == "cross":              # ( D1D2 - D1R2 - D2R1 + R1R2 ) / R1R2
        terms = [c+'D1', c+'D2', c+'R1', c+'R2']
        dat = [cfg[terms[0]], cfg[terms[1]],
               cfg[terms[2]], cfg[terms[3]]]
    elif corr_type == "auto":                            # ( DD - DR + RR ) / RR
        terms = [c+'D0', c+'R0']
        dat = [cfg[terms[0]], cfg[terms[1]]]
        
    else:
        raise ValueError("Provided corr_type in qualifier.ini for n = " +
                         str(n) + " of " + corr_type + " is invalid.")
    
    #For dat, if 'PARENT/' is in the file name, replace with the parent directory
    for i in range(len(dat)):
        if dat[i][0:7] == 'PARENT/':
            dat[i] = dat[i].replace('PARENT', parent)
    
    #Get the miscellaneous information
    misc = [cfg[c+"_str"], cfg[c+"_use_zbins"], cfg[c+"_normalize"]]
    
    #with whether or not to use zbins or normalize, specified by 'y' and 'n'
    for i in [1, 2]:
        if misc[i] == 'y':
            misc[i] = True
        else:
            misc[i] = False

    return corr_type, terms, dat, misc

def meta_hdf():
    """
    Load all of the meta information from qualifier.ini and save to meta.hdf
    
    meta.hdf looks like this:
    hdfs:  'abins' :  (float 'edges'),
           'zbins' :  (float 'edges'),
           'terms' :  (str   'name',  str 'path', int 'num', str 'type'),
           'other' :  (int   'nside')
           'pops'  :  (ints...  terms...)  #(populated after)
    
    terms format:  [indexing by term]
        
    index  name    path          num  type    
    0.     'C0D1'  '/path...'    0    'cross'
    1.     'C0R1'  '/path...'    0    'cross'
    2.     'C0D2'  '/path...'    0    'cross'      
    3.     'C0R2'  '/path...'    0    'cross'      
    4.     'C1D0'  '/path...'    1    'auto'               
    ...    ...     ...           ...  ...
    
    corrs format   [indexing by correlation number]
    
    index  type     D0          D1          D2           R0    R1    R2  
    0.     'cross'  ''          '/path...'  '/path...'   ...   ...   ...
    1.     'auto'   '/path...'  ''          ''           ...   ...   ...
    ...    ...      ...         ...         ...          ...   ...   ...
    
    [continued]
    
    index   str    use_zbins  normalize
    0.      'name'  True       True
    1.      'name'  False      True
    ...     ...        ...
    
    """
    #Step 1: Load up qualifier.ini with the file manager
    directory = os.path.dirname(os.path.realpath(__file__))
    parent = os.path.dirname(directory)
    cfg = file_manager.read_config_file(directory + "/qualifier.ini")
    
    #Step 2: Prepare the meta.hdf HDFStore
    meta_st = pd.HDFStore(parent + "/cats/ready/meta.hdf", mode='w')
    
    #Step 3. Get and append abins and zbins
    meta_st.put("abins", pd.DataFrame(str2bins(cfg['abins']),
                   columns=['abins']), format='table')
    meta_st.put("zbins", pd.DataFrame(str2bins(cfg['zbins']),
                   columns=['zbins']), format='table')
    
    #Step 4. Get the other meta information (n_corrs, nside) and append to meta
    other_df = pd.DataFrame()
    n_corrs = int(cfg['n_corrs'])
    other_df['n_corrs'] = [n_corrs]
    other_df['nside'] = [int(cfg['nside'])]
    meta_st.put('other', other_df, format='table')

    #Step 5. Load up the information about the individual correlations and terms
    term_df_dat = []
    corrs_df_dat = []
    for corr in range(n_corrs):
        ctype, terms, dat, misc = corr_cfg_read(corr, cfg)
        
        #Step 5.1. Get the info for each term of each correlation, ordered by 
        #          term
        for i in range(len(terms)):
            term_df_dat.append([terms[i], dat[i],  corr, ctype])
            
        #Step 5.2. And get the same info, ordered by correlation number
        corrs = [ctype]
        if ctype == 'auto':
            corrs = corrs + [dat[0],     '',     '', dat[1],     '',     '']
        elif ctype == 'cross':
            corrs = corrs + [    '', dat[0], dat[1],     '', dat[2], dat[3]]

        corrs_df_dat.append(corrs+misc)
        
    
    #Step 6. and turn them into DataFrames and save them to the meta hdf
    meta_st.put('terms',  pd.DataFrame(data = term_df_dat,
                columns=['name', 'path', 'num', 'type']), format='table')
    meta_st.put('corrs',   pd.DataFrame(data = corrs_df_dat,
                columns=['type', 'D0', 'D1', 'D2', 'R0', 'R1', 'R2', 'str',
                         'use_zbins', 'normalize']), format='table')
    
    #Step 7. Close the meta HDF
    meta_st.close()
    
def proc_corr(ncorr):
    """
    Process a correlation, by drawing data from the meta hdf, with index ncorr
    """    
    #Get directory and file names
    directory = os.path.dirname(os.path.realpath(__file__))
    parent = os.path.dirname(directory)
    temp_dir = parent + "/cats/temps/"
    output_dir = parent + "/cats/ready/"
    meta_st_address = parent + "/cats/ready/meta.hdf"
    
    """     
    1. Read input data from meta hdf
    """
    meta_st = pd.HDFStore(meta_st_address, mode='r')
    corrs_df = meta_st.get('corrs')
    other = meta_st.get('other')
    zbins = meta_st.get('zbins')
    nside   = other['nside'][0]
    zbins   = zbins['zbins'].tolist()
    meta_st.close()
    
    """
    2. For each real data catalog and associated random catalog in each
           correlation:
               a. Random catalog
                   i.   Normalize the random catalog to the real catalog
                   ii.  Divide by pixels to save. Save a copy for each z bin.
               b. Real catalog
                   i.   Record the number density per pixel and redshift dist.
                   ii.  Divide the data set by z bin and pixel and save
        Also record the population per (broad) pixel, saving to meta hdf
    """
    pop_dct = {}
    zedges_dct = {}
    zhists_dct = {}
    
    r_out_files = []
    d_out_files = []
    if corrs_df['type'][ncorr] == 'auto':
        rold_name = "C" + str(ncorr) + "R0_old"
        dname = "C" + str(ncorr) + "D0"
        rname = "C" + str(ncorr) + "R0"
        
        rpo, dp, rp, do, ro, dzp, rzp, zhist, zedges = (
                              hdf_sorter.proc_pair(corrs_df['D0'][ncorr],
                              corrs_df['R0'][ncorr], temp_dir, output_dir,
                              ncorr, 0, nside, zbins,
                              corrs_df['use_zbins'][ncorr],
                              corrs_df['normalize'][ncorr],
                              normalize_nside=8))
        
        #Record the full catalog populations and redshift distributions
        pop_dct.update({rold_name : rpo, dname : dp, rname : rp})
        zhists_dct.update({dname : zhist}) 
        zedges_dct.update({dname : zedges})
        
        #Record the populations per z bin
        zrange = range(len(zbins)-1)
        if not corrs_df['use_zbins'][ncorr]:
            zrange = [0]
        for z in zrange:
            pop_dct.update({(dname+"_zbin"+str(z)) : dzp[z],
                            (rname+"_zbin"+str(z)) : rzp[z]})
        
        #Record used files
        r_out_files.append(ro)
        d_out_files.append(do)
        
    elif corrs_df['type'][ncorr] == 'cross':
        r1old_name = "C" + str(ncorr) + "R1_old"
        r2old_name = "C" + str(ncorr) + "R2_old"
        d1name, d2name = "C" + str(ncorr) + "D1", "C" + str(ncorr) + "D2"
        r1name, r2name = "C" + str(ncorr) + "R1", "C" + str(ncorr) + "R2"
        
        rpo1, dp1, rp1, do1, ro1, dzp1, rzp1, zhist1, zedges1 = (
                                   hdf_sorter.proc_pair(corrs_df['D1'][ncorr],
                                   corrs_df['R1'][ncorr], temp_dir, output_dir,
                                   ncorr, 1, nside, zbins,
                                   corrs_df['use_zbins'][ncorr],
                                   corrs_df['normalize'][ncorr],
                                   normalize_nside=8))
        rpo2, dp2, rp2, do2, ro2, dzp2, rzp2, zhist2, zedges2 = (
                                   hdf_sorter.proc_pair(corrs_df['D2'][ncorr],
                                   corrs_df['R2'][ncorr], temp_dir, output_dir,
                                   ncorr, 2, nside, zbins,
                                   corrs_df['use_zbins'][ncorr],
                                   corrs_df['normalize'][ncorr],
                                   normalize_nside=8))
        
        #Record the full catalog populations and redshift distributions
        pop_dct.update({r1old_name : rpo1,    r2old_name : rpo2, 
                        d1name     : dp1,     d2name     : dp2,
                        r1name     : rp1,     r2name     : rp2})
        zhists_dct.update({d1name : zhist1,  d2name: zhist2})
        zedges_dct.update({d1name : zedges1, d2name: zedges2})
    
        #Record the populations per z bin
        zrange = range(len(zbins)-1)
        if not corrs_df['use_zbins'][ncorr]:
            zrange = [0]
        for z in zrange:
            pop_dct.update({(d1name+"_zbin"+str(z)) : dzp1[z],
                            (d2name+"_zbin"+str(z)) : dzp2[z],
                            (r1name+"_zbin"+str(z)) : rzp1[z], 
                            (r2name+"_zbin"+str(z)) : rzp2[z]})
        
        #Record used files
        r_out_files.append(ro1)
        d_out_files.append(do1)
        r_out_files.append(ro2)
        d_out_files.append(do2)
    
    #Update the population and z dist hdfs with the already existing data
    meta_st = pd.HDFStore(meta_st_address, mode='r')
    if '/pop' in meta_st.keys():
        old_pop = meta_st.get('pop')
        for col_name in old_pop.columns:
            pop_dct.update({col_name : old_pop[col_name].values})
    if '/zedges' in meta_st.keys():
        old_zedges = meta_st.get('zedges')
        for col_name in old_zedges.columns:
            zedges_dct.update({col_name : old_zedges[col_name].values})
    if '/zhists' in meta_st.keys():
        old_zhists = meta_st.get('zhists')
        for col_name in old_zhists.columns:
            zhists_dct.update({col_name : old_zhists[col_name].values})
    meta_st.close()
    
    #Remove any empty z distributions from zdi_dct (because z is not being used
    #or is not available for this correlation term)
    zhists_keys = list(zhists_dct.keys())
    for i in range(len(zhists_keys)):
        if len(zhists_dct[zhists_keys[i]]) == 0:
            zhists_dct.pop(zhists_keys[i])
            zedges_dct.pop(zhists_keys[i])
    
    #Store our new meta data...
    meta_st = pd.HDFStore(meta_st_address, mode='a')
    meta_st.put('pop', pd.DataFrame.from_dict(pop_dct), format='table')
    meta_st.put('zedges', pd.DataFrame.from_dict(zedges_dct), format='table')
    meta_st.put('zhists', pd.DataFrame.from_dict(zhists_dct), format='table')
    meta_st.close()
    
def main():
    """
     0. Clear out the catalog directory, or make a new one. 
    """
    directory = os.path.dirname(os.path.realpath(__file__))
    parent = os.path.dirname(directory)
    output_dir = parent + "/cats/ready/"
    file_manager.empty_dir(output_dir)
    temp_dir = parent + "/cats/temps/"
    file_manager.empty_dir(temp_dir)
    
    """
     1. Read the config file and make meta.hdf
    """
    meta_hdf()
    meta_st = pd.HDFStore(parent + "/cats/ready/meta.hdf", mode='r')
    corrs_df = meta_st.get('corrs')
    other = meta_st.get('other')
    n_corrs = other['n_corrs'][0]
    meta_st.close()

    """
     2. Process each correlation
     
         Should also get out the names of the saved catalogs and put them in
         the meta hdf
    """
    start_time = int(time.time())
    for n in range(n_corrs):
        print("\n\nProcessing correlation " + str(n))
        print("Time is " + str(current_time(start_time)))
        print("Correlation info is \n" + str(corrs_df.iloc[n]))
        proc_corr(n)
    
    """
     3. As a sanity check, print the contents of meta.hdf
    """
    meta_st = pd.HDFStore(parent + "/cats/ready/meta.hdf", mode='r')
    for key in meta_st.keys():
        print(meta_st.get(key))
    meta_st.close()
    
if __name__ == "__main__":
    main()