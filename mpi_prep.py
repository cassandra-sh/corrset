#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mpi_prep.py
@author Cassandra Henderson
cassandra.s.henderson@gmail.com

qualifier -> * mpi_prep * -> mpi_run -> mpi_read -> jackknifer


Program Description:
    Using the output of qualifier, prepares pixel-pixel pairs of information
    between the relevant data sets and pickles them for mpi_run to process.
    
    End result should be a directory with a list of pickled job inputs and a
    meta job input, ready to send to a Slurm computing cluster for mpi_run.py
    to process.

Directory info:
    
         parent 
         │
         ├── cats
         │   ├── ready         #Input of this file
         │   └── ...     
         ├── corrset
         │   ├── mpi_prep.py   #This file
         │   └── ...
         └── mpi_jobs          #Output of this file
             ├── jj
             ├── j1
             └── ...     
"""

import os
import numpy as np
import pickle
import pandas as pd
import file_manager
import gen_pairs

def main(job_suf=""):
    """
    @params
        job_suf - suffix for job directory
                  alternate for different job directories
                  might code this into qualifier.ini but for now its
                  hardcoded
    Steps:
        
    0. Clear out the job directory, or make a new one for the new jobs.
    
    1. Get meta information from meta.hdf
        
    2. Use meta info to figure out what pair counts need to be done, and
       where the catalogs that contain the relevant data are.  
               
    3. Load the desired catalogs for each pixel-pixel pair count job and
       pickle for mpi_run.py along with information regarding where to save
       and angular bins. Should pickle dictionaries of the form:
       
       dictionary = {'out'         : str  'filename_out',  #name, not path
                     'ra1'         : list [float ra1  values],
                     'dec1'        : list [float dec1 values],
                     'ra2'         : list [float ra2  values],
                     'dec2'        : list [float dec2 values],
                     'angular_bins': list [float bin  values]}
       
       out format is 'C#[D/R]#[D/R]#z#pp#_#'
       
    4. Save meta information for mpi_run, mainly the number of total jobs
       to be done, and the number of (possible) pairs per job, for estimating
       which jobs will take the longest. 
    """
    
    #0. Clearing output directory
    directory = os.path.dirname(os.path.realpath(__file__))
    parent = os.path.dirname(directory)
    job_dir = parent + ("/mpi_jobs"+job_suf+"/")
    out_dir = parent + ("/mpi_outs"+job_suf+"/")
    file_manager.empty_dir(job_dir)
    file_manager.empty_dir(out_dir)
    input_dir = parent + "/cats/ready/"
    
    #1. Getting meta information
    meta_st = pd.HDFStore(parent + "/cats/ready/meta.hdf", mode='r')
    pix_pops = meta_st.get('pop')
    corrs    = meta_st.get('corrs')            
    other = meta_st.get('other')
    nside   = other['nside'][0]
    zbins   = meta_st.get('zbins')['zbins'].tolist()
    abins   = meta_st.get('abins')['abins'].tolist()
    meta_st.close()
    
    #2. Figure out what jobs need to be prepared
    #   This is just a somewhat complex filename generating job
    pairs = gen_pairs.pixpairs(nside)
    jobs = []   #list of [name, path1, path2]
    
    for index, row in corrs.iterrows():
        corr_number = index
        use_zbins   = row['use_zbins']
        corr_type   = row['type']
            
        print("")
        print(row)
            
        if corr_type == "auto":
            #Only use a pixel if it has sources for data and random
            d_pref = ("C"+str(corr_number)+"D0")
            r_pref = ("C"+str(corr_number)+"R0")
            pixels_good = np.logical_and(np.greater(pix_pops[d_pref], 0),
                                         np.greater(pix_pops[r_pref], 0))
            

            successes = 0            
            for pair in pairs:
                if pixels_good[pair[0]] and pixels_good[pair[1]]:
                    zbins_to_use = [0]
                    if use_zbins: zbins_to_use = range(len(zbins)-1)
                    for zbin in zbins_to_use:
                        
                        # Check whether the files associated with this pixel
                        # pixel pair all exist for this redshift bin.
                        # (should have been ensured by pixels_good)
                        all_files_ok = True
                        for file_name in [(d_pref+"_zbin"+str(zbin)+
                                             "_pix"+str(pair[0])),
                                          (r_pref+"_zbin"+str(zbin)+
                                             "_pix"+str(pair[1]))]:
                            if not os.path.isfile(input_dir + file_name):
                                print("file " + file_name + " is not available")
                                all_files_ok = False
                        if not all_files_ok:
                            print("pair " + str(pair) + ", zbin " + str(zbin) +
                                  ", is missing files!")
                            print("")
                            pass
                        
                        successes = successes + 1
                        
                        # Come up with the appropriate term jobs and add to list
                        jobs.append([('C'+str(corr_number)+'D0D0z'+str(zbin)+
                                     'pp'+str(pair[0])+"_"+str(pair[1])+".npy"),
                                     (d_pref+"_zbin"+str(zbin)+
                                             "_pix"+str(pair[0])),
                                     (d_pref+"_zbin"+str(zbin)+
                                             "_pix"+str(pair[1]))])
                        jobs.append([('C'+str(corr_number)+'D0R0z'+str(zbin)+
                                     'pp'+str(pair[0])+"_"+str(pair[1])+".npy"),
                                     (d_pref+"_zbin"+str(zbin)+
                                             "_pix"+str(pair[0])),
                                     (r_pref+"_zbin"+str(zbin)+
                                             "_pix"+str(pair[1]))])
                        jobs.append([('C'+str(corr_number)+'R0R0z'+str(zbin)+
                                     'pp'+str(pair[0])+"_"+str(pair[1])+".npy"),
                                     (r_pref+"_zbin"+str(zbin)+
                                             "_pix"+str(pair[0])),
                                     (r_pref+"_zbin"+str(zbin)+
                                             "_pix"+str(pair[1]))])
            print("successes = " + str(successes))
            print("")                                     

        elif corr_type == "cross":
            #Only use a pixel if it has sources for data and random in all 4
            #catalogs (D1, D2, R1, R2)
            d1_pref = ("C"+str(corr_number)+"D1")
            r1_pref = ("C"+str(corr_number)+"R1")
            d2_pref = ("C"+str(corr_number)+"D2")
            r2_pref = ("C"+str(corr_number)+"R2")
            
            pixels_good = np.logical_and(np.logical_and(
                                              np.greater(pix_pops[d1_pref], 0),
                                              np.greater(pix_pops[r1_pref], 0)),
                                         np.logical_and(
                                              np.greater(pix_pops[d2_pref], 0),
                                              np.greater(pix_pops[r2_pref], 0)))
            
            print("")
            print(row)
            successes = 0
            for pair in pairs:
                if pixels_good[pair[0]] and pixels_good[pair[1]]:
                    zbins_to_use = [0]
                    if use_zbins: zbins_to_use = range(len(zbins)-1)
                    for zbin in zbins_to_use:
                        
                        # Check whether the files associated with this pixel
                        # pixel pair all exist for this redshift bin.
                        # (should have been ensured by pixels_good)
                        all_files_ok = True
                        for file_name in [(d1_pref+"_zbin"+str(zbin)+
                                             "_pix"+str(pair[0])),
                                          (d2_pref+"_zbin"+str(zbin)+
                                             "_pix"+str(pair[1])),
                                          (r1_pref+"_zbin"+str(zbin)+
                                             "_pix"+str(pair[0])),
                                          (r2_pref+"_zbin"+str(zbin)+
                                             "_pix"+str(pair[1]))]:
                            if not os.path.isfile(input_dir + file_name):
                                print("file " + file_name + " is not available")
                                all_files_ok = False
                        if not all_files_ok:
                            print("pair " + str(pair) + ", zbin " + str(zbin) +
                                  ", is missing files!")
                            print("")
                            pass
                        successes = successes + 1
                        
                        # Come up with the appropriate term jobs and add to list
                        jobs.append([('C'+str(corr_number)+'D1D2z'+str(zbin)+
                                     'pp'+str(pair[0])+"_"+str(pair[1])+".npy"),
                                     (d1_pref+"_zbin"+str(zbin)+
                                             "_pix"+str(pair[0])),
                                     (d2_pref+"_zbin"+str(zbin)+
                                             "_pix"+str(pair[1]))])
                        jobs.append([('C'+str(corr_number)+'D1R2z'+str(zbin)+
                                     'pp'+str(pair[0])+"_"+str(pair[1])+".npy"),
                                     (d1_pref+"_zbin"+str(zbin)+
                                             "_pix"+str(pair[0])),
                                     (r2_pref+"_zbin"+str(zbin)+
                                             "_pix"+str(pair[1]))])
                        jobs.append([('C'+str(corr_number)+'D2R1z'+str(zbin)+
                                     'pp'+str(pair[0])+"_"+str(pair[1])+".npy"),
                                     (d2_pref+"_zbin"+str(zbin)+
                                             "_pix"+str(pair[0])),
                                     (r1_pref+"_zbin"+str(zbin)+
                                             "_pix"+str(pair[1]))])
                        jobs.append([('C'+str(corr_number)+'R1R2z'+str(zbin)+
                                     'pp'+str(pair[0])+"_"+str(pair[1])+".npy"),
                                     (r1_pref+"_zbin"+str(zbin)+
                                             "_pix"+str(pair[0])),
                                     (r2_pref+"_zbin"+str(zbin)+
                                             "_pix"+str(pair[1]))])
            print("successes = " + str(successes))
            print("")
    
    #3. Load up and pickle the job inputs
    time_index = []
    n_jobs = 0
    for n in range(len(jobs)):
        d1, d2 = [], []
        
        try:
            d1 = pd.read_hdf((input_dir + jobs[n][1]), key='primary')
        except FileNotFoundError:
            print("PROBLEM!!! with " + str(jobs[n]) + " d2 " +
                  str((input_dir + jobs[n][1])))
        
        try:
            d2 = pd.read_hdf((input_dir + jobs[n][2]), key='primary')
        except FileNotFoundError:
            print("PROBLEM!!! with " + str(jobs[n]) + " d2 " +
                  str((input_dir + jobs[n][2])))
        
        if len(d1) == 0 or len(d2) == 0:
            print("Unable to prepare job " + str(n))
            continue
        
        time_index.append(len(d1)*len(d2))
        f = open((job_dir+"j"+str(n_jobs)), 'wb')
        pickle.dump({'out'         : jobs[n][0],  'angular_bins': abins,
                     'ra1'         : d1['ra'],    'dec1'        : d1['dec'],
                     'ra2'         : d2['ra'],    'dec2'        : d2['dec']}, f)
        n_jobs = n_jobs + 1
        f.close()
        
    print("Of " + str(len(jobs)) + " prepared jobs, " + str(n_jobs) + " had " +
          "existing file inputs.")
    
    
    #4. Save job meta info to job dir
    f = open((job_dir + "jj"), 'wb')
    pickle.dump({'n_jobs' : n_jobs, 'time_index' : time_index}, f)
    f.close

if __name__ == "__main__":
    main()