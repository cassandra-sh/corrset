#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mpi_read.py
@author Cassandra Henderson
cassandra.s.henderson@gmail.com

qualifier -> mpi_prep -> mpi_run -> * mpi_read * -> jackknifer


Program Description:
    Read the mpi outputs. You'll probably have to download them from wherever
    you ran the mpi_run jobs.
    
    Steps:
        0. Clear 
        
        1.  Get meta information from meta.hdf
        
        2.  Load up the count numbers from the pixel count storage HDF prepared
            by the qualifier. 
        
        3.  Make CountMatrix objects for storing the output
        
        4.  Read all the job outputs and parse their names. Should follow
            expected info from qualifier_meta.hdf. Save results to CountMatrix
            objects.
        
        5.  Save CountMatrix objects
           
Directory info:

         parent
         │
         ├── corrset
         │   ├── mpi_read.py
         │   └── ...
         ├── cats
         │   ├── raw
         │   ├── matrices     #Outputs from mpi_read.py get saved here
         │   └── ready
         └── mpi_outs/        #Inputs for mpi_read.py go here
             ├── out0.npy
             ├── out1.npy
             └── ...
    
"""
from itertools           import combinations
from scipy.optimize      import curve_fit
import matplotlib.pyplot as plt
import CountMatrix       as cm
import pandas            as pd
import numpy             as np
import file_manager
import camb_model
import email_bot
import gen_pairs
import pickle
import shutil
import camb
import glob
import time
import os
import gc

def add_to_cm(cm, path, pair):
    """
    Load a numpy array from the given path and add it to the given CountMatrix
    at the given pixel location. Skip if the array does not exist
    
    @params
        cm   - CountMatrix object with same nside as pair
        path - Path to numpy file to try to load
        pair - pixel-pixel pair this path correpsonds to
    """
    arr = []
    
    if os.path.isfile(path+".npy"):
        f = open(path+".npy", mode='rb')
        arr = np.load(f)
        f.close()
        f = None
    elif os.path.isfile(path):
        f = open(path, mode='rb')
        arr = np.load(f)
        f.close()
        f = None
    else:
        print("file_not_found: " + path)
        return 0
    
    cm.add(arr, pair)
    arr = None
    gc.collect()
    return 1


def current_time(start_time):
    return (int(time.time()) - start_time)


def just_read_jobs(job_suf=""):
    """
    @params
        job_suf - suffix for job/out directory
                  alternate for different job directories
                  might code this into qualifier.ini but for now its
                  hardcoded
        
    """
    #Getting output directory, clearing CountMatrix output directory 
    directory = os.path.dirname(os.path.realpath(__file__))
    parent    = os.path.dirname(directory)
    out_dir   = parent + ("/mpi_outs"+job_suf+"/")
    cm_dir    = parent + ("/cats/matrices"+job_suf+"/")
    file_manager.empty_dir(cm_dir)
    
    #Getting meta information
    meta_st  = pd.HDFStore(parent + "/cats/ready/meta.hdf", mode='r')
    
    pix_pops = meta_st.get('pop')
    corrs    = meta_st.get('corrs')      
    zhists   = meta_st.get('zhists')     
    zedges   = meta_st.get('zedges')            
    other    = meta_st.get('other')
    zbins    = meta_st.get('zbins')['zbins'].tolist()
    abins    = np.array(meta_st.get('abins')['abins'].tolist(), dtype=float)
    nside    = other['nside'][0]
    meta_st.close()
    
    #Prepare to keep track of time
    start_time = int(time.time())

    #Figure out what jobs were done and read them out
    pairs = gen_pairs.pixpairs(nside)
    for index, row in corrs.iterrows():        
        #Print the row
        print("Reading correlation "+str(index)+". Row information follows.")
        print(row)
        print("Time is " + str(current_time(start_time)))
        
        #Keep track of the number of files successfully read in this correlation
        files_read = 0
        
        #Get information associated with this correlation
        corr_number = index
        use_zbins   = row['use_zbins']
        corr_type   = row['type']
        
        #Autocorrelation routine
        if corr_type == "auto":
            
            #Determine file names
            d_pref = ("C"+str(corr_number)+"D0")
            r_pref = ("C"+str(corr_number)+"R0")
            
            #Only use a pixel if it has sources for data and random
            pixels_good = np.logical_and(np.greater(pix_pops[d_pref], 0),
                                         np.greater(pix_pops[r_pref], 0))
            
            #Get the redshift distribution information to include in the
            #count matrices
            z_hist, z_edges = [], []
            if d_pref in zhists:
                z_hist  = zhists[d_pref]
                z_edges = zedges[d_pref]
            
            #Determine zbins to use and iterate through them
            zbins_to_use = [0]
            if use_zbins: zbins_to_use = range(len(zbins)-1)
            for zbin in zbins_to_use:
                
                #Get population specific to this z bin
                dpop = pix_pops[("C"+str(corr_number)+"D0_zbin"+str(zbin))]
                rpop = pix_pops[("C"+str(corr_number)+"R0_zbin"+str(zbin))]
                
                #Determine names for the count matrices
                ddcm = ("C"+str(corr_number)+"D0D0_zbin"+str(zbin)+"mat")
                drcm = ("C"+str(corr_number)+"D0R0_zbin"+str(zbin)+"mat")
                rrcm = ("C"+str(corr_number)+"R0R0_zbin"+str(zbin)+"mat")
                
                #Build count matrices
                ddcm = cm.CountMatrix(nside = nside, filename = (cm_dir+ddcm),
                                      tabname = ddcm, abins = abins,
                                      pop1= dpop.values, pop2 = dpop.values,
                                      z_hist1 = z_hist, z_edges1 = z_edges)
                
                drcm = cm.CountMatrix(nside = nside, filename = (cm_dir+drcm),
                                      tabname = drcm, abins = abins,
                                      pop1 = dpop.values, pop2 = rpop.values)
                
                rrcm = cm.CountMatrix(nside = nside, filename = (cm_dir+rrcm),
                                      tabname = rrcm, abins = abins,
                                      pop1 = rpop.values, pop2 = rpop.values)
                
                #Get out correlation results per valid pair of pixels
                for pair in pairs:
                    if pixels_good[pair[0]] and pixels_good[pair[1]]:                        
                        #Determine file names of mpi_run outputs
                        dd = (out_dir+'C'+str(corr_number)+'D0D0z'+str(zbin)+
                                     'pp'+str(pair[0])+"_"+str(pair[1]))
                        dr = (out_dir+'C'+str(corr_number)+'D0R0z'+str(zbin)+
                                     'pp'+str(pair[0])+"_"+str(pair[1]))
                        rr = (out_dir+'C'+str(corr_number)+'R0R0z'+str(zbin)+
                                     'pp'+str(pair[0])+"_"+str(pair[1]))
                        
                        #LOAD and RECORD pair counting info into CountMatrices
                        #Make sure to except FileNotFound errors
                        dd_read = add_to_cm(ddcm, dd, pair)
                        dr_read = add_to_cm(drcm, dr, pair)
                        rr_read = add_to_cm(rrcm, rr, pair)
                        
                        files_read = files_read + dd_read + dr_read + rr_read
                        gc.collect()
                
                #SAVE CountMatrix objects
                ddcm.save(dump = True)
                drcm.save(dump = True)
                rrcm.save(dump = True)
                
                #Close the count matrix objects and clear them
                ddcm, drcm, rrcm = 0, 0, 0
                gc.collect()
                        
        #Cross correlation routine
        elif corr_type == "cross":
            
            #Determine file names
            d1_pref = ("C"+str(corr_number)+"D1")
            r1_pref = ("C"+str(corr_number)+"R1")
            d2_pref = ("C"+str(corr_number)+"D2")
            r2_pref = ("C"+str(corr_number)+"R2") 
            
            #Only use a pixel if it has sources for data and random in all 4
            #catalogs (D1, D2, R1, R2)
            pixels_good = np.logical_and(np.logical_and(
                                              np.greater(pix_pops[d1_pref], 0),
                                              np.greater(pix_pops[r1_pref], 0)),
                                         np.logical_and(
                                              np.greater(pix_pops[d2_pref], 0),
                                              np.greater(pix_pops[r2_pref], 0)))
            
            #Get the redshift distribution information to include in the
            #count matrices
            z_hist1, z_edges1, z_hist2, z_edges2 = [], [], [], []
            if d1_pref in zhists:
                z_hist1  = zhists[d1_pref]
                z_edges1 = zedges[d1_pref]
                z_hist2  = zhists[d2_pref]
                z_edges2 = zedges[d2_pref]
            
            #Determine zbins to use and iterate through them
            zbins_to_use = [0]
            if use_zbins: zbins_to_use = range(len(zbins)-1)
            for zbin in zbins_to_use:
                
                #Get population specific to this z bin
                d1pop = pix_pops[("C"+str(corr_number)+"D1_zbin"+str(zbin))]
                r1pop = pix_pops[("C"+str(corr_number)+"R1_zbin"+str(zbin))]
                d2pop = pix_pops[("C"+str(corr_number)+"D2_zbin"+str(zbin))]
                r2pop = pix_pops[("C"+str(corr_number)+"R2_zbin"+str(zbin))]
                
                #Determine names for the count matrices
                d1d2cm = ("C"+str(corr_number)+"D1D2_zbin"+str(zbin)+"mat")
                d1r2cm = ("C"+str(corr_number)+"D1R2_zbin"+str(zbin)+"mat")
                d2r1cm = ("C"+str(corr_number)+"D2R1_zbin"+str(zbin)+"mat")
                r1r2cm = ("C"+str(corr_number)+"R1R2_zbin"+str(zbin)+"mat")
                
                #Build count matrices
                d1d2cm = cm.CountMatrix(nside = nside, filename=(cm_dir+d1d2cm),
                                        tabname = d1d2cm, abins = abins,
                                        pop1= d1pop.values, pop2= d2pop.values,
                                        z_hist1=z_hist1, z_hist2 = z_hist2,
                                        z_edges1=z_edges1, z_edges2=z_edges2)
                
                d1r2cm = cm.CountMatrix(nside = nside, filename=(cm_dir+d1r2cm),
                                        tabname = d1r2cm, abins = abins,
                                        pop1= d1pop.values, pop2= r2pop.values)
                
                d2r1cm = cm.CountMatrix(nside = nside, filename=(cm_dir+d2r1cm),
                                        tabname = d2r1cm, abins = abins,
                                        pop1= d2pop.values, pop2= r1pop.values)
                
                r1r2cm = cm.CountMatrix(nside = nside, filename=(cm_dir+r1r2cm),
                                        tabname = r1r2cm, abins = abins,
                                        pop1= r1pop.values, pop2= r2pop.values)
                
                #Get out correlation results per valid pair of pixels
                for pair in pairs:
                    if pixels_good[pair[0]] and pixels_good[pair[1]]:
                        #Determine file names of mpi_run outputs
                        d1d2 = (out_dir+'C'+str(corr_number)+'D1D2z'+str(zbin)+
                                     'pp'+str(pair[0])+"_"+str(pair[1]))
                        d1r2 = (out_dir+'C'+str(corr_number)+'D1R2z'+str(zbin)+
                                     'pp'+str(pair[0])+"_"+str(pair[1]))
                        d2r1 = (out_dir+'C'+str(corr_number)+'D2R1z'+str(zbin)+
                                     'pp'+str(pair[0])+"_"+str(pair[1]))
                        r1r2 = (out_dir+'C'+str(corr_number)+'R1R2z'+str(zbin)+
                                     'pp'+str(pair[0])+"_"+str(pair[1]))
                        
                        #LOAD and RECORD pair counting info into CountMatrices
                        #Make sure to except FileNotFound errors
                        d1d2_read = add_to_cm(d1d2cm, d1d2, pair)
                        d1r2_read = add_to_cm(d1r2cm, d1r2, pair)
                        d2r1_read = add_to_cm(d2r1cm, d2r1, pair)
                        r1r2_read = add_to_cm(r1r2cm, r1r2, pair)
                        
                        files_read = (files_read + d1d2_read + d1r2_read +
                                                   d2r1_read + r1r2_read)
                        
                        gc.collect()
                        
                        
                    
                #SAVE CountMatrix objects
                d1d2cm.save(dump = True)
                d1r2cm.save(dump = True)
                d2r1cm.save(dump = True)
                r1r2cm.save(dump = True)
                
                #Close the count matrix objects and clear them
                d1d2cm, d1r2cm, d2r1cm, r1r2cm = 0, 0, 0, 0
                gc.collect()

    
    
    
def read_jobs(job_suf="", mods=False, ERRORBARS=False):
    """
    @params
        job_suf - suffix for job/out directory
                  alternate for different job directories
                  might code this into qualifier.ini but for now its
                  hardcoded
        
    """
    #Getting output directory, clearing CountMatrix output directory 
    directory = os.path.dirname(os.path.realpath(__file__))
    parent    = os.path.dirname(directory)
    out_dir   = parent + ("/mpi_outs"+job_suf+"/")
    cm_dir    = parent + ("/cats/matrices"+job_suf+"/")
    file_manager.empty_dir(cm_dir)
    
    #Getting meta information
    meta_st  = pd.HDFStore(parent + "/cats/ready/meta.hdf", mode='r')
    
    pix_pops = meta_st.get('pop')
    corrs    = meta_st.get('corrs')      
    zhists   = meta_st.get('zhists')     
    zedges   = meta_st.get('zedges')            
    other    = meta_st.get('other')
    zbins    = meta_st.get('zbins')['zbins'].tolist()
    abins    = np.array(meta_st.get('abins')['abins'].tolist(), dtype=float)
    nside    = other['nside'][0]
    meta_st.close()

    #Prep some arrays to store the correlation signals
    cnums, zbins_used, names, signals, dms, dmrs, errors = [], [], [], [], [], [], []
    d1fs, d2fs = [], []
    blurbs = []
    
    #Prepare to keep track of time
    start_time = int(time.time())

    #Figure out what jobs were done and read them out
    pairs = gen_pairs.pixpairs(nside)
    for index, row in corrs.iterrows():        
        #Print the row
        print("Reading correlation "+str(index)+". Row information follows.")
        print(row)
        print("Time is " + str(current_time(start_time)))
        
        #Keep track of the number of files successfully read in this correlation
        files_read = 0
        
        #Get information associated with this correlation
        corr_number = index
        use_zbins   = row['use_zbins']
        corr_type   = row['type']
        name        = row['str']
        
        d1f, d2f = "", ""
        
        #Autocorrelation routine
        if corr_type == "auto":
            
            d1f, d2f = row['D0'], row['D0']
            
            #Determine file names
            d_pref = ("C"+str(corr_number)+"D0")
            r_pref = ("C"+str(corr_number)+"R0")
            
            #Only use a pixel if it has sources for data and random
            pixels_good = np.logical_and(np.greater(pix_pops[d_pref], 0),
                                         np.greater(pix_pops[r_pref], 0))
            
            #Get the redshift distribution information to include in the
            #count matrices
            z_hist, z_edges = [], []
            if d_pref in zhists:
                z_hist  = zhists[d_pref]
                z_edges = zedges[d_pref]
            
            #Determine zbins to use and iterate through them
            zbins_to_use = [0]
            if use_zbins: zbins_to_use = range(len(zbins)-1)
            for zbin in zbins_to_use:
                
                #Get population specific to this z bin
                dpop = pix_pops[("C"+str(corr_number)+"D0_zbin"+str(zbin))]
                rpop = pix_pops[("C"+str(corr_number)+"R0_zbin"+str(zbin))]
                
                #Determine names for the count matrices
                ddcm = ("C"+str(corr_number)+"D0D0_zbin"+str(zbin)+"mat")
                drcm = ("C"+str(corr_number)+"D0R0_zbin"+str(zbin)+"mat")
                rrcm = ("C"+str(corr_number)+"R0R0_zbin"+str(zbin)+"mat")
                
                #Build count matrices
                ddcm = cm.CountMatrix(nside = nside, filename = (cm_dir+ddcm),
                                      tabname = ddcm, abins = abins,
                                      pop1= dpop.values, pop2 = dpop.values,
                                      z_hist1 = z_hist, z_edges1 = z_edges)
                
                drcm = cm.CountMatrix(nside = nside, filename = (cm_dir+drcm),
                                      tabname = drcm, abins = abins,
                                      pop1 = dpop.values, pop2 = rpop.values)
                
                rrcm = cm.CountMatrix(nside = nside, filename = (cm_dir+rrcm),
                                      tabname = rrcm, abins = abins,
                                      pop1 = rpop.values, pop2 = rpop.values)
                
                print(zbin)
                
                #Get out correlation results per valid pair of pixels
                for pair in pairs:
                    if pixels_good[pair[0]] and pixels_good[pair[1]]:
                        
                        print(pair)
                        
                        #Determine file names of mpi_run outputs
                        dd = (out_dir+'C'+str(corr_number)+'D0D0z'+str(zbin)+
                                     'pp'+str(pair[0])+"_"+str(pair[1]))
                        dr = (out_dir+'C'+str(corr_number)+'D0R0z'+str(zbin)+
                                     'pp'+str(pair[0])+"_"+str(pair[1]))
                        rr = (out_dir+'C'+str(corr_number)+'R0R0z'+str(zbin)+
                                     'pp'+str(pair[0])+"_"+str(pair[1]))
                        
                        #LOAD and RECORD pair counting info into CountMatrices
                        #Make sure to except FileNotFound errors
                        dd_read = add_to_cm(ddcm, dd, pair)
                        
                        print(dd)
                        
                        dr_read = add_to_cm(drcm, dr, pair)
                        
                        print(dr)
                        
                        rr_read = add_to_cm(rrcm, rr, pair)
                        
                        print(rr)
                        
                        files_read = files_read + dd_read + dr_read + rr_read
                        
                        gc.collect()
                
                #SAVE CountMatrix objects
                ddcm.save()
                drcm.save()
                rrcm.save()
                
                
                #generate a blurb about this correlation, including source and
                #number of objects
                dnum, rnum = drcm.population()
                blurb = ("Corr: "+str(name)+"\n"+"zbin: ("+str(zbins[zbin])+
                         ", "+str(zbins[zbin+1])+")\n"+
                         "D0 = "+str(row['D0'])+"  num = "+str(dnum)+"\n"+
                         "R0 = "+str(row['R0'])+"  num = "+str(rnum)+"\n")
                blurbs.append(blurb)
                
                
                if ERRORBARS:
                    parts, areas = rrcm.partition()
                    
        
                    #
                    # Get every combination with one area removed 
                    #
                    parts_tupled = [tuple(parts[i]) for i in range(len(parts))]
                    perms = set()
                    for perm in combinations(parts_tupled, 1):
                        perms.add(perm)
                    parts = []
                    for perm in perms:
                        part_to_add = []
                        for tup in perm:
                            for index in tup:
                                part_to_add.append(index)
                        parts.append(part_to_add)
                        
                    
                    #
                    # Figure out the area associated with each combination
                    #
                    r1_areas, r2_areas = rrcm.pix_area()
                    
                    areas = np.zeros(len(parts))
                    for i in range(len(parts)):
                        for j in range(len(parts[i])):
                            areas[i] = areas[i] + r1_areas[parts[i][j]]
                    areas = np.sum(r1_areas)-areas
                    
                    areas = np.ones(len(parts), dtype=float)
                    
                    
                    dd_s = [ddcm.get_signal(cut=part) for part in parts]
                    dr_s = [drcm.get_signal(cut=part) for part in parts]
                    rr_s = [rrcm.get_signal(cut=part) for part in parts]
                    
                    signal = [((dd_s[i]-2*dr_s[i]+rr_s[i])/rr_s[i]) for i in range(len(dd_s))]
                    
                    signal_mean = np.average(signal, axis=0, weights=areas)
                    signal_std = np.sqrt(np.average((signal-signal_mean)**2, axis=0, weights=areas))
                    
                    signals.append(signal_mean)
                    errors.append(signal_std)
                    
                    
                    print(signal_mean)
                    print(signal_std)
                    gc.collect()
                    
                    
                else:
                    dd = ddcm.get_signal()
                    dr = drcm.get_signal()
                    rr = rrcm.get_signal()
                    
                    signals.append((dd - 2*dr + rr)/rr)
                    errors.append(np.zeros(len(dd)))
                    gc.collect()
                    
                    
                zbins_used.append(zbin)
                names.append(name)
                cnums.append(corr_number)
                
                if mods:
                    dm, dmr = ddcm.get_mod(min_z = zbins[zbin],
                                           max_z = zbins[zbin+1])
                    dms.append(dm)
                    dmrs.append(dmr)
                    
                #Close the count matrix objects and clear them
                ddcm, drcm, rrcm = 0, 0, 0
                gc.collect()
                        
        #Cross correlation routine
        elif corr_type == "cross":
            
            d1f, d2f = row['D1'], row['D2']
            
            #Determine file names
            d1_pref = ("C"+str(corr_number)+"D1")
            r1_pref = ("C"+str(corr_number)+"R1")
            d2_pref = ("C"+str(corr_number)+"D2")
            r2_pref = ("C"+str(corr_number)+"R2") 
            
            #Only use a pixel if it has sources for data and random in all 4
            #catalogs (D1, D2, R1, R2)
            pixels_good = np.logical_and(np.logical_and(
                                              np.greater(pix_pops[d1_pref], 0),
                                              np.greater(pix_pops[r1_pref], 0)),
                                         np.logical_and(
                                              np.greater(pix_pops[d2_pref], 0),
                                              np.greater(pix_pops[r2_pref], 0)))
            
            #Get the redshift distribution information to include in the
            #count matrices
            z_hist1, z_edges1, z_hist2, z_edges2 = [], [], [], []
            if d1_pref in zhists:
                z_hist1  = zhists[d1_pref]
                z_edges1 = zedges[d1_pref]
                z_hist2  = zhists[d2_pref]
                z_edges2 = zedges[d2_pref]
            
            #Determine zbins to use and iterate through them
            zbins_to_use = [0]
            if use_zbins: zbins_to_use = range(len(zbins)-1)
            for zbin in zbins_to_use:
                
                #Get population specific to this z bin
                d1pop = pix_pops[("C"+str(corr_number)+"D1_zbin"+str(zbin))]
                r1pop = pix_pops[("C"+str(corr_number)+"R1_zbin"+str(zbin))]
                d2pop = pix_pops[("C"+str(corr_number)+"D2_zbin"+str(zbin))]
                r2pop = pix_pops[("C"+str(corr_number)+"R2_zbin"+str(zbin))]
                
                #Determine names for the count matrices
                d1d2cm = ("C"+str(corr_number)+"D1D2_zbin"+str(zbin)+"mat")
                d1r2cm = ("C"+str(corr_number)+"D1R2_zbin"+str(zbin)+"mat")
                d2r1cm = ("C"+str(corr_number)+"D2R1_zbin"+str(zbin)+"mat")
                r1r2cm = ("C"+str(corr_number)+"R1R2_zbin"+str(zbin)+"mat")
                
                #Build count matrices
                d1d2cm = cm.CountMatrix(nside = nside, filename=(cm_dir+d1d2cm),
                                        tabname = d1d2cm, abins = abins,
                                        pop1= d1pop.values, pop2= d2pop.values,
                                        z_hist1=z_hist1, z_hist2 = z_hist2,
                                        z_edges1=z_edges1, z_edges2=z_edges2)
                
                d1r2cm = cm.CountMatrix(nside = nside, filename=(cm_dir+d1r2cm),
                                        tabname = d1r2cm, abins = abins,
                                        pop1= d1pop.values, pop2= r2pop.values)
                
                d2r1cm = cm.CountMatrix(nside = nside, filename=(cm_dir+d2r1cm),
                                        tabname = d2r1cm, abins = abins,
                                        pop1= d2pop.values, pop2= r1pop.values)
                
                r1r2cm = cm.CountMatrix(nside = nside, filename=(cm_dir+r1r2cm),
                                        tabname = r1r2cm, abins = abins,
                                        pop1= r1pop.values, pop2= r2pop.values)
                
                #Get out correlation results per valid pair of pixels
                for pair in pairs:
                    if pixels_good[pair[0]] and pixels_good[pair[1]]:
                        #Determine file names of mpi_run outputs
                        d1d2 = (out_dir+'C'+str(corr_number)+'D1D2z'+str(zbin)+
                                     'pp'+str(pair[0])+"_"+str(pair[1]))
                        d1r2 = (out_dir+'C'+str(corr_number)+'D1R2z'+str(zbin)+
                                     'pp'+str(pair[0])+"_"+str(pair[1]))
                        d2r1 = (out_dir+'C'+str(corr_number)+'D2R1z'+str(zbin)+
                                     'pp'+str(pair[0])+"_"+str(pair[1]))
                        r1r2 = (out_dir+'C'+str(corr_number)+'R1R2z'+str(zbin)+
                                     'pp'+str(pair[0])+"_"+str(pair[1]))
                        
                        #LOAD and RECORD pair counting info into CountMatrices
                        #Make sure to except FileNotFound errors
                        d1d2_read = add_to_cm(d1d2cm, d1d2, pair)
                        d1r2_read = add_to_cm(d1r2cm, d1r2, pair)
                        d2r1_read = add_to_cm(d2r1cm, d2r1, pair)
                        r1r2_read = add_to_cm(r1r2cm, r1r2, pair)
                        
                        files_read = (files_read + d1d2_read + d1r2_read +
                                                   d2r1_read + r1r2_read)
                        
                        gc.collect()
                        
                        
                    
                #SAVE CountMatrix objects
                d1d2cm.save()
                d1r2cm.save()
                d2r1cm.save()
                r1r2cm.save()
                
                
                #generate a blurb about this correlation, including source and
                #number of objects
                d1num, r2num = d1r2cm.population()
                d2num, r1num = d2r1cm.population()
                blurb = ("Corr: "+str(name)+"\n"+"zbin: ("+str(zbins[zbin])+
                         ", "+str(zbins[zbin+1])+")\n"+
                         "D1 = "+str(row['D1'])+"  num = "+str(d1num)+"\n"+
                         "R1 = "+str(row['R1'])+"  num = "+str(r1num)+"\n"+
                         "D2 = "+str(row['D2'])+"  num = "+str(d2num)+"\n"+
                         "R2 = "+str(row['R2'])+"  num = "+str(r2num)+"\n")
                blurbs.append(blurb)
                
                #get the signal, for posterity
                
                
                if ERRORBARS:
                    
                    #
                    # Get 10 groups of equal-ish area regions [in index forms]
                    #
                    parts, areas = r1r2cm.partition(num=10)
        
                    #
                    # Get every combination thereof 
                    #
                    parts_tupled = [tuple(parts[i]) for i in range(len(parts))]
                    perms = set()
                    for perm in combinations(parts_tupled, 1):
                        perms.add(perm)
                    parts = []
                    for perm in perms:
                        part_to_add = []
                        for tup in perm:
                            for index in tup:
                                part_to_add.append(index)
                        parts.append(part_to_add)
                    
                    #
                    # Figure out the area associated with each combination
                    #
                    r1_areas, r2_areas = r1r2cm.pix_area()
                    
                    areas = np.zeros(len(parts))
                    for i in range(len(parts)):
                        for j in range(len(parts[i])):
                            areas[i] = areas[i] + r1_areas[parts[i][j]]
                    areas = np.sum(r1_areas)-areas
                    
                    areas = np.ones(len(parts), dtype=float)
                    
                    d1d2_s = [d1d2cm.get_signal(cut=part) for part in parts]
                    d1r2_s = [d1r2cm.get_signal(cut=part) for part in parts]
                    d2r1_s = [d2r1cm.get_signal(cut=part) for part in parts]
                    r1r2_s = [r1r2cm.get_signal(cut=part) for part in parts]
                    
                    signal = [((d1d2_s[i]-d1r2_s[i]-d2r1_s[i]+r1r2_s[i])/r1r2_s[i]) for i in range(len(d1d2_s))]
                    
                    signal_mean = np.average(signal, axis=0, weights=areas)
                    signal_std = np.sqrt(np.average((signal-signal_mean)**2, axis=0, weights=areas))
                    
                    print(signal_mean)
                    print(signal_std)
                    
                    signals.append(signal_mean)
                    errors.append(signal_std)
                    
                    gc.collect()
                    
                else:
                    d1d2 = d1d2cm.get_signal()
                    d1r2 = d1r2cm.get_signal()
                    d2r1 = d2r1cm.get_signal()
                    r1r2 = r1r2cm.get_signal()
                    
                    signals.append((d1d2 - d1r2 - d2r1 + r1r2)/r1r2)
                    errors.append(np.zeros(len(d1d2)))
                    
                    gc.collect()
                    
                #blurbs.append("")
                names.append(name)
                zbins_used.append(zbin)
                cnums.append(corr_number)
                
                if mods:
                    dm, dmr   = d1d2cm.get_mod(min_z = zbins[zbin],
                                               max_z = zbins[zbin+1])
                    dms.append(dm)
                    dmrs.append(dmr)
        
        
                #Close the count matrix objects and clear them
                d1d2cm, d1r2cm, d2r1cm, r1r2cm = 0, 0, 0, 0
                gc.collect()
                
                
        d1fs.append(d1f)
        d2fs.append(d2f)
    
        #Report runtime and number of files read (should be == number of jobs)
        print("Reading corr " + str(corr_number) + " finished at time = " + 
              str(current_time(start_time)) + " with " +  str(files_read) +
              " files read in total. \n\n")
                
    return (cnums, zbins_used, names, signals, errors, abins,
            zbins, d1fs, d2fs, dms, dmrs, blurbs)



def main(job_suf="", mods=False, save=True, load=False, email=True, log=True):
    directory = os.path.dirname(os.path.realpath(__file__))
    parent = os.path.dirname(directory)
    savename = parent + "/cats/mpi_read"+job_suf+".pickle"
    out = tuple()
    
    #
    # Either generate new results or get old ones
    #
    if save or log:
        out = read_jobs(job_suf=job_suf, mods=mods)
        f = open(savename, 'wb')
        pickle.dump(out, f)
    elif load:
        f = open(savename, 'rb')
        out = pickle.load(f)
    else:
        out = read_jobs(job_suf = job_suf, mods=mods)
    
    #
    # Assign all variables
    #
    cnums, zbins_used, names, signals, errors, abins, zbins, d1fs, d2fs, modv, modr, blurbs = out
    
    #
    # Get the cosmology information set up
    #
    params, results = 0,0
    params = camb.CAMBparams()
    params.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    params.InitPower.set_params(ns=0.965)
    results = camb.get_results(params)
    results.calc_power_spectra()
    
    #
    # Figure out cosmological distance scales
    #
    zbins = np.array(zbins, dtype=float)
    a_bin_middles = (abins[1:] + abins[:-1]) / 2
    z_bin_middles = (zbins[1:] + zbins[:-1]) / 2
    xlabels = []
    xvalues = []
    for z_mid in z_bin_middles:
        xlabels.append(("(Mpc) scale at z = " + str(z_mid)))
        comoving_dist = results.comoving_radial_distance(z_mid)
        xvalues.append(comoving_dist*np.tan(a_bin_middles*np.pi/180.0)) 
        
        
    #
    # Prepare to fit a power law to the data
    #
    def func_powerlaw(x, a, k):
        return a * ( x**k )
    
    powerlaws = []
    a = []
    k = []
    xrange = np.linspace(abins[0], abins[-1], 100)
    for i, sig in enumerate(signals):
        sig_to_use = [sig[j] for j in np.where(sig > 0)[0]]
        amids_to_use = [a_bin_middles[j] for j in np.where(sig > 0)[0]]
        errs_to_use = [errors[i][j] for j in np.where(sig > 0)[0]]
        
        for j in range(len(errs_to_use)):
            if errs_to_use[j] == 0:
                errs_to_use[j] = 1.0
        
        popt, pcov = curve_fit(func_powerlaw, amids_to_use, sig_to_use,
                               maxfev=2000, sigma=errs_to_use)
        
        a.append(popt[0])
        k.append(popt[1])
        powerlaws.append(func_powerlaw(xrange, *popt))
    
    #
    # Adjust the errors to play nice on a log space graph
    #
    # That is, if the error is greater than the signal (and would lead to a log
    # plot trying to plot something negative, adjust it so it just gets close
    # to zero
    #
    print(signals)
    print(errors)
    
    errs_corrected = []
    for i in range(len(signals)):
        lowerrs = []
        for j in range(len(signals[i])): 
            if errors[i][j] > signals[i][j]:
                lowerrs.append(signals[i][j]*(1.0-0.0001))
            else:
                lowerrs.append(errors[i][j])
        errs_corrected.append([lowerrs, errors[i]])
    errors = errs_corrected
    
    #
    # Save a file with the output information to be sent to me in an email
    #
    outfile_txt = directory + "/output.txt"
    outfile_pdf = directory + "/output.pdf"
    
    f = open(outfile_txt, 'w')
    
    f.write("mpi_read output")
    f.write("\nzbins:")
    f.write(str(zbins))
    f.write("\nabins")
    f.write(str(abins))
    f.write("\nxlabels")
    f.write(str(xlabels))
    f.write("\nxvalues")
    f.write(str(xvalues))
    f.write("\n\nData follows for each correlation in order:\n")
    f.write("cnum, zbin, name, signal, errors, blurb, model x and y if applicable\n")
    for i in range(len(cnums)):
        f.write(str(cnums[i])+"\n")
        f.write(str(names[i])+"\n")
        f.write(str(signals[i])+"\n")
        f.write(str(errors[i])+"\n")
        f.write(str(blurbs[i])+"\n")
        if mods:
            if len(modv[i]) > 0:
                f.write(str(modr[i])+"\n")
                f.write(str(modv[i])+"\n")
        f.write("\n\n\n\n")
    f.close()
    

    #
    # Develop a color scheme for plotting, with one unique color per correlation
    #
    color_options = ['r', 'g', 'c', 'b', 'y', 'm', 'k']
    colors = {}
    for name in names:
        if name in colors:
            pass
        else:
            colors[name] = color_options.pop()
            
    print(colors)
            
    #
    # Plot the results
    #    
    fig = plt.figure(figsize=(15,10,))
    ax0 = None
    
    min_y = 1.0
    max_y = 0.0001
    
    
    for i in range(0, len(zbins)-1):
        ax = None
        if i == 0:
            ax0 = fig.add_subplot(int("1"+str(len(zbins)-1)+ str(i+1)))
            ax = ax0
        else:
            ax = fig.add_subplot(int("1"+str(len(zbins)-1)+ str(i+1)),
                            sharex=ax0, sharey=ax0)
        plt.title(str("z bin " + str(zbins[i]) + ", " + str(zbins[i+1])), y=1.08)
        this_bin = np.where(np.array(zbins_used, dtype=int) == i)[0]
        
        if i == 0:
            for j in this_bin:
                if mods:
                    if len(modv[j]) > 0:
                        plt.plot(modr[j], modv[j], label=names[j]+" CAMB MODEL",
                                 color=colors[names[j]])
                plt.errorbar(xvalues[i], signals[j], yerr=errors[j], label=names[j],
                            c=colors[names[j]], fmt='o')
                
                print(signals[j])
                print(errors[j])
                
                plt.plot(comoving_dist*np.tan(xrange*np.pi/180.0), powerlaws[j], c=colors[names[j]],
                         label=(str(a[j]) + r"x^" + str(k[j])))
                
                if max(signals[j]) > max_y:
                    max_y = max(signals[j])
                    
                sig_pos = [signals[j][i] for i in np.where(signals[j] > 0)[0]]
                if min(sig_pos) < min_y:
                    min_y = min(sig_pos)
                    
                print(blurbs[j])
        else:
            for j in this_bin:
                if mods:
                    if len(modv[j]) > 0:
                        plt.plot(modr[j], modv[j], color=colors[names[j]])
                        
                plt.errorbar(xvalues[i], signals[j], yerr=errors[j], c=colors[names[j]], fmt='o')
                
                print(signals[j])
                print(errors[j])
                
                plt.plot(comoving_dist*np.tan(xrange*np.pi/180.0), powerlaws[j], c=colors[names[j]],
                         label=(str(a[j]) + r"x^" + str(k[j])))
                
                print(blurbs[j])
                
                if max(signals[j]) > max_y:
                    max_y = max(signals[j])
                sig_pos = [signals[j][i] for i in np.where(signals[j] > 0)[0]]
                if min(sig_pos) < min_y:
                    min_y = min(sig_pos)
        
        theta, model = camb_model.mod_test(zmin=zbins[i], zmax=zbins[i+1], imax=24.0)
        plt.plot(comoving_dist*np.tan(np.array(theta,dtype=float)*np.pi/180.0),
                 model, color='black', label='Autocorr model')
        
                  
        plt.legend()
        plt.xlabel(xlabels[i])         
        
                    
        
        if i == 0:
            plt.ylabel("Angular correlation (omega)")
        
        factor = 3.0
        plt.axis([min(xvalues[0])/factor, factor*max(xvalues[0]),
                  min_y/factor,           factor*max_y          ])
        plt.loglog()
        
        ax2 = ax.twiny()
        ax2.set_xscale('log')
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(xvalues[i])
        ax2.set_xticklabels(np.around(abins*60,decimals=2))
        ax2.set_xlabel("Angle (arcminutes)")
        
        
        plt.tight_layout()
    
    #
    # Save a picture
    #
    fig.savefig(outfile_pdf)
    
    
    #
    # Send me an email with results
    #
    files = [outfile_txt, outfile_pdf, savename, (directory+"/qualifier.ini")]
    if email:
        email_bot.email_me("mpi_read output", subject="mpi_read_output",
                           attachments=files)
    
    #
    # Save the results to a log directory
    #    
    if log:
        logfiles = glob.glob(parent+"/log/*/")
        nums = [0]
        for logfile in logfiles: nums.append(int(logfile.split("/")[-2]))
        file_manager.ensure_dir(parent+"/log/"+str(max(nums)+1)+"/")
        for file in files:
            name = file.split('/')[-1]
            shutil.copyfile(file, (parent+"/log/"+str(max(nums)+1)+"/"+name))
            
    #
    # Show the figure
    #
    plt.show()  

def log_read(n=-1, suf="", mods=False):
    """
    Read a correlation result from the log files. 
    
    parameter n is either the log file number or the nth log file 
    (i.e. enter -1 for the latest log file, etc)
    """
    directory = os.path.dirname(os.path.realpath(__file__))
    parent = os.path.dirname(directory)
    logdirs = glob.glob(parent+"/log/*/")
    
    if len(logdirs) == 0:
        raise FileNotFoundError("Log file "+str(n)+" not found.")
    
    logdirs_nums = {}
    for logdir in logdirs:
        num = int(logdir.split("/")[-2])
        logdirs_nums.update({num:logdir})
        
    dir_to_use = ""
    if n in logdirs_nums:
        dir_to_use = logdirs_nums[n]
    else:
        nums_sorted = list(logdirs_nums)
        nums_sorted.sort()
        dir_to_use = logdirs_nums[nums_sorted[n]]
    
    pickles = glob.glob(dir_to_use + "*.pickle")
    source = pickles[0]
    
    destination = parent+"/cats/mpi_read"+suf+".pickle"
    shutil.copyfile(source, destination)
    
    main(job_suf=suf, mods=mods, save=False, load=True, email=False, log=False)
    
    
if __name__ == "__main__":
    main(mods=False)
    #log_read(-1)
    #log_read(-2)
