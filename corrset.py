#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 11:23:13 2017

Where the magic happens

@author: Cassandra Henderson


development goals:
    1. Autocorrelation option
    2. More options for qualifier
    3. Split AGN catalog into types 1 and 2
    4. Save RR result between corrsets
    5. Port over to Tiger
    6. Default method to save result.
"""

import gc
import io
import os
import sys
import time
import psutil
import qualifier
import subprocess
import numpy                   as np
import pandas                  as pd
import healpy                  as hp
import multiprocessing         as mp
import matplotlib.pyplot       as plt
import _pickle                 as pickle
from   corruscant          import twopoint
from   corruscant          import clustering
from   sklearn.neighbors   import KDTree

"""
File options
"""
dir_path  = "/scr/depot0/csh4/"
code_path = '/scr/depot0/csh4/py/codes/ver1/corrset/'
logfile =  dir_path + "logs/corrset.txt"
filepref = dir_path + "cats/corrs/corr_"
p_rand_q = dir_path + "cats/processed/rand_q.hdf5"
p_agn_q =  dir_path + "cats/processed/agn_q.hdf5"
p_hsc_q =  dir_path + "cats/processed/hsc_q.hdf5"
figbin  =  dir_path + "figures/name.pdf"
mpi_path=  dir_path + "cats/mpi/holder/h"

"""
Bin opetions
"""
nside_default = 8
angular_bins  = np.logspace(np.log10(.0025), np.log10(1.5), num=12)
redshift_bins = np.linspace(0.3, 1.5, num=4)

"""
Program options
"""
CROSS_CORRELATE_AGAIN = True
QUALIFY_AGAIN = False
rep_to_file = False

"""
Log Functions
"""
if rep_to_file:
    logfile = open(logfile, "a")
start_time = int(time.time())
#Defining a couple in-method helper functions
def current_time():
    return (int(time.time()) - start_time)
def report(report_string):
    global rep_to_file
    if rep_to_file:
        sys.stdout.flush()
        time = current_time()
        logfile.write(""+ "\n")
        logfile.write("--- corrset reporting ---"+ "\n")
        logfile.write(report_string+ "\n")
        logfile.write("Time is " + str(time) + " seconds from start. " + "\n")
        logfile.write("Memory use is " + str(psutil.virtual_memory().percent) + "%"+ "\n")
        logfile.write(""+ "\n")
        sys.stdout.flush()
        gc.collect()
    else:
        sys.stdout.flush()
        time = current_time()
        print("")
        print("--- corrset reporting ---")
        print(report_string)
        print("Time is " + str(time) + " seconds from start. ", end="")
        print("Memory use is " + str(psutil.virtual_memory().percent) + "%")
        print("")
        sys.stdout.flush()
        gc.collect()

        
"""
Other global things
"""
n_cores = mp.cpu_count()
current_jobs = 0
total_jobs   = 0

#Making a global list of pair-pair combos to run correlations on
pairs = []
n_pix = hp.nside2npix(nside_default)
#First do the self-self combos
for i in range(n_pix):
    pairs.append([i, i])
#Now do the self-other combos
for i in range(n_pix):
    to_add = hp.pixelfunc.get_all_neighbours(nside_default, i)
    for j in range(len(to_add)):
        pair_to_add = []
        if to_add[j] > i:
            pair_to_add = [i, to_add[j]]
        else:
            pair_to_add = [to_add[j], i]
        if pair_to_add in pairs:
            pass
        else:
            pairs.append(pair_to_add)
    


def main():
    """
    Runs a test cross-correlation on a small data sample. 
    """
    if QUALIFY_AGAIN:
        qualifier.main(sparsify=0.33, smallbox=True)

    report("First report: datetime = " + time.asctime(time.localtime(time.time())))
    
    report("Starting correlation process")
    
    corrset = Corrset(filepref=filepref, 
                      d1=p_agn_q, d2=p_hsc_q, r=p_rand_q, 
                      zbins=redshift_bins, abins = angular_bins,
                      d1names = ['ra', 'dec', 'pix', 'redshift'],
                      d2names = ['ra', 'dec', 'pix', 'redshift'],
                      rnames  = ['ra', 'dec', 'pix'],
                      crosscorr_now = CROSS_CORRELATE_AGAIN)
    
    corrset.prep_crosscorrelation(save=False)
    
    report("Finished generating/loading correlation. Now reading results")
    
    corrs, errs = corrset.read_crosscorrelation()
    a_bin_middles = (angular_bins[1:] + angular_bins[:-1]) / 2
    
    report("Finished reading results. Plotting")
    
    fig = plt.figure()
    for i in range(0, len(redshift_bins)-1):
        report("z bin "+str(redshift_bins[i])+", "+str(redshift_bins[i+1])+"\n"+
               "corrs = "+str(corrs[i])+" \nerrs = "+str(errs[i]))
        fig.add_subplot(int("13"+ str(i+1)))
        plt.title(str("z bin " + str(redshift_bins[i]) + ", " + str(redshift_bins[i+1])))
        plt.errorbar(a_bin_middles, corrs[i], yerr=errs[i])
    fig.savefig(figbin)
    plt.show()
    
    report("Done")


class Corrset:
    """
    Handler for the hdf which contains the correlation set. 
    
    This HDF5 file will contain (n*m) + 3 HDF5 tables. Two for meta info, and
    the rest for the counted pairs
        "meta_info"     - A table with meta info (name, files used, nside, etc)
        "z_bin_info"    - A table with the list of redshift bins
        "a_bin_info"    - A table with the shape of each angular bin
        "bin_00"        - The zeroth z bin, zeroth angular bin, 
        "bin_01"        - The zeroth z bin, 1st angular bin,
        ...             - Etc.
    
    This handler will be able to:
        -construct the hdf with ra, dec, and z data from two catalogs, 
         !quickly! and !parallelized! (hopefully)
        -read the hdf for statistical information 
        -Compute a cross correlation as well as an autocorrelation
    """
    def __init__(self, **kwargs): 
        """
        Reads a corrset from a file, or writes a new corrset to a file. 
        Gets/Initializes the info tables. 
        Gets/Initializes the metadata.
        
        @params
            filepref     -- prefix of the files in which to save the cross 
                            correlation hdfs
            
            d1, d2, r    -- file locations of hdfs containing d1, d2, and r for
                            the cross correlation
            
            d1names,     -- column names in d1, d2, and r for ra, dec, pixel and
            d2names,        redshift
            rnames
            
            zbins, abins -- the redshift and angular bins to use in the analysis
            
            corr_now     -- If True, run the correlation right now and save.
            
        """
        self.filepref = kwargs['filepref']
        self.d1       = kwargs['d1']
        self.d2       = kwargs['d2']
        self.r        = kwargs['r']
        self.zbins    = kwargs['zbins']
        self.abins    = kwargs['abins']
        self.d1_names = kwargs['d1names']
        self.d2_names = kwargs['d2names']
        self.r_names  = kwargs['rnames']
        
        self.matrices = {}
        
        if kwargs['crosscorr_now']:
            self.prep_crosscorrelation(True)
        
    def corruscant(self):
        """
        CROSS-CORRELATION
        
        Implementation of corrscant courtesy of A. Pellegrino
        
        May or may not be significantly faster. Probably is.
        
        Returns the correlation. Errors are probably contained therein.
        """
        #Step 0: Retrieve Data
        report("corruscant(): Preparing inputs.")
        ra1, dec1, pix1, z1 = get_cols(self.d1, self.d1_names[:4])
        ra2, dec2, pix2, z2 = get_cols(self.d2, self.d2_names[:4])
        ra3, dec3, pix3 = get_cols(self.r, self.r_names[:3])
            
        #Step 1: Divide into redshift bins.
        ra1z, ra2z, dec1z, dec2z = [], [], [], []
        for z in range(0, (len(self.zbins)-1)):
            in_zbin1 = np.where(np.logical_and(np.greater(z1, self.zbins[z]),
                                                np.less(z1, self.zbins[z+1])))[0]
            
            in_zbin2 = np.where(np.logical_and(np.greater(z2, self.zbins[z]),
                                                np.less(z2, self.zbins[z+1])))[0]
            ra1z.append([ra1[i] for i in in_zbin1])
            dec1z.append([dec1[i] for i in in_zbin1])
            ra2z.append([ra2[i] for i in in_zbin2])
            dec2z.append([dec2[i] for i in in_zbin2])
        
        #Step 2: Prepare to pass to Corruscant by making trees
        report("corruscant(): growing trees.")
        dx,dy,dz=sph2cart(1.0, ra3, dec3, degree=True)
        dataxyz=np.array([dx,dy,dz]).transpose()
        rand_tree = clustering.tree(dataxyz)
        
        gal_trees = []
        aa_trees = []
        for z in range(len(self.zbins)-1):
            dx,dy,dz=sph2cart(1.0, ra2z[z], dec2z[z], degree=True)
            dataxyz=np.array([dx,dy,dz]).transpose()
            gal_trees.append(clustering.tree(dataxyz))
            
            dx,dy,dz=sph2cart(1.0, ra1z[z], dec1z[z], degree=True)
            dataxyz=np.array([dx,dy,dz]).transpose()
            aa_trees.append(clustering.tree(dataxyz))
        
        #Step 3: Pass to Corruscant, redshift bin by redshift bin...
        results = []
        for z in range(len(self.zbins)-1):
            report("corruscant(): Running correlation for z bin " + str(z+1) +
                   " of " + str(len(self.zbins)-1) + ".")
            results.append(twopoint.angular.crosscorr(aa_trees[z], gal_trees[z], 
                                                      rand_tree, rand_tree,
                                                      self.abins, num_threads=4, 
                                                      est_type="landy-szalay"))
        return results
    
    def read_crosscorrelation(self, jackknife_pix=range(0, n_pix)):
        """
        CROSS-CORRELATION
        
        Finds the pair count term associated with a given z_bin.
        Returns the total number of objects, the numpy arrays of counts and the
        bins. 
        
        @params
            jackknife_pix - int, array of ints, or array of array of ints
                            The pixels to remove from the analysis before
                            calculating the correlation signal. 
                            
                            Removes each pixel or list of pixels, one at a time,
                            and returns the mean and standard deviation of the
                            result.
                            
                            At the moment, we also add an empty list to whatever
                            the computation is, so there is also 1 instance of 
                            the entire data set in the resulting PDFs
        
        @returns
            results, errors - list of float numpy arrays with results, both in
                              shape = (len(z_bins)-1, len(a_bins)-1)
                              
        """
        #Pull out the dicts which contain the CountMatrix objects for each pair
        d1d2 = self.matrices['d1d2']
        d1r  = self.matrices['d1r']
        d2r  = self.matrices['d2r']
        rr   = self.matrices['rr']
        
        
        #Prepare the list to save the lists of matrices in
        by_zbin = []
        for i in range(len(self.zbins)-1):
            by_zbin.append([[], [], [], []])
        
        #Go through each dict and pull out the relevant bins per list.
        #What we want: shape of by_zbin = (len(zbins), 4, len(abins))
        matrix_dicts = [d1d2, d1r, d2r]
        for j in range(len(matrix_dicts)):
            for i in range(len(self.zbins)-1):
                a_bin = []
                for k in range(len(self.abins)-1):
                    matrix = matrix_dicts[j]['bin_' + str(i) + str(k)].mat
                    a_bin.append(matrix)
                by_zbin[i][j] = a_bin
        for k in range(len(self.abins)-1):
            matrix = rr['bin_' + str(0) + str(k)].mat
            for i in range(len(self.zbins)-1): 
                by_zbin[i][3].append(matrix)
                
        #Now that we have the matrices, it's time to jackknife for each bin.
        results_by_bin = []
        errors_by_bin = []
        for i in range(len(self.zbins)-1):
            results, errors = self.jackknife_crosscorr(by_zbin[i], jackknife_pix)
            results_by_bin.append(results)
            errors_by_bin.append(errors)
        
        return results_by_bin, errors_by_bin
    
    def read_autocorrelation(self, d='d1', jackknife_pix=range(0, n_pix)):
        """
        AUTOCORRELATION
        
        Nearly identical to the cross correlation, except with a couple terms
        renamed and some logic switched around.
        
        Finds the pair count term associated with a given z_bin.
        Returns the total number of objects, the numpy arrays of counts and the
        bins. 
        
        @params
            d             - 'd1' or 'd2' - the data set to run the autocorr on
            jackknife_pix - int, array of ints, or array of array of ints
                            The pixels to remove from the analysis before
                            calculating the correlation signal. 
                            
                            Removes each pixel or list of pixels, one at a time,
                            and returns the mean and standard deviation of the
                            result.
                            
                            At the moment, we also add an empty list to whatever
                            the computation is, so there is also 1 instance of 
                            the entire data set in the resulting PDFs
        
        @returns
            results, errors - list of float numpy arrays with results, both in
                              shape = (len(z_bins)-1, len(a_bins)-1)
                              
        """
        #Pull out the dicts which contain the CountMatrix objects for each pair
        dd = self.matrices[(d+d)]
        dr  = self.matrices[(d+'r')]
        rr   = self.matrices['rr']
        
        
        #Prepare the list to save the lists of matrices in
        by_zbin = []
        for i in range(len(self.zbins)-1):
            by_zbin.append([[], [], []])
        
        #Go through each dict and pull out the relevant bins per list.
        #What we want: shape of by_zbin = (len(zbins), 4, len(abins))
        matrix_dicts = [dd, dr]
        for j in range(len(matrix_dicts)):
            for i in range(len(self.zbins)-1):
                a_bin = []
                for k in range(len(self.abins)-1):
                    matrix = matrix_dicts[j]['bin_' + str(i) + str(k)].mat
                    a_bin.append(matrix)
                by_zbin[i][j] = a_bin
        for k in range(len(self.abins)-1):
            matrix = rr['bin_' + str(0) + str(k)].mat
            for i in range(len(self.zbins)-1): 
                by_zbin[i][3].append(matrix)
                
        #Now that we have the matrices, it's time to jackknife for each bin.
        results_by_bin = []
        errors_by_bin = []
        for i in range(len(self.zbins)-1):
            results, errors = self.jackknife_autocorr(by_zbin[i], jackknife_pix)
            results_by_bin.append(results)
            errors_by_bin.append(errors)
        
        return results_by_bin, errors_by_bin
    
    
    def jackknife_crosscorr(self, matrix_group_list, jackknife_pix):
        """
        CROSS-CORRELATION
        
        Take the matrices and the jackknife pix and compute the crosscorrelations. 
    
        Corresponds to all the data in 1 z bin.         
        
        @params:
            matrix_group_list
                list of lists of matrices in 
                shape = (4, len(a_bins)-1, n_pix, n_pix)
                which is to have the jackknife_pix removed before having its
                crosscorrelation signal calculated
            jackknife_pix
                as in read_correlation(). 
                int or list of ints or list of list of ints.
        
        @returns:
            mean and standard deviation of the crosscorrelation signal from these
            matrices and jackknifing parameters.
        """
        if type(jackknife_pix) is int:
            jackknife_pix = [jackknife_pix]
        elif type(jackknife_pix) is range:
            jackknife_pix = list(jackknife_pix)
        jackknife_pix.append([])
        
        correlation_results = []
        for pix in jackknife_pix:
            jackknifed_group = [[],[],[],[]] 
            #Here we require there be data in the pixels before jackknifing them. 
            if (self.check_rowcol_sum(matrix_group_list[0][0], pix) and
                self.check_rowcol_sum(matrix_group_list[3][0], pix)):
                for d in range(len(jackknifed_group)):
                    for a in range(len(self.abins)-1):
                        mat_to_use = np.matrix(matrix_group_list[d][a])
                        jackknifed = self.drop_rowcol(mat_to_use, pix)
                        jackknifed_group[d].append(jackknifed)
                result = self.crosscorrelate(jackknifed_group)
                if len(np.where(np.isnan(result))[0]) > 0 or len(result) == 0:
                    pass
                else:
                    correlation_results.append(result)
        mean = np.mean(np.array(correlation_results), axis=0)
        stdev = np.std(np.array(correlation_results), axis=0)
        return mean, stdev
    
    
    def jackknife_autocorr(self, matrix_group_list, jackknife_pix):
        """
        AUTOCORRELATION
        
        Take the matrices and the jackknife pix and compute the crosscorrelations. 
    
        Corresponds to all the data in 1 z bin.         
        
        @params:
            matrix_group_list
                list of lists of matrices in 
                shape = (3, len(a_bins)-1, n_pix, n_pix)
                which is to have the jackknife_pix removed before having its
                autocorrelation signal calculated
            jackknife_pix
                as in read_correlation(). 
                int or list of ints or list of list of ints.
        
        @returns:
            mean and standard deviation of the autocorrelation signal from these
            matrices and jackknifing parameters.
        """
        if type(jackknife_pix) is int:
            jackknife_pix = [jackknife_pix]
        elif type(jackknife_pix) is range:
            jackknife_pix = list(jackknife_pix)
        jackknife_pix.append([])
        
        correlation_results = []
        for pix in jackknife_pix:
            jackknifed_group = [[],[],[]] 
            #Here we require there be data in the pixels before jackknifing them. 
            if (self.check_rowcol_sum(matrix_group_list[0][0], pix) and
                self.check_rowcol_sum(matrix_group_list[2][0], pix)):
                for d in range(len(jackknifed_group)):
                    for a in range(len(self.abins)-1):
                        mat_to_use = np.matrix(matrix_group_list[d][a])
                        jackknifed = self.drop_rowcol(mat_to_use, pix)
                        jackknifed_group[d].append(jackknifed)
                result = self.autocorrelate(jackknifed_group)
                if len(np.where(np.isnan(result))[0]) > 0 or len(result) == 0:
                    pass
                else:
                    correlation_results.append(result)
        mean = np.mean(np.array(correlation_results), axis=0)
        stdev = np.std(np.array(correlation_results), axis=0)
        return mean, stdev

    
    def crosscorrelate(self, matrix_list):
        """
        CROSS-CORRELATION
        
        Computes the landy-szalay cross-correlation from this matrix (2d) list
        
        @params
            matrix_list shape = (4, len(abins), n_pix, n_pix)
        @returns
            The landy-szalay crosscorrelation signal associated with this matrix.
        """
        sumlist = [[],[],[],[]]
        for i in range(len(matrix_list)):
            for j in range(len(matrix_list[i])):
                sumlist[i].append(matrix_list[i][j].sum())
        for i in range(len(matrix_list)):
            sumlist[i] = np.array(sumlist[i], dtype=float)#/n_pix
        
        if len(np.where(np.equal(sumlist[3], 0.0))[0]) > 0:
            to_ret = np.zeros(np.shape(sumlist))
            to_ret.fill(np.nan)
            return to_ret
        else: 
            return ((sumlist[0] - sumlist[1] - sumlist[2] + sumlist[3])/sumlist[3])
    
    def autocorrelate(self, matrix_list):
        """
        AUTO-CORRELATION
        
        Computes the landy-szalay auto-correlation from this matrix (2d) list
        
        @params
            matrix_list shape = (3, len(abins), n_pix, n_pix)
                                 3: d1d1, d1r, rr
        @returns
            The landy-szalay autocorrelation signal associated with this matrix.
        """
        sumlist = [[],[],[]]
        for i in range(len(matrix_list)):
            for j in range(len(matrix_list[i])):
                sumlist[i].append(matrix_list[i][j].sum())
        for i in range(len(matrix_list)):
            sumlist[i] = np.array(sumlist[i], dtype=float)#/n_pix
        
        if len(np.where(np.equal(sumlist[2], 0.0))[0]) > 0:
            to_ret = np.zeros(np.shape(sumlist))
            to_ret.fill(np.nan)
            return to_ret
        else: 
            return ((sumlist[0] - 2*sumlist[1] + sumlist[2])/sumlist[2])
        pass
    
    
    def prep_crosscorrelation(self, save, special_r = False, overwrite=False, 
                              special_r_cat = '/scr/depot0/csh4/cats/corrs/corr_rr'):
        """
        CROSS-CORRELATION
        
        Either generates or loads up the cross correlation, in prep for 
        read_crosscorrelation. If a matrix is already
        
        @params
            save - whether or not to generate new correlations (True) or load
                   an already calculated one.
                   
            overwrite - if there are already matrices under the correct names
                        prep_crosscorrelation will only re-compute correlations
                        if overwrite == True
            
            special_r     - Set True if counting RR if RR has been computed before
            special_r_cat - The name of the file where the old RR is stored
                            Using special_r forces save to be False
        
        @results
            Gets all of the cross-correlation matrices into self.matrices
        """
        report("prep_crosscorrelation(save = " + str(save) + "): Starting...")
        #FORK BY DATA PAIR
        if 'd1d2' in self.matrices and overwrite == False:
            pass
        else:
            report("d1d2")
            #D1D2 
            self.pair_count(save, self.d1, self.d2, False, False,
                            'd1d2', self.d1_names, self.d2_names)
        
        if 'd1r' in self.matrices and overwrite == False:
            pass
        else:        
            report("d1r")
            #D1R
            self.pair_count(save, self.d1,  self.r, False,  True,
                            'd1r',  self.d1_names,  self.r_names)
        
        if 'd2r' in self.matrices and overwrite == False:
            pass
        else:       
            report("d2r")
            #D2R
            self.pair_count(save, self.d2,  self.r, False,  True,
                            'd2r',  self.d2_names,  self.r_names)
        
        if 'rr' in self.matrices and overwrite == False:
            pass
        else:        
            report("rr")
            #RR
            self.pair_count(save,  self.r,  self.r,  True,  True,
                            'rr',   self.r_names,   self.r_names,
                            special_r = special_r, 
                            special_r_cat = special_r_cat)
    
        report("prep_crosscorrelation(save = " + str(save) + "):done")
    
    def prep_autocorrelation(self, save, d="d1", special_r = False, overwrite=False,
                             special_r_cat = '/scr/depot0/csh4/cats/corrs/corr_rr'):    
        """
        AUTOCORRELATION
        
        As prep_crosscorrelation, but for an autocorrelation 
        
        @params
            save - whether or not to generate new correlations (True) or load
                   an already calculated one.
            
            d    - the data set to run the autocorrelation on. Can be 'd1' or 'd2'
                   
            overwrite - if there are already matrices under the correct names
                        prep_crosscorrelation will only re-compute correlations
                        if overwrite == True
            
            special_r     - Set True if counting RR if RR has been computed before
            special_r_cat - The name of the file where the old RR is stored
                            Using special_r forces save to be False
        
        @results
            Gets all of the cross-correlation matrices into self.matrices
        """
        report("prep_autocorrelation(save = " + str(save) + "): Starting...")
        
        
        if d == "d1":
            if 'd1d1' in self.matrices and overwrite == False:
                pass
            else:
                report("d1d1")
                #D1D2 
                self.pair_count(save, self.d1, self.d1, False, False,
                                'd1d1', self.d1_names, self.d1_names)
            
            if 'd1r' in self.matrices and overwrite == False:
                pass
            else:
                report("d1r")
                #D1R
                self.pair_count(save, self.d1,  self.r, False,  True,
                                'd1r',  self.d1_names,  self.r_names)
            
            if 'rr' in self.matrices and overwrite == False:
                pass
            else:
                report("rr")
                #RR
                self.pair_count(save,  self.r,  self.r,  True,  True,
                                'rr',   self.r_names,   self.r_names,
                                special_r = special_r, 
                                special_r_cat = special_r_cat)
        elif d == "d2":
            if 'd2d2' in self.matrices and overwrite == False:
                pass
            else:
                report("d2d2")
                #D1D2 
                self.pair_count(save, self.d2, self.d2, False, False,
                                'd2d2', self.d2_names, self.d2_names)
            
            if 'd2r' in self.matrices and overwrite == False:
                pass
            else:
                report("d2r")
                #D1R
                self.pair_count(save, self.d2,  self.r, False,  True,
                                'd2r',  self.d2_names,  self.r_names)
            
            if 'rr' in self.matrices and overwrite == False:
                pass
            else:
                report("rr")
                #RR
                self.pair_count(save,  self.r,  self.r,  True,  True,
                                'rr',   self.r_names,   self.r_names,
                                special_r = special_r, 
                                special_r_cat = special_r_cat)
        else: 
            raise ValueError((str(d) + " is not 'd1' or 'd2'"))
    
        report("prep_autocrosscorrelation(save = " + str(save) + "):done")
        
    
    def pair_count(self, save, d1, d2, no_z1, no_z2, name, colnames1, colnames2,
                   special_r = False, 
                   special_r_cat = '/scr/depot0/csh4/cats/corrs/corr_rr'):
        """
        Does the pair counting in a smart parallelized way
        
        @params
            save          - Whether or not to save a new correlation (True) or
                            load an old one (False)
            d1, 2         - The file paths to datasets 1 and 2
            no_z1, 2      - Whether or not to use z binning for datasets 1 and 2
            name          - The name to append to self.filepref to get the file
                            path where the correlation will be saved
            colnames1, 2  - Array of the names of the columns for data sets 1 
                            and 2
            
            special_r     - Set True if counting RR if RR has been computed before
            special_r_cat - The name of the file where the old RR is stored
                            Using special_r forces save to be False
        
        @result
            sets self.matrices[name] to a list of corresponding correlation 
            matrices of shape = ((len(z_bins)-1 or 1 (for no_z1 or 2 = True)), 
                                 len(a_bins)-1, n_pix, n_pix)
        """
        if save == True:
            try:
                os.remove(self.filepref+name)
            except FileNotFoundError:
                pass
        
        if special_r:
            mats  = correlate_dat(save=False, load=True, no_z1=no_z1, no_z2=no_z1,
                                  filename=special_r_cat, zbins=self.zbins,
                                  abins=self.abins, colnames1 = colnames1, colnames2=colnames2,
                                  d1=d1, d2=d2, datname=name)
            self.matrices[name] = mats
        
        else:
            mats  = correlate_dat(save=save, load=True, no_z1=no_z1, no_z2=no_z2,
                                  filename=(self.filepref+name), zbins=self.zbins,
                                  abins=self.abins, colnames1 = colnames1, colnames2=colnames2,
                                  d1=d1, d2=d2, datname=name)
            self.matrices[name] = mats

          
    def check_rowcol_sum(self, matrix, index_list):    
        """
        Return True if the row and column at the indices or index given in 
        index_list have anything other than all zeros. False if not.
        
        Basically, this will be true if any data is contained in the given
        pixel numbers. 
        
        @params 
            matrix      - np.matrix to look in
            index_list  - int or list of ints of rows/columns to check for
                          values in
        
        @returns
            True if there are non zero values in the given rows and columnsm
            False if all the values are zero
        """
        if index_list == []:
            return True
        to_check = 0
        if type(index_list) is int:
            index_list = [index_list]
        for index in index_list:
            to_check = to_check+np.sum(matrix[index,:])+np.sum(matrix[:,index])
        if to_check == 0:
            return False #no need to actually subtract this pixel, as no data
        else:            #will be removed
            return True
    
    def drop_rowcol(self, matrix, index_list):
        """
        Remove the rows and columns corresponding to index list from the given
        matrix. Return the clipped matrix.
        
        
        @params 
            matrix      - np.matrix to clip
            index_list  - int or list of ints of rows/columns to clip from matrix
        
        @returns
            the matrix with rows and columns from index list removed
        """
        to_ret = np.copy(matrix)
        if type(index_list) is int:
            index_list = [index_list]
        return np.delete(np.delete(to_ret, index_list, axis=0), index_list, axis=1)
    
    
class CountMatrix:
    """
    Handler for the numpy matrix that holds the pair counting data.
    
    We use a symmetric 2d square matrix of size n*n, where n = the number of
    healpix pixels. The values on the diagonal are for counting the pairs
    within a pixel. Boundary pairs are placed above the diagonal. 
    
    dev note: we may or may not add counts above and below, but add the below
              to above afterwards to maintain the above property. Whatever is
              faster
    
    This class will have the following functionality:
        -save or load this matrix from an hdf. 
            This includes managing its data in the bin_info table
        -balance the matrix so above the diagonal contains all the pair info.
        -read the pair info without a given pixel (jackknife resampling)
        -construct the matrix from input data
    """
    def __init__(self, n=hp.nside2npix(nside_default), filename="", tabname="",
                 load=False, save=False):
        #Initialize the empty matrix
        self.name = tabname
        self.file = filename
        self.n = n
        self.mat = np.zeros((n, n), dtype=float)
        if save:
            self.save()
        if load:
            self.load()
            
    def load(self):
        #Load the hdf matrix from the filename and hdf table tabname
        df = pd.read_hdf(self.file, key=self.name)
        self.mat = df.as_matrix().astype(float)
    
    def save(self):
        #Save the hdf matrix to the filename and hdf table tabname
        df = pd.DataFrame(data=self.mat)
        df.to_hdf(self.file, key=self.name, format="table")
        
    def balance(self):
        #for each value below the diagonal, add to tranverse and set to zero
        self.mat = np.triu(self.mat) + np.tril(self.mat, k=-1).T
    
    def clear(self):
        self.mat = np.zeros((self.n, self.n), dtype=float)

      
def correlate_dat( **kwargs):
    """
    Saves a term
    
    @params
        save = False  - whether or not to make a new term or...
        load = True   - get an existing one
                        Put true for both if you want to save and get the
                        correlation
        
        filename      - file from which to save/load
        
        d1, d2        - source catalogs for the data to correlate
        
        no_z1, no_z2  - default False. Whether or not to use z binning for
                        d1 and d2
                        
        colnames1, 2  - the names of the columns of (in this order)
                        ra, dec, pixel number, redshift. If not using 
                        redshift, then put nothing or anything for redshift
                        in colnames.
                        
        zbins, abins  - redshift and angular bins to use. Default vals are
            abins = np.logspace(np.log10(.0025), np.log10(1.5), num=12)
            zbins = np.linspace(0.3, 1.5, num=4)
        
    @returns
        A list of CountMatrix objects IF load == True
        
    """
    load = kwargs.get('load',False)
    save = kwargs.get('save',True)
    no_z1 = kwargs.get('no_z1', False)
    no_z2 = kwargs.get('no_z2', False)
    zbins = kwargs.get('zbins', np.linspace(0.3, 1.5, num=4))
    abins = kwargs.get('abins', np.logspace(np.log10(.0025), np.log10(1.5), num=12))
    filename = kwargs.get('filename')
    datname  = kwargs.get('datname')
    
        
    if save == True:
        #Get all the parameters to run the correlation on
        d1, d2 = kwargs.get('d1'), kwargs.get('d2')
        
        colnames1, colnames2 = kwargs.get('colnames1'), kwargs.get('colnames2')
        z1, z2 = kwargs.get('z1'), kwargs.get('z2')
        
        #Save the metadata part 1
        pd.DataFrame(zbins).to_hdf(filename, key='zbins')
        pd.DataFrame(abins).to_hdf(filename, key='abins')
        pd.DataFrame(colnames1).to_hdf(filename, key='colnames1')
        pd.DataFrame(colnames2).to_hdf(filename, key='colnames2')
        meta = pd.DataFrame()
        meta['d1'] = [d1]
        meta['d2'] = [d2]
        meta['no_z1'] = [no_z1]
        meta['no_z2'] = [no_z2]
        meta['fn'] = [filename]
        meta['ms'] = [("metastring_time=" +  time.asctime( time.localtime(time.time())))]
        meta.to_hdf(filename, key='meta')
        
        #Get the data out
        ra1, dec1, pix1, z1 = [], [], [], []
        if no_z1:
            ra1, dec1, pix1 = get_cols(d1, colnames1[:3])
        else:
            ra1, dec1, pix1, z1 = get_cols(d1, colnames1[:4])
        
        ra2, dec2, pix2, z2 = [], [], [], []
        if no_z2:
            ra2, dec2, pix2 = get_cols(d2, colnames2[:3])
        else:
            ra2, dec2, pix2, z2 = get_cols(d2, colnames2[:4])
            
        
        
        report("correlate_dat(): Sending to correlate_zbin() with filename "+
               str(filename)+"\n"+"And len(ra1)="+str(len(ra1))+
               " and len(ra2)="+str(len(ra2)))
        
        #Prepare the bins
        global n_cores
        zbins1 = zbins
        zbins2 = zbins
        if no_z1 == True:
            zbins1 = []
        if no_z2 == True:
            zbins2 = []
        cart_bins = angular_dist_to_euclidean_dist(angular_bins)
        
        #Divide up the ra and dec by z bin and pixel
        report("correlate_dat(): Sorting groups by z bin and pixel")
        group1, len1 = by_bin(ra1, dec1, z1, pix1, zbins=zbins1)
        group2, len2 = by_bin(ra2, dec2, z2, pix2, zbins=zbins2)

        #Generate the arguments to the processes
        z = 0
        
        report("correlate_dat(): Run the jobs.")
        #Be cognizant of whether or not redshift is being used in a given analysis
        if no_z1 == True and no_z2 == True:
            correlate_zbin(file = filename,
                           coords1 = group1[0], coords2 = group2[0],
                           z = z, bins = cart_bins,
                           lenra1 = len1[0], lenra2 = len2[0],
                           name = datname)
            z = z + 1
        elif no_z1 == True and no_z2 == False:
            for i in range(0, len(group2)):
                correlate_zbin(file = filename,
                               coords1 = group1[0], coords2 = group2[i],
                               lenra1 = len1[0],     lenra2 = len2[i],
                               z = z, bins = cart_bins, name = datname)
                z = z + 1
        elif no_z1 == False and no_z2 == True:
            for i in range(0, len(group1)):
                correlate_zbin(file = filename,
                               z = z, bins = cart_bins, name = datname,
                               coords1 = group1[i], coords2 = group2[0],
                               lenra1 = len1[i],     lenra2 = len2[0])
                z = z + 1
        else:
            for i in range(0, len(group1)):
                correlate_zbin(file = filename,
                               z = z, bins = cart_bins, name = datname,
                               coords1 = group1[i], coords2 = group2[i],
                               lenra1 = len1[i],     lenra2 = len2[i])
                z = z + 1
        
        report("Running remaining jobs")
        run_mpi_jobs()

        report("Prepping count matrices")
        z = 0
        count_matrix_array_array = []
        if no_z1 == True and no_z2 == True:
            count_matrix_array = []
            for a in range(len(abins)-1):
                mat = CountMatrix(filename=(filepref + datname),
                                  tabname=("bin_" + str(z) + str(a)),
                                  load=False, save=True)
                count_matrix_array.append(mat)
            count_matrix_array_array.append(count_matrix_array)
            z = z + 1
        else:
            for i in range(0, len(group1)):
                count_matrix_array = []
                for a in range(len(abins)-1):
                    mat = CountMatrix(filename=(filepref + datname),
                                      tabname=("bin_" + str(z) + str(a)),
                                      load=False, save=True)
                    count_matrix_array.append(mat)
                count_matrix_array_array.append(count_matrix_array)
                z = z + 1
            
        report("Now putting the job results together")
        z = 0
        global mpi_path
        if no_z1 == True and no_z2 == True:
            get_mpi_results(z, datname, filepref, abins, mpi_path,
                            count_matrix_array_array[0])
            z = z + 1
        else:
            for i in range(0, len(group1)):
                get_mpi_results(z, datname, filepref, abins, mpi_path,
                                count_matrix_array_array[z])
                z = z + 1

    
    if load == True:
        #Load relevant metadata
        zbins = pd.read_hdf(filename, key='zbins')[0].tolist()
        abins = pd.read_hdf(filename, key='abins')[0].tolist()
        meta = pd.read_hdf(filename, key='meta')
        no_z1 = meta['no_z1'][0]
        no_z2 = meta['no_z2'][0]
        
        a_range = range(len(abins)-1)
        z_range = range(len(zbins)-1)
        if no_z1 and no_z2:
            z_range = [0]
        
        #key format is bin_xy where x is z bin number and y is angular bin number 
        #Load the correlation
        matrices = {}
        for a in a_range:
            for z in z_range:
                mname = ("bin_" + str(z) + str(a))
                matrices[mname] = CountMatrix(filename=filename, tabname=mname, load=True)
        return matrices
    

def correlate_zbin( **kwargs):
    """    
    @kwargs
    
        file         - the name of the file where the corrset will be saved
        
        datname      - the key to save this corrset to within the file
        
        coords1, 2   - The coordinates for a given z bin, sorted by healpix. 
                       Must be given in shape (n_pix, 2, *), so it is appropriate
                       to assign jobs per z bin using by_bin(), defined below. 
                       
        bins         - the edges of the angular bins, already converted into
                       euclidian!!! So throw them through 
                       angular_dist_to_euclidean_dist(bins) first!
                       
        z            - The number of the z_bin being processed by this job.
                       This number factors into the name (bin_na) where n is
                       the number of the z bin and a the number of the angular 
                       bin. It is also what turn the job will wait for before 
                       saving. 
                      
        lenra1, 2    - The total number of coordinate pairs in this redshift bin
                       for each data set
                       
    """
    file =    kwargs['file']
    coords1 = kwargs['coords1'] 
    coords2 = kwargs['coords2'] 
    z =       kwargs['z'] 
    bins =    kwargs['bins'] 
    lenra1 =  kwargs['lenra1'] 
    lenra2 =  kwargs['lenra2'] 
    datname = kwargs['name'] 
    forcejoin = kwargs.get('forcejoin', False)
    
    #1. Make the KDTrees that will be used for computing the pair counts
    #   Ensure that KDTrees are not made for empty pixels. 
    #   Give a report at the end.
    tree_array = []
    n_trees = 0
    for i in range(len(coords2)):
        ra, dec = np.array(coords2[i][0]), np.array(coords2[i][1])
        if len(ra) == 0: #Empty Pixel
            tree_array.append(None)
        else:
            tree_array.append(KDTree(np.asarray(ra_dec_to_xyz(ra, dec),
                                            order='F').T))
            n_trees = n_trees + 1
    report("correlate_zbin(): Just finished growing "+str(n_trees)+
           " trees corresponding to coords2 shape = "+str(np.shape(coords2))+"\n"+
           "File: " + file + " \n" +
           "Generating MPI inputs")
    
        
    #3. Generate multiprocessing Processes for running the correlations. 
    #   But if either pixel in each pixel-pixel pair is empty, don't add a job.
    global pairs
    global current_jobs
    global total_jobs
    global mpi_path
    for pair in pairs:
        if len(coords1[pair[0]][0]) == 0 or len(coords2[pair[1]][0]) == 0:
            pass
        else:
            #i.   Figure out where inputs and outputs for the mpi helper file
            #     will be saved
            input_path = str(mpi_path + str(current_jobs))
            output_path = str(mpi_path + "_" + datname + "_" + str(z) +
                              "_" + str(pair[0]) + "_" + str(pair[1]) + ".npy")
            
            #ii.  Put together the input for the mpi helper file
            args = {'coords'  : coords1[pair[0]],
                    'tree'    : tree_array[pair[1]],
                    'bins'    : bins,
                    'lenra2'  : lenra2,       
                    'lenra1'  : lenra1,
                    'path'    : output_path}
            
            #iii. Use pickle to save the input, indexed by current_jobs, which
            #     is a stand in for rank
            f = open(input_path, 'wb')
            pickle.dump(args, f)
            f.close()
            
            #iv.  Keep track of how many jobs have been prepared
            current_jobs = current_jobs + 1
            total_jobs   = total_jobs   + 1
        
        #v.  If we've reached the number of cores, run the jobs, thus allowing
        #    for more jobs to be prepared
        if current_jobs == n_cores-1:
            run_mpi_jobs()
    
    #vi.  If the user specifies, join all remaining jobs. 
    if forcejoin:
        run_mpi_jobs()

    report("Ran a total of " + str(total_jobs) + " jobs.")

def run_mpi_jobs():
    """
    mpi4py implementation for multiprocessing.
    
    Runs all the jobs, waits for them to return.
    """
    global current_jobs
    global code_path
    helper = code_path + 'corrset_mpi_helper.py'
    cmd = "mpiexec -n " + str(current_jobs) + " python " + helper
    report("Running " + str(current_jobs) + " jobs. \n" +
           "popen command is as follows: " + cmd)
    
    
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    
    wrapper = io.TextIOWrapper(process.stdout, encoding="utf-8")
    
    jobs_finished = 0
    while jobs_finished < current_jobs:
        for line in wrapper:  
            print("mpi_out_f"+str(jobs_finished)+": "+line, end="")
            if "code:exit" in line:                    
                jobs_finished = jobs_finished+1       
        process.wait()
    
    report("Ran " + str(current_jobs) + " jobs.")
    current_jobs = 0

       
def get_mpi_results(z, datname, filepref, bins, mpi_path, count_matrix_array):
    """
    mpi4py implementation for multiprocessing.
    """
    #i.  Go through each valid pixel-pixel pair
    global pairs
    for pair in pairs:
        #ii.  Figure out what the MPI output name would be
        output_path = str(mpi_path + "_" + datname + "_" + str(z) +
                          "_" + str(pair[0]) + "_" + str(pair[1]) + ".npy")
        
        #iii. Check if there is an output file
        if os.path.isfile(output_path): 
            
            #iv.  If there is, load it and add it to the count matrix array
            output_data = np.load(output_path)
            for a in range(len(bins)-1):
                count_matrix_array[a].mat[pair[0], pair[1]] = output_data[a]
                
    #v.  Save the results
    for a in range(len(bins)-1):
        count_matrix_array[a].save()    
    
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

def by_bin(ra, dec, z, pix, zbins=redshift_bins, nside=nside_default):
    """
    Take the given ra and dec and redshift/heapix information and order the 
    coordinates to take pixel and redshift binning into account. 
    
    @params
        ra, dec - The RA and DEC of the sources to sort
        z       - The redshift of the sources. 
                  Give anything if not binning by redshift. 
        pix     - The healpix of the sources
        zbins   - list of bin edges
                  Give zbins=[] if not binning by redshift
        nside   - healpix nside parameter. Default = nside_default = 8
    
    @returns
        list of shape (zbins, npix, 2, *) where * is the number of coordinates
        in each bin, which will vary from 0 for none in a bin to len(ra) for all
        in one bin.
    """
    n_pix = hp.nside2npix(nside)
    
    if len(zbins) <= 1: #Just sort pixels if zbins is length 0
        to_return = [[]]
        for p in range(n_pix):
            inpix = np.where(pix == p)[0]
            to_return[0].append([[ra[j] for j in inpix],
                                [dec[j] for j in inpix]])
        return to_return, [len(ra)]
        
        
    #Step 1: Construct the list to return.
    groups = []
    for i in range(len(zbins)-1): #Z bin indexing
        groups.append([])
        for j in range(n_pix): #Healpix indexing
            groups[i].append([[],  #The RA list
                              []]) #The DEC list
    
    #Step 2: Ascertain where in the list to return the RA and DEC should go
    indices = to_zbins(z, zbins)
    report("by_bin(): Indices gotten with shape "+str(np.shape(indices))+"\n"+
           "Min: "+str(min(indices))+" and Max: "+str(max(indices)))
    
    #Step 3: Put the RA and DEC in the appropriate place in the list
    len_list = []
    for z in range(len(zbins)-1):
        inbin = np.where(indices == z)[0]
        len_list.append(len(inbin))
        specific_z_binned_ra  = [ra[j]  for j in inbin]
        specific_z_binned_dec = [dec[j] for j in inbin]
        specific_z_binned_pix = np.array([pix[j] for j in inbin], dtype=int)

        for p in range(n_pix):
            inpix = np.where(specific_z_binned_pix == p)[0]
            groups[z][p][0] =  [specific_z_binned_ra[j] for j in inpix]
            groups[z][p][1] = [specific_z_binned_dec[j] for j in inpix]
     
    #Cast as numpy float arrays
    for i in range(len(zbins)-1): #Z bin indexing
        for j in range(n_pix): #Healpix indexing
            groups[i][j][0] = np.array(groups[i][j][0], dtype=float)
            groups[i][j][1] = np.array(groups[i][j][1], dtype=float)
    len_list = np.array(len_list, dtype=int)
    
    report("by_bin(): Done sorting into bins. Returning")
    return groups, len_list
        
def to_zbins(zlist, zbins=redshift_bins):
    """
    Given a set of z values, return a list of indices pointing from zlist to
    zbins, effectively sorting zlist into component bins.
    
    Note: there will be len(zbins)-1 possible indices, starting from 0. 
    
    @params
        zlist - np.array or similar of redshift values
        zbins - list of bin edges
    @returns
        indices from each z in zlist to the appropriate z bin
    """
    indices = np.zeros(np.shape(zlist), dtype=int)
    for i in range(len(zbins)-1):
        in_bin = np.logical_and(np.greater(zlist, zbins[i]),
                                   np.less(zlist, zbins[i+1]))
        for j in np.where(in_bin)[0]: indices[j] = i
    return indices

def get_cols(file, cols, chunksize=None):
    """
    Helper function gets all the columns you want from a given HDF file path.
    Returns a list of pd.Series objects.
    
    @params
        file                  - the name of the file
        cols                  - a list of the names of the columns to get
        chunksize = 1,000,000 - chunks to read the file in.
    """
    coldata = []
    chunks = pd.read_hdf(file, key="primary", chunksize=chunksize)
    if chunksize == None:
        chunks = [chunks]
    for chunk in chunks:
        if coldata == []:
            for colname in cols:
                coldata.append(chunk[colname])
        else:
            for i in range(len(coldata)):
                coldata[i] = coldata[i].append(chunk[colname])
    return coldata

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

def angular_dist_to_euclidean_dist(D, r=1):
    """convert angular distances to euclidean distances"""
    D_to_use = np.array(D, dtype=float)
    return 2 * r * np.sin(0.5 * D_to_use * np.pi / 180.)

def sph2cart(r, phi, theta, degree=False):
    if degree:
        phi = phi * (np.pi/180.)
        theta = theta * (np.pi/180.)
    rcos_theta = r * np.cos(theta)
    x = rcos_theta * np.cos(phi)
    y = rcos_theta * np.sin(phi)
    z = r * np.sin(theta)
    return x, y, z

if __name__ == "__main__":
    main()
    
