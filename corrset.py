#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 11:23:13 2017

Where the magic happens

@author: csh4
"""

import gc
import os
import sys
import time
import psutil
import qualifier
import numpy                   as np
import pandas                  as pd
import healpy                  as hp
import multiprocessing         as mp
import matplotlib.pyplot       as plt
import astroML.correlation     as cr
from   scipy               import spatial
from   corruscant          import twopoint
from   corruscant          import clustering
from   sklearn.neighbors   import BallTree
from   sklearn.neighbors   import KDTree

dir_path = "/scr/depot0/csh4/"
d1d2_p   = dir_path + "cats/corrs/d1d2.hdf5"
d1r_p    = dir_path + "cats/corrs/d1r.hdf5"
d2r_p    = dir_path + "cats/corrs/d2r.hdf5"
rr_p     = dir_path + "cats/corrs/rr.hdf5"

n_cores = mp.cpu_count()
nside_default = 8
angular_bins  = np.logspace(np.log10(.0025), np.log10(1.5), num=12)
redshift_bins = np.linspace(0.3, 1.5, num=4)

start_time = int(time.time())
#Defining a couple in-method helper functions
def current_time():
    return (int(time.time()) - start_time)
def report(report_string):
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
    


CORRELATE_AGAIN = True
QUALIFY_AGAIN = True

def main():
    """
    Handles everything to run the standard correlation we're interested in
    """
    if QUALIFY_AGAIN:
        qualifier.main()

    
    report("Welcome to corrset.main(). Running a correlation on the default catalogs.")
    
    dir_path = "/scr/depot0/csh4/"
    
    p_rand_q = dir_path + "cats/processed/rand_q.hdf5"
    p_agn_q =  dir_path + "cats/processed/agn_q.hdf5"
    p_hsc_q =  dir_path + "cats/processed/hsc_q.hdf5"
        
    angular_bins  = np.logspace(np.log10(.0025), np.log10(1.5), num=12)
    redshift_bins = np.linspace(0.3, 1.5, num=4)
    
    report("Starting correlation process")
    
    corrset = Corrset(filepref="/scr/depot0/csh4/cats/corrs/corr_", 
                      d1=p_agn_q, d2=p_hsc_q, r=p_rand_q, 
                      zbins=redshift_bins, abins = angular_bins,
                      d1names = ['ra', 'dec', 'pix', 'redshift'],
                      d2names = ['ra', 'dec', 'pix', 'redshift'],
                      rnames  = ['ra', 'dec', 'pix'],
                      corr_now = CORRELATE_AGAIN)
    
    corrset.prep_correlation(save=False)
    
    report("Finished generating/loading correlation. Now reading results")
    
    corrs, errs = corrset.read_correlation()
    a_bin_middles = (angular_bins[1:] + angular_bins[:-1]) / 2
    
    report("Finished reading results. Plotting")
    
    fig = plt.figure()
    for i in range(0, len(redshift_bins)-1):
        report("z bin "+str(redshift_bins[i])+", "+str(redshift_bins[i+1])+"\n"+
               "corrs = "+str(corrs[i])+" \nerrs = "+str(errs[i]))
        fig.add_subplot(int("13"+ str(i+1)))
        plt.title(str("z bin " + str(redshift_bins[i]) + ", " + str(redshift_bins[i+1])))
        plt.errorbar(a_bin_middles, corrs[i], yerr=errs[i])
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
         !quickly! and !parallelized!
        -read the hdf for statistical information
        -
    """
    def __init__(self, **kwargs): 
        """
        Reads a corrset from a file, or writes a new corrset to a file. 
        Gets/Initializes the info tables. 
        Gets/Initializes the metadata.
        
        @params
            filepref     --
            
            d1, d2, r    --
            
            d1names,     --
            d2names,
            rnames
            
            zbins, abins --
            
            corr_now     -- Run the correlation right now and save and read out
                            the results.
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
        
        if kwargs['corr_now']:
            self.prep_correlation(True)
        
    
    def read_correlation(self, jackknife_pix=range(0, n_pix)):
        """
        Finds the pair count term associated with a given z_bin.
        Returns the total number of objects, the numpy arrays of counts and the
        bins. 
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
            results, errors = self.jackknife(by_zbin[i], jackknife_pix)
            results_by_bin.append(results)
            errors_by_bin.append(errors)
        
        return results_by_bin, errors_by_bin
    
    def jackknife(self, matrix_group_list, jackknife_pix):
        """
        Take the matrices and the jackknife pix and compute the correlations. 
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
                result = self.correlate(jackknifed_group)
                if len(np.where(np.isnan(result))[0]) > 0 or len(result) == 0:
                    pass
                else:
                    correlation_results.append(result)
        mean = np.mean(np.array(correlation_results), axis=0)
        stdev = np.std(np.array(correlation_results), axis=0)
        return mean, stdev
        
    def check_rowcol_sum(self, matrix, index_list):    
        """
        Return True if the row and column at the indices or index given in 
        index_list have anything other than all zeros. False if not.
        
        Basically, this will be true if any data is contained in the given
        pixel numbers. 
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
        to_ret = np.copy(matrix)
        if type(index_list) is int:
            index_list = [index_list]
        return np.delete(np.delete(to_ret, index_list, axis=0), index_list, axis=1)
    
    def correlate(self, matrix_list):
        """
        @params
            matrix_list shape = (4, len(abins), n_pix, n_pix)
        @returns
            The landy-szalay correlation signal associated with this matrix.
        """
        sumlist = [[],[],[],[]]
        n_pix = np.shape(matrix_list)[2]**2
        print("n_pix for this matrix is " + str(n_pix))
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
            print(sumlist)
            return ((sumlist[0] - sumlist[1] - sumlist[2] + sumlist[3])/sumlist[3])
    
    
    def prep_correlation(self, save):
        """
        Either generates or loads up the correlation, in prep for read_correlation. 
        """
        #Calls pair_count for each relevant set (d1d2, d1r, d2r, rr)
        #Loads if possible instead of running again
        #End goal is to get every count matrix into self.matrices and every name
        #into self.mat_name_list.
        
        #read and prep correlation have to handle all the funky shapes for the
        #correlations involving r.
        
        report("Actually doing the pair counting...")
        
        report("d1d2")
        #D1D2 
        self.pair_count(save, self.d1, self.d2, False, False,
                        'd1d2', self.d1_names, self.d2_names)
        
        report("d1r")
        #D1R
        self.pair_count(save, self.d1,  self.r, False,  True,
                        'd1r',  self.d1_names,  self.r_names)
        
        report("d2r")
        #D2R
        self.pair_count(save, self.d2,  self.r, False,  True,
                        'd2r',  self.d2_names,  self.r_names)
        
        report("rr")
        #RR
        self.pair_count(save,  self.r,  self.r,  True,  True,
                        'rr',   self.r_names,   self.r_names)
    
        report("done")
    
    def pair_count(self, save, d1, d2, no_z1, no_z2, name, colnames1, colnames2):
        """
        Does the pair counting in a smart parallelized way
        """
        if save == True:
            try:
                os.remove(self.filepref+name)
            except FileNotFoundError:
                pass
        mats  = correlate_dat(save=save, load=True, no_z1=no_z1, no_z2=no_z2,
                              filename=(self.filepref+name), zbins=self.zbins,
                              abins=self.abins, colnames1 = colnames1, colnames2=colnames2,
                              d1=d1, d2=d2)
        self.matrices[name] = mats

  
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
            
        
        
        report("correlate_dat(): Sending to job handler job with filename "+
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
        n = 0
        
        report("correlate_dat(): Run the jobs.")
        #Be cognizant of whether or not redshift is being used in a given analysis
        if no_z1 == True and no_z2 == True:
            correlate_zbin(filename, group1[0], group2[0], n, cart_bins, len1[0], len2[0])
            n = n + 1
        elif no_z1 == True and no_z2 == False:
            for i in range(0, len(group2)):
                correlate_zbin(filename, group1[0], group2[i], n, cart_bins, len1[0], len2[i])
                n = n + 1
        elif no_z1 == False and no_z2 == True:
            for i in range(0, len(group1)):
                correlate_zbin(filename, group1[i], group2[0], n, cart_bins, len1[i], len2[0])
                n = n + 1
        else:
            for i in range(0, len(group1)):
                correlate_zbin(filename, group1[i], group2[i], n, cart_bins, len1[i], len2[i])
                n = n + 1
    
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
    

def correlate_zbin(file, coords1, coords2, z, bins, lenra1, lenra2, nside=nside_default):
    """    
    @params
    
        file         - the name of the file where the corrset will be saved
        
        name         - the key to save this corrset to within the file
        
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
    report("correlate_zbin(): Just finished growing " + str(n_trees) +
           " trees corresponding to coords2 shape = " + str(np.shape(coords2)))
    
    #2. Make empty count matrices to save results to.
    count_matrix_array = []
    for a in range(len(bins)-1):
        mat = CountMatrix(filename=file, tabname=("bin_" + str(z) + str(a)),
                          load=False, save=True)
        count_matrix_array.append(mat)
        
    #3. Generate multiprocessing Processes for running the correlations. 
    #   But if either pixel in each pixel-pixel pair is empty, don't add a job.
    global pairs
    turn = 0
    fin = mp.Value('i', 0)
    save_queue = mp.Manager().list([])
    processes = []
    for pair in pairs:
        if len(coords1[pair[0]][0]) == 0 or len(coords2[pair[1]][0]) == 0:
            pass
        else:
            args = (coords1, coords1, pair, tree_array, bins,
                    count_matrix_array, turn, fin, save_queue,
                    lenra1, lenra2)
            processes.append(mp.Process(target=correlate_pixpair, args=args))
            turn = turn + 1
                
    
    #4. Run the processes.
    #   Start by doing one process per core, until there are few processes left.
    n_processes_started = 0
    n_processes_finished = fin.value
    n_processes_going =  n_processes_started - n_processes_finished
    
    #While we have a large number of processes (< n cores) to go, wait for each
    #one to report finished before adding another to the pile
    report("correlate_zbin(): Starting the " + str(len(processes)) + " job(s)")
    global n_cores
    while n_processes_started < turn:
        if n_processes_going < n_cores:
            processes[n_processes_started].start()
            n_processes_started = n_processes_started + 1
            
            report("correlate_zbin(): Jobs going: "+str(n_processes_going)+"\n"+
                   "Jobs finished: "+str(n_processes_finished)+"\n"+
                   "Jobs started: "+str(n_processes_started)+"\n"+
                   "Total jobs: " + str(len(processes)))
        else:
            time.sleep(.05)
        n_processes_finished = fin.value
        n_processes_going =  n_processes_started - n_processes_finished
    
    #Once we have few jobs to go, wait for them to finish up and get out. 
    for process in processes:
        process.join()
        n_processes_finished = fin.value
        
        report("correlate_zbin(): Joining.\nJobs going: "+str(n_processes_going)+"\n"+
               "Jobs finished: "+str(n_processes_finished)+"\n"+
               "Jobs started: "+str(n_processes_started)+"\n"+
               "Total jobs: " + str(len(processes)))
    

def correlate_pixpair(coords1, coords2, pair, tree_array, bins,
                      count_matrix_array, turn, fin, savequeue,
                      lenra1, lenra2):
    """
    subjob() runs the correlations, waits for its turn to save, and saves. 
    
    I'm trying out a queueing saving system. Once done with computing its result
    each subjob will add its turn number to the savequeue and then wait until
    its number is at the front of the queue. In theory this means that there is
    no pileup and one long time calculation can work in peace while the shorter
    ones go ahead. 
    """
    result = two_point_angular_corr_part(coords1[pair[0]][0],
                                                 coords1[pair[0]][1],
                                                 tree_array[pair[1]], bins)
    result = np.array(np.diff(result), dtype=float)/(lenra1*lenra2)
        
    
    savequeue.append(turn)                    #With result, ready to save. Add 
                                              #our turn number to the savequeue
    returned = False
    while not returned:
        current_turn = savequeue[0]           #Now see whose turn it is.
        if current_turn == turn:              #If it is our turn, load the
            for i in range(len(result)):      #matrix, set our values and save.
                count_matrix_array[i].load()
                count_matrix_array[i].mat[pair[0],pair[1]] = result[i]
                count_matrix_array[i].save()
                report("Saving " + count_matrix_array[i].name)
            fin.value = fin.value + 1         #Then report that we have finished
            savequeue.pop(0)                  #and remove our number from the queue
            return                            #finally, return.
        else:
            time.sleep(.05)                   #If it isn't our turn, wait and
                                              #check again.
            
    
    
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
    
    #This part is really slow for some reason so go ahead and try to improve it
    
    #Attempt 2: tried to break the process up into two list construction parts
    #Doubled speed, but it still takes a few minutes for the test sample :\
    
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

if __name__ == "__main__":
    main()
