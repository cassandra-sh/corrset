#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 11:23:13 2017

Where the magic happens

@author: csh4
"""

import sys
import gc
import time
import psutil
import numpy                   as np
import pandas                  as pd
import healpy                  as hp
import multiprocessing         as mp
import matplotlib.pyplot       as plt
import astroML.correlation     as cr
from   scipy               import spatial
from   .utils              import check_random_state
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
for i in range(len(n_pix)):
    pairs.append([i, i])
#Now do the self-other combos
for i in range(len(n_pix)):
    to_add = hp.get_all_neighbors(nside_default, i)
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
    Handles everything to run the standard correlation we're interested in
    """    
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
                      corr_now = True)
    
    report("Finished generating correlation. Now reading results")
    
    corrs = corrset.read_correlation()
    a_bin_middles = (angular_bins[1:] + angular_bins[:-1]) / 2
    
    report("Finished reading results. Plotting")
    
    fig = plt.figure()
    for i in range(0, len(redshift_bins)-1):
        fig.add_subplot(int("13"+ str(i+1)))
        plt.title("z bin " + str(redshift_bins[i] + ", " + str(redshift_bins[i+1])))
        plt.plot(a_bin_middles, corrs[i])
        plt.loglog()
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
        self.lens = {}
        
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
        part = [[], [], [], []] #lists for d1d2, d1r, d2r, rr for each z bin
        by_zbin = []
        for i in range(len(self.zbins)-1):
            by_zbin.append(part)
        
        #Go through each dict and pull out the relevant bins per list.
        #What we want: shape of by_zbin = (len(zbins), 4, len(abins))
        matrix_dicts = [d1d2, d1r, d2r]
        for i in range(len(self.zbins)-1):
            for j in range(len(matrix_dicts)):
                for k in range(len(matrix_dicts[j])):
                    matrix = matrix_dicts[j]['bin_' + str(i) + str(k)].mat
                    by_zbin[i][j].append(matrix)
        for k in range(len(rr)):
            matrix = rr['bin_' + str(i) + str(k)].mat
            for i in range(len(part)): 
                by_zbin[i][3].append(matrix)
        
        #Now that we have the matrices, it's time to jackknife for each bin.
        results_by_bin = []
        errors_by_bin = []
        for i in range(len(self.zbins)-1):
            results, errors = self.jackknife(by_zbin[i], jackknife_pix)
            results_by_bin.append(results)
            errors_by_bin.append(errors)
        
        return results_by_bin
    
    def jackknife(self, matrix_list, jackknife_pix):
        """
        Take the matrices and the jackknife pix and compute the correlations. 
        """
        results = []
        for pix in jackknife_pix:
            to_correlate = []
            for matrix in matrix_list:
                to_correlate.append(self.drop_rowcol(matrix,pix))
            results.append(self.correlate(to_correlate))
        results = np.array(results)
        mean = np.mean(results, axis=0)
        stdev = np.stdev(results, axis=0)
        return mean,stdev
        
        
    
    def drop_rowcol(matrix, index):
        return np.delete(np.delete(matrix, index, axis=0), index, axis=1)
    
    def correlate(matrix_list):
        """
        @params
            matrix_list shape = (4, len(abins))
        @returns
            The landy-szalay correlation signal associated with this matrix.
        """
        sumlist = [[],[],[],[]]
        for i in range(len(matrix_list)):
            for j in range(len(matrix_list[i])):
                sumlist[i].append(matrix_list[i][j].sum())
        for i in range(len(matrix_list)):
            sumlist[i] = np.array(sumlist[i], dtype=float)
        
        return ((sumlist[0] - sumlist[1] - sumlist[2] - sumlist[3])/sumlist[3])
    
    
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
        mats= term(save=save, load=True, no_z1=no_z1, no_z2=no_z2,
                   filename=(self.filepref+name), zbins=self.zbins,
                   abins=self.abins, colnames1 = colnames1, colnames2=colnames2)
        self.matrices[name] = mats

        
def term( **kwargs):
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
        
        #Save the metadata
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
        
        #Run the correlation
        job_handler(ra1, dec1, z1, pix1, no_z1,
                    ra2, dec2, z2, pix2, no_z2,
                    abins, zbins, filename)

    if load == True:
        #Load relevant metadata
        zbins = pd.read_hdf(filename, key='zbins')[0].tolist()
        abins = pd.read_hdf(filename, key='abins')[0].tolist()
        meta = pd.read_hdf(filename, key='meta')
        no_z1 = meta['no_z1'][0]
        no_z2 = meta['no_z2'][0]
        
        a_range = range(len(abins))
        z_range = range(len(zbins))
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
        if load:
            self.load()
        if save:
            self.save()
            
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



def job_handler(ra1, dec1, z1, pix1, no_z1, 
                ra2, dec2, z2, pix2, no_z2, 
                angular_bins, zbins, filename):
    """
    Function which assigns jobs to a multiprocessing queue
    
    @params
        ra1, dec1, ra2, dec2 - coordinate pairs for both sides of pair count
        z1, z2               - redshifts of both sides
        pix1, pix2           - healpix pixel numbers for both sides
        no_z1, no_z2         - whether or not to use redshift in the analysis of
                               each side of the pair count.
        angular_bins         - angular bins to use in the analysis
        zbins                - redshift bins to use in the analysis
        filename             - the hdf file to save the results in
    
    """
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
    group1 = by_bin(ra1, dec1, z1, pix1, zbins=zbins1)
    group2 = by_bin(ra2, dec2, z2, pix2, zbins=zbins2)
    
    #Generate the arguments to the processes
    jobin = []
    turn = mp.Value('i', 0)
    n = 0
    
    #Be cognizant of whether or not redshift is being used in a given analysis
    if no_z1 == True and no_z2 == True:
        jobin.append([filename, group1[0], group2[0], n, cart_bins, turn])
        n = n + 1
    elif no_z1 == True and no_z2 == False:
        for zbin in group2:
            jobin.append([filename, group1[0], zbin, n, cart_bins, turn])
            n = n + 1
    elif no_z1 == False and no_z2 == True:
        for zbin in group2:
            jobin.append([filename, zbin, group2[0], n, cart_bins, turn])
            n = n + 1
    else:
        for i in range(0, len(group1)):
            jobin.append([filename, group1[i], group2[i], n, cart_bins, turn])
            n = n + 1
    
    #Make the processes
    processes = [mp.Process(target=job, args=j) for j in jobin]
    
    #Start the process, one per core. 
    n_processes_started = 0
    n_processes_finished = turn.value
    n_processes_going =  n_processes_started - n_processes_finished
    
    #While we have a large number of processes (< n cores) to go, wait for each
    #one to report finished before adding another to the pile
    while n_processes_finished+n_cores < n:
        if n_processes_going < n_cores:
            processes[n_processes_started].start()
            n_processes_started = n_processes_started + 1
        else:
            time.sleep(1)
        n_processes_finished = turn.value
        n_processes_going =  n_processes_started - n_processes_finished
    
    #Once we have few jobs to go, wait for them to finish up and get out. 
    for process in processes:
        process.join()
    

def job(file, coords1, coords2, z, bins, turn, nside=nside_default):
    """
    End point process, which computes a correlation set and saves it.
    
    Designed to be fed into multiprocessing.Process()
    
    @params
    
        file         - the name of the file where the corrset will be saved
        
        name         - the key to save this corrset to within the file
        
        coords1, 2   - The coordinates for a given z bin, sorted by healpix. 
                       Must be given in shape (n_pix, 2, *), so it is appropriate
                       to assign jobs per z bin using by_bin(), defined below. 
                       
        bins         - the edges of the angular bins, already converted into
                       euclidian!!! So throw them through 
                       angular_dist_to_euclidean_dist(bins) first!
                       
        n            - The number of the z_bin being processed by this job.
                       This number factors into the name (bin_na) where n is
                       the number of the z bin and a the number of the angular 
                       bin. It is also what turn the job will wait for before 
                       saving. 
    """
    n_pix = hp.nside2npix(nside)
    to_save = np.ndarray((len(bins), len(n_pix), len(n_pix)), dtype=int)
    to_save.fill(0)
    global pairs
    #1. Make the KDTrees that will be used for computing the pair counts
    tree_array = []
    for i in range(len(coords2)):
        ra, dec = coords2[i][0], coords2[i][1]
        tree_array.append(KDTree(np.asarray(ra_dec_to_xyz(ra, dec),
                                            order='F').T))
    #2. Make empty count matrices
    count_matrix_array = []
    for a in range(len(bins)-1):
        mat = CountMatrix(filename=file, tabname=("bin_" + str(z) + str(a)))
        count_matrix_array.append(mat)
    #3. Run correlations
    for pair in pairs:
        #i.   Run the pair counting
        result = two_point_angular_corr_part(coords1[pairs[0]][0],
                                             coords1[pairs[0]][1],
                                             tree_array[pair[1]], bins)
        
        #ii.  Normalize by number of coordinate pairs
        result = np.diff(result)/(len(coords1[pairs[0]][0])*len(coords2[pairs[1]][0]))
        
        #iii. Save each angular bin to its appropriate count matrix
        for i in range(len(result)):
            count_matrix_array[i][pair[0]][pair[1]] = result[i]
    #4. Save the result, but only if it is our turn
    while turn.value != z:
        time.sleep(1)
        
    for countmatrix in count_matrix_array:
        countmatrix.save()
    turn.value = n + 1
            
    
    
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
    #Step 1: Construct the list to return.
    groups = []
    n_pix = hp.nside2npix(nside)
    for i in range(len(zbins)-1): #Z bin indexing
        groups.append([])
        for j in range(len(n_pix)): #Healpix indexing
            groups[i].append([np.array([], dtype=float),  #The RA list
                              np.array([], dtype=float)]) #The DEC list 
    #Step 2: Ascertain where in the list to return the RA and DEC should go
    indices = np.zeros(len(ra), dtype=int)
    if zbins != []:
        indices = to_zbins(z, zbins)
    #Step 3: Put the RA and DEC in the appropriate place in the list
    for i in range(len(ra)):
        np.append(groups[indices[i]][pix[i]][0],  ra[i])
        np.append(groups[indices[i]][pix[i]][1], dec[i])
    return groups
        
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
    for i in range(1, len(zbins)):
        in_bin = np.logical_and(np.greater(zlist, zbins[i-1]),
                                   np.less(zlist, zbins[i]))
        for j in np.where(in_bin)[0]: indices[j] = i
    return indices

def get_cols(file, cols, chunksize=1000000):
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
    sin_ra = np.sin(ra * np.pi / 180.)
    cos_ra = np.cos(ra * np.pi / 180.)

    sin_dec = np.sin(np.pi / 2 - dec * np.pi / 180.)
    cos_dec = np.cos(np.pi / 2 - dec * np.pi / 180.)

    return  [cos_ra * sin_dec, sin_ra * sin_dec, cos_dec]

def angular_dist_to_euclidean_dist(D, r=1):
    """convert angular distances to euclidean distances"""
    return 2 * r * np.sin(0.5 * D * np.pi / 180.)

if __name__ == "__main__":
    main()