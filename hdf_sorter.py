# -*- coding: utf-8 -*-
"""
hdf_sorter.py
@author Cassandra Henderson
cassandra.s.henderson@gmail.com

Program Description:
    Contains some useful methods for qualifier.py
    
These are all memory conservative sorting and normalizing functions using
pandas DataFrames - they all utilize chunking, and thus the performance is
slower, but the memory use is very low. 

"""

import os
import glob
import pandas as pd
import numpy as np
import shutil
import healpy as hp
import file_manager

chks_def = 2000000  #Default chunksize. Really depends on the catalogs you're
                    #using and the number of objects in them, and amount of
                    #other data, system memory, etc. But 2,000,000 is an ok
                    #guess and doesn't use too much memory. 

def proc_pair(D, R, temp_dir, target_dir, corr_num, term_num,
              pixelate_nside, zbins, use_zbins, norm, 
              normalize_nside=8, chunksize=chks_def,
              D_key_in = 'primary',  R_key_in = 'primary',
              D_key_out = 'primary', R_key_out = 'primary'):
    """
    Given a pair of HDFStore file locations D and R, treat them as data and
    random catalogs to be prepared for mpi_prepping
    
    @params
        D - file path to the HDFStore object that contains data for this term
        R - file path for the corresponding random catalog
        temp_dir   - where temporary files will be saved
        target_dir - where the output will be saved. Needs to be clean, and
                     will be managed by proc_pair. 
        corr_num   - the number of this correlation (i.e. C0 -> 0)
        term_num   - the number for this term (i.e. D0 -> 0)
                     C0D1, C0R1 as D and R means corr_num = 0, term_num = 1
        
        D_key_in, R_key_in   - hdf keys for the stores used as input
        D_key_out, R_key_out - hdf keys for desired output. This code uses
                               'primary' as default for a single-hdf HDFStore
                               file. 
    
    returns 5 arrays of dtypes (int, int, int, array(str), array(str))
    
            the population per pixel of the random catalog (before normalizing),
            of the data catalog, of the random catalog (after normalizing), the
            names of the data files saved to [per z bin], the names of the 
            random files saved to [per z bin]
    """
    #Step 0. Determine file names
    d_pref = "C" + str(corr_num)            + "D" + str(term_num)
    r_pref = "C" + str(corr_num)            + "R" + str(term_num)
    r_temp = "C" + str(corr_num) + "_temp_" + "R" + str(term_num)
    
    
    #Step 0.5 Get the population per pixel and z distribution before any divison
    dpop = per_pix(D, nside=pixelate_nside, key=D_key_in,
                   chunksize=chunksize)
    
    to_print = {}
    total = 0
    for i in range(len(dpop)):
        if dpop[i] > 0:
            to_print.update({str(i):dpop[i]})
            total = total + dpop[i]
    print("data catalog population info = ")
    print(to_print)
    print("total num objects = " + str(total))
    print("total num valid pixels = " + str(len(to_print)))
    
    rpop_old = per_pix(R, nside=pixelate_nside, key=R_key_in,
                   chunksize=chunksize)
    
    to_print = {}
    total = 0
    for i in range(len(rpop_old)):
        if rpop_old[i] > 0:
            to_print.update({str(i):rpop_old[i]})
            total = total + rpop_old[i]
    print("random catalog population info = ")
    print(to_print)
    print("total num objects = " + str(total))
    print("total num valid pixels = " + str(len(to_print)))
    
    rpop = []
    
    
    zhist, zedges = [], []
    if use_zbins:
        zedges = np.linspace(zbins[0], zbins[-1], 30)
        zhist = z_histogram(D, zedges, key=D_key_in, chunksize=chunksize)
    
    #Step 1. Normalize the random catalog in preparation to split it
    if norm:
        normalize(D, R, temp_dir, target_dir+r_temp, nside = normalize_nside,
                  chunksize = chunksize, target_key=R_key_in, 
                  a_key = R_key_in, b_key = D_key_in)
        rpop = per_pix((target_dir+r_temp), nside=pixelate_nside,
                        key=R_key_in, chunksize=chunksize)
    else:
        shutil.copy(R, (target_dir + r_temp))
        
        store = pd.HDFStore(target_dir + r_temp)
        store.get_node(R_key_in)._f_rename('primary')
        store.close()
    
        rpop = rpop_old
    
    #Step 2. Remove old files in the target directory under the same name
    #        as the new ones
    target_directory_files = glob.glob(target_dir + "/*")
    for file_path in target_directory_files:
        file_name = file_path.split("/")[-1]
        if (d_pref in file_name) or (r_pref in file_name):
            os.remove(file_path)
            
    #Step 3. Split the data catalog, first by redshift, then by pixel
    d_files = div_by_z(D, target_dir, d_pref, zbins, no_z=(not use_zbins),
                       key_arg=D_key_in, chunksize=chunksize)
    
    d_outputs = []
    d_pix_pops = []
    for file in d_files:
        outs, pix, pop = div_by_pix(file, target_dir, file.split(sep="/")[-1],
                                    nside=pixelate_nside, chunksize=chunksize,
                                    key_target=D_key_out)
        d_outputs.append(outs)
        d_pix_pops.append(pop)
    
    #Step 4. Split the random catalog, [first copying for redshift],
    #        then splitting by pixel
    r_files = div_by_z(target_dir+r_temp, target_dir, r_pref, zbins,
                       no_z=(not use_zbins), chunksize=chunksize) 
   
    r_outputs = []
    r_pix_pops = []
    for file in r_files:
        outs, pix, pop = div_by_pix(file, target_dir, file.split(sep="/")[-1],
                                    nside=pixelate_nside, chunksize=chunksize,
                                    key_target=R_key_out)
        r_outputs.append(outs)
        r_pix_pops.append(pop)
    
    return (rpop_old, dpop, rpop, d_outputs, r_outputs,
            d_pix_pops, r_pix_pops, zhist, zedges) 

def normalize(a, b, temp, target, nside=8, chunksize=chks_def,
              a_key = "primary", b_key = "primary", target_key = "primary"):
    """
    Normalize b to a, and save result to target. Use the directory temp for 
    saving intermediate products. 

    Params:
        a, b   - str paths to input HDFs
        target - str path where output HDF will be saved
        temp   - directory where temporary HDFs will be stored during
                 calculation. Will make temp if temp does not exist. 
    
    Results:
        target should be saved with a catalog that is normalized to a. 

    Returns:
        Number per pix for both catalogs and for the target catalog in form
        [per_pix(a), per_pix(b), per_pix(target)]
    
    """
    #0. Clear temp directory
    file_manager.ensure_dir(temp)
    file_manager.empty_dir(temp)
    
    #1. Count the number density of a and b per pixel (defined by nside) 
    apop = per_pix(a, nside=nside, chunksize=chunksize, key=a_key)
    bpop = per_pix(b, nside=nside, chunksize=chunksize, key=b_key)
    
    #2. Compute normalization factor (out of 1) per pixel to normalze b to a
    apop_frac = apop/np.max(apop)
    bpop_frac = bpop/np.max(bpop)
    
    ratio = apop_frac/bpop_frac
    
    for i in range(len(ratio)):
        if np.isnan(ratio[i]) or np.isinf(ratio[i]):
            ratio[i] = 0.0
    
    ratio_normed = (ratio/np.max(ratio))
    
    print("Normalized ratio for normalizing " + b + " to " + a + ":")
    print(str(ratio_normed))
    
    #3. Divide b by pixel, saving to temp
    files, pix, pop = div_by_pix(b, temp, "temp", key_arg = b_key)
    
    st = pd.HDFStore(target, mode='w')
    st.close()
    
    #5. Draw factor*N_rand_in_pixel objects from the temp hdf per pixel,
    #      adding to target.
    for i in range(len(pix)):
        draw_into_target(files[i], ratio_normed[pix[i]],
                         target, chunksize=chunksize, key_target=target_key)
        
    #6. Count the number density of target
    targpop = per_pix(target, nside=nside, chunksize=chunksize, key=target_key)
    
    return [apop, bpop, targpop]


def per_pix(hdf, nside=8, key="primary", chunksize=chks_def):
    """
    Count the number of sources per pixel in an HDF
    
    @params
        hdf   - file path to the HDF file
        nside - nside for healpix
        key   - name of hdf within the HDF file
        
    @returns
        a numpy array of the numbers of objects per pixel. 
    """
    pop = np.zeros(hp.nside2npix(nside), dtype=int)
    chunks = pd.read_hdf(hdf, key=key, chunksize=chunksize,
                         columns=['ra', 'dec', 'redshift'])
    if chunksize == None: chunks = [chunks]
    for chunk in chunks:
        chunk_pix = hp.ang2pix(nside, chunk['ra'], chunk['dec'], lonlat=True)
        pop = pop + np.bincount(chunk_pix, minlength=hp.nside2npix(nside))
    return pop

def draw_into_target(arg, frac, target,  key_arg="primary",
                     key_target="primary", chunksize=chks_def):
    """
    Draw a random number of rows from arg as a fraction of the total and
    add those rows to target
    
    @params
        arg  - path to hdf file to draw from
        frac - fraction of rows to take from each chunk
        target - hdf to save the draw to
        key_arg  - hdf key to pull within arg
        key_target - hdf key to add to within target
        chunksize - for chunking
    
    """
    if frac == 0.0:
        return
    
    #Open arg by chunk 
    chunks = pd.read_hdf(arg, key=key_arg, chunksize=chunksize,
                         columns=['ra', 'dec', 'redshift'])
    
    if chunksize == None: chunks = [chunks]
    for chunk in chunks:
        #Step 1: Get information about the chunk
        chunk_length = len(chunk.index)
        
        #Step 2: Get random rows to use from the chunk, frac of total
        rows_to_use = np.random.choice(len(chunk.index), int(frac*chunk_length),
                                       replace=False)
        
        if len(rows_to_use) > 0:
            #Step 3: Draw out frac fraction of the total rows
            draw = chunk.iloc[rows_to_use]
            
            #Step 4: Add to the HDFStore
            add_to_hdfstore(draw, target, key_target)
    
    
 
def div_by_pix(arg, targ_dir, filepref, nside=8, chunksize=chks_def,
                key_arg="primary", key_target="primary"):
    """
    Take an HDF and save 
    
    @params
        arg       - str file path to the HDF store file to sort
        targ_dir  - str directory where output HDFs will be saved
        filepref  - beginning of name for the output HDFs. 
                        out format is filepref + "_pix#"
        nside=8   - nside parameter for healpix
        chunksize - chunksize for chunking
        key_arg   - key of desired input hdf within arg HDFStore
        key_target- key of desired output hdfs
        
    @returns
        list of file paths saved to, list of pixels used
    """
    #Erase the output files if they already exist
    for pixel in range(hp.nside2npix(nside)):
        store = (targ_dir + filepref + "_pix" + str(pixel))
        if os.path.isfile(store):
            os.remove(store)
    
    #Open the arg in chunks
    n_pix = hp.pixelfunc.nside2npix(nside)
    files_used, pix_usage = [], np.zeros(n_pix)
    chunks = pd.read_hdf(arg, key=key_arg, chunksize=chunksize,
                         columns=['ra', 'dec', 'redshift'])
    if chunksize == None: chunks = [chunks]
    for chunk in chunks:
        
        #Step 1: Figure out what pixels are being used in this chunk, by each
        #        row and in total.
        chunk_pix = hp.ang2pix(nside, chunk['ra'], chunk['dec'], lonlat=True)
        pix_usage_this_chunk = np.bincount(chunk_pix, minlength=n_pix)
        pix_usage = pix_usage + pix_usage_this_chunk
        
        #Step 2: For each pixel in use, get a DataFrame subset of rows 
        #        corresponding to that pixel
        for pixel in [i for i in np.where(pix_usage_this_chunk > 0)[0]]:
            rows_used = np.where(chunk_pix == pixel)[0]
            df_subset = chunk.iloc[rows_used]
            
            #Step 3: And add it to the relevant HDFStore
            store = (targ_dir + filepref + "_pix" + str(pixel))
            add_to_hdfstore(df_subset, store, key_target)
            
            #Step 4: Add the name of the file used to the list, if it isn't
            #        already in it
            if store not in files_used:
                files_used.append(store)
    return files_used, [i for i in np.where(pix_usage > 0)[0]], pix_usage
    
def add_to_hdfstore(df, store, key, reindex=True, mode='a'):
    """
    Adds a DataFrame object to an existing HDF
    
    @params
        df      - DataFrame object to add
        store   - file path to HDFStore 
        key     - key to hdf within store
        reindex - whether or not to continue the hdf's indexing
        mode    - whether or not to overwrite the existing hdf with df
                    a - append
                    w - write
    """
    #Step 1: Open the HDFStore for the target
    st = pd.HDFStore(store, mode=mode)
    
    #Step 2: Figure out how many rows the hdf store has for in hdf key
    nrows = 0
    if ("/"+key) in st.keys():
        nrows = st.get_storer(key).nrows
        if nrows == None:
            nrows = 0
    
    #Step 3: (Optional) Reindex df to contine the hdf store indexing
    if reindex:
        df.reset_index(drop=True, inplace=True)
        df.set_index([list(range(nrows, nrows+len(df)))], inplace=True)
    
    #Step 4: Add to target.
    if mode == 'w':
        st.put(key, df, format="table")
    elif nrows > 0:
        st.append(key, df, format="table", append=True)
    elif mode == 'a': 
        st.put(key, df, format="table")
        
    #Step 5: Close the store.
    st.close()
    

def zbins_calc(z, zbins):
    """
    Calculates the zbin being used by each z
    
    @params
        z     - numpy array of redshift in use
        zbins - zbin edges
    @returns
        index to each zbin (min value 0, max value len(zbins)-1)
        if not in any zbin, value used is len(zbins)-1 (1 above max)
    """
    n_zbins = len(zbins)-1
    indices = n_zbins*np.ones(len(z), dtype=int)
    for n in range(n_zbins):
        print(" range between " + str(zbins[n]) + " and " + 
              str(zbins[n+1]))
        in_this_bin = np.logical_and(np.greater_equal(z, zbins[n]),
                                     np.less(z, zbins[n+1]))
        print("in this bin are " + str(len(np.where(in_this_bin)[0])))
        for i in np.where(in_this_bin)[0]:
            indices[i] = n
    return indices

def copy(arg, targ, key_arg = "primary", key_targ = "primary",
         chunksize=chks_def, columns = ['ra', 'dec']):
    
    chunks = pd.read_hdf(arg, key_arg, chunksize=chunksize,
                         columns=columns)
    if chunksize == None:
        chunks = [chunks]
    
    for chunk in chunks:
        add_to_hdfstore(chunk, targ, key_targ)
        
def z_histogram(D, edges, key='primary', chunksize=chks_def):
    """
    Get the redshift from the given data file and compute the histogram with
    the given edges
    
    @params
        D        - str file path to the HDF file to read redshift from
        key      - key for hdf to use within the given HDF file
        edges    - edges of the histogram to use as range in np.hist
        
    @returns
        numpy array of the histogram values
    """
    zhist = []
    chunks = pd.read_hdf(D, key=key, chunksize=chunksize,
                         columns=['redshift'])
    if chunksize == None: chunks = [chunks] 
    for chunk in chunks:
        chunk_z = chunk['redshift']
        chunk_hist, chunk_edges = np.histogram(chunk_z, bins=edges,
                                               density=True) 
        
        if zhist == []:
            zhist = chunk_hist
        else:
            zhist = zhist + chunk_hist
            
    
    return zhist

def div_by_z(arg, targ_dir, filepref, zbins, no_z=False,
                key_arg="primary", key_target="primary",
                chunksize=chks_def):
    """
    Use the redshift measurement for each source in the given hdf to divide
    into the given zbins. Save the resulting catalogs as HDFs in targ_dir
    
    IF NO 'redshift' COLUMN IS IN hdf - SAVES ONE COPY OF hdf PER ZBIN
    (note: there are len(zbins)-1 bins, as zbins contains the edges)
    
    If you want only one copy per zbin, enter no_z as True
    
    Does not preserve redshift in the divided catalogs
    
    @params
        arg      - str file path to the HDF file to sort
        targ_dir - str directory where output HDFs will be saved
        zbins    - list of floats, bin edges of the redshift bins desired
        no_z     - If True, then copy to _zbin0 and return. [1 copy only]
        filepref - beginning of name for the output HDFs. 
                        out format is filepref + "_zbin#"
        key      - key for hdf to use within the given HDF file
        
    @returns
        list of file paths saved to, and a list of the corresponding pixel nums
    """
    print("")
    print(arg)
    print(targ_dir)
    print(no_z)
    print(zbins)
    
    #Clear the output files
    for zbin_num in range(len(zbins)-1):
        store = (targ_dir + filepref + "_zbin" + str(zbin_num))
        if os.path.isfile(store):
            os.remove(store)
    
    if no_z:
        store = (targ_dir + filepref + "_zbin0")
        copy(arg, store, key_arg=key_arg, key_targ=key_target,
             chunksize=chunksize)
        return [store]
    
    
    #Open the arg in chunks
    z_used, z_checked = True, False
    files_used = []  
    
    chunks = pd.read_hdf(arg, key=key_arg, chunksize=chunksize,
                         columns=['ra', 'dec', 'redshift'])
    if chunksize == None: chunks = [chunks] 
    for chunk in chunks:
        #Figure out if redshift is being used in this file.
        if not z_checked:
            try:
                chunk_z = chunk['redshift']
            except KeyError:
                z_used = False

        if not z_used:        
            #Branch 1. If redshift is not in use, simply copy the chunk into  
            #          each of the desired output files
            for zbin in range(len(zbins)-1):
                store = (targ_dir + filepref + "_zbin" + str(zbin))
                add_to_hdfstore(chunk, store, key_target)
                if store not in files_used:
                    files_used.append(store)
                
        else:
            #Branch 2.1: Figure out what z bins are being used in this chunk, by 
            #            each row and in total.
            chunk_z = np.array(chunk['redshift'], dtype=float)
            chunk_zbin = zbins_calc(chunk_z, zbins)
            zbin_usage = np.bincount(chunk_zbin, minlength=len(zbins))
            zbins_used = np.where(zbin_usage > 0)[0]
            #Branch 2.2: For each pixel in use, get a DataFrame subset of rows 
            #            corresponding to that pixel
            for zbin in zbins_used:
                if zbin == len(zbins)-1:
                    pass
                else:
                    rows_used = np.where(chunk_zbin == zbin)[0]
                    df_subset = chunk[['ra', 'dec']].iloc[rows_used]
                    
                    #Branch 2.3: And add it to the relevant HDFStore
                    store = (targ_dir + filepref + "_zbin" + str(zbin))
                    add_to_hdfstore(df_subset, store, key_target)
                    if store not in files_used:
                        files_used.append(store)
    return files_used

