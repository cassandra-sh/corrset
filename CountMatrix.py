# -*- coding: utf-8 -*-
"""
CountMatrix.py
@author Cassandra Henderson
cassandra.s.henderson@gmail.com

Program Description:
    class CountMatrix stores count information with a few convenience
    functions.                               
"""

import gc
import numpy      as np
import pandas     as pd
import healpy     as hp
import camb_model as cm

class CountMatrix:
    """
    CountMatrix is a handler for pair counting information. Each CountMatrix
    corresponds to a given term in a correlation estimator. I.e. for Landy-Szalay
    each term of DD, DR, RR gets a CountMatrix, as well as each redshift bin. 
    
    So a regular autocorrelation with 3 redshift bin will get 9 CountMatrix objects.
    Regular cross correlation gets 12 (D1D2, D1R2, D2R1, R1R2)

    Each CountMatrix stores pair counts in different angular bins and pixel-pixel pairs
    
    The data structure used is a pseudo-matrix (a specially indexed list that has
    special handling to make it (inefficiently) act like a matrix). Matricies were
    originally used but we ran into memory constraints - this is much faster unless
    you are trying to run a correlation for a large data set with many pixels that
    have data in them, in which case moving back to a matrix structure might make
    sense. 
    
    The idea is you can pass it a pixel to "leave out" when evaluating the 
    correlation signal. Thus, you can partition the pixels to jackknife.
    
    self.mats is of shape (n angular bin edges) and contains Pandas DataFrame objects
    with 3 columns each - column 1 for first pixel number, column 2 for second 
    pixel number, and column 3 for counts between those pixels within the bin edge
    
    So 8 angular bins means 9 angular bin edges, and each value is the number of counts
    within that angular value. We have to do a little work to turn that into 
    """
    
    def __init__(self, nside=8, filename="", tabname="",
                 load=False, save=False, abins=[], pop1 = [], pop2 = [],
                 z_hist1 = [], z_hist2=[], z_edges1=[], z_edges2=[], 
                 min_z = 0, max_z = 2.0):
        """
        Initialize a CountMatrix object
        
        @params
            nside - the HEALPIX nside parameter
            abins - the angular bin *edges* associated with this correlation
            filename - prefix for where to save or load from
            tabname  - prefix for various key names to be used internally
                       (probably won't break if I drop this)
            load  - true if you want to load from
            z_*   - deprecated stuff for handling redshift distributions internally
            pop1, pop2 - the population of objects per pixel
        """
        #Get the meta data
        self.name = tabname
        self.file = filename
        self.n    = hp.nside2npix(nside)
        
        self.abins = abins       
        self.nabins = len(abins)
        if self.nabins == 0: self.nabins = 1
        
        # Get the population data for each pixel
        if len(pop1) == 0:
            self.pop1 = np.zeros(self.n, dtype=int)
        else:
            self.pop1 = pop1
        
        if len(pop2) == 0:
            self.pop2 = np.zeros(self.n, dtype=int)
        else:
            self.pop2 = pop2
        
        # Get the redshift information about this count matrix
        self.z_hist1  = z_hist1
        self.z_edges1 = z_edges1
        self.z_hist2  = z_hist2
        self.z_edges2 = z_edges2
        self.min_z = min_z
        self.max_z = max_z
        
        # Get the matrix information (leave them blank for now)
        self.mats = []
        for i in range(self.nabins):
            self.mats.append(pd.DataFrame.from_dict({'x':[], 'y':[], 'z':[]}))
            
        # Save or load from memory
        if save:
            self.save()
        if load:
            self.load()
    

    
    def add(self, d, pair):
        """
        Adds the given data to the matrices
        
        @params
            d    -  list of (int or float) values to add to the matrices. Must 
                    be same length as abins
            pair -  coordinates in matrices to add to (pixel pixel pair)
        """
        for a in range(self.nabins):
            loc = self.mats[a].query('(x == '+str(pair[0])+' and '+
                                      'y == '+str(pair[1])+')').index
            
            if len(loc) == 0:
                df_to_add = pd.DataFrame.from_dict({'x':[pair[0]],
                                                    'y':[pair[1]],
                                                    'z':[d[a]]    })
                self.mats[a] = self.mats[a].append(df_to_add,
                                                   ignore_index=True)
            elif len(loc) > 1:
                print(self.mats[a])
                raise IndexError('self mats a = '+str(a)+" for "+self.name+
                                 " has multiple entries for pair "+str(pair))
            else:
                self.mats[a]['z'][loc[0]] = self.mats[a]['z'][loc[0]] + d[a]
    
    def assign(self, d, pair):
        """
        Assigns the given data to the matrices, overwriting previous values
        
        @params
            d    -  list of (int or float) values to set in the matrices. Must 
                    be same length as abins
            pair -  coordinates in matrices to add to (pixel pixel pair)
        """
        for a in range(self.nabins):
            loc = self.mats[a].query('(x == '+pair[0]+' and '+
                                      'y == '+pair[1]+')').index
            
            if len(loc) == 0:
                df_to_add = pd.DataFrame.from_dict({'x':[pair[0]],
                                                    'y':[pair[1]],
                                                    'z':[d[a]]       })
                self.mats[a] = self.mats[a].append(df_to_add,
                                                   ignore_index=True)
            elif len(loc) > 1:
                print(self.mats[a])
                raise IndexError('self mats a = '+str(a)+" for "+self.name+
                                 " has multiple entries for pair "+str(pair))
            else:
                self.mats[a]['z'][loc[0]] = d[a]
    
    
    def balance(self):
        """
        Balances the matrices by adding the lower diagonal to the upper diagonal
        and setting the lower diagonal to zero.
        """
        
        #
        # For each data frame
        #
        for a in range(self.nabins):
            df = self.mats[a]
            
            #
            # Get the pairs and the reversed pairs
            #
            pairs   = np.array([df['x'], df['y']]).T
            pairs_r = np.array([df['y'], df['x']]).T
            
            #
            # The rule is: every pair should be ascending or equal i.e.
            # pixels a and b should either be stored in a > b or a == b
            #
            indices_to_drop = []
            for i, pair in enumerate(pairs):
                #
                # If this is a self, self pair or in the lower diagonal, pass
                #
                if pair[0] <= pair[1]:
                    pass
                else:
                    #
                    # Otherwise, look for reversed matches
                    #
                    c = (pair == pairs_r).T
                    c = np.logical_and(c[0], c[1])
                    c = np.where(c)[0]
                    if len(c) == 0:
                        pass
                    elif len(c) > 1:
                        print(self.mats[a])
                        raise IndexError('self mats a = '+str(a)+" for "+self.name+
                                         " has multiple entries for pair "+str(pair))
                        
                    #
                    # If they exist, add to the non-reversed match and delete
                    # those pairs from the reversed pairs
                    #
                    else:
                        pairs_r = np.delete(pairs_r, [c[0], i], axis=0)
                        df[i]['z'] = df[i]['z'] + df[c[0]]['z']
                        indices_to_drop.append(c[0])
                        
            #
            # Store result, dropping the identified repeats
            #
            self.mats[a] = df.drop(index=indices_to_drop)
                            
                        
    
    def clear(self):
        """
        Clear the matrices and set them back to zero.
        """
        self.mats = []
        for i in range(self.nabins):
            self.mats.append(pd.DataFrame.from_dict({'x':[], 'y':[], 'z':[]}))
    
    def population(self, cut=[]):
        """
        Count the total population of sources stored in pop1 and pop2.
        
        If pixel numbers to cut are provided, return the count without those
        pixels involved.
        
        @params
            cut  -   optional, list of ints which are the pixel numbers to cut
        
        @returns 2 ints of population in pop1 and pop2
        """
        if cut == []:
            return self.pop1.sum(), self.pop2.sum()
        return np.delete(self.pop1, cut).sum(), np.delete(self.pop2, cut).sum()
    
    def get_signal(self, cut=[]):
        """
        Get the correlation signal in this matrix by counting the pairs in each
        angular range, taking the difference and normalizing it with the
        population. 
        
        @params
            cut  -   optional, list of ints which are the pixel numbers to cut
        
        @returns the correlation signal in this matrix, ready to be put into a
                 correlation signal estimator. 
        """
        #Return term info, normalized
        pop1, pop2 = self.population(cut=cut)
        norm, signal = pop1*pop2, []
        if len(cut) == 0:
            signal =  np.array([self.mats[m]['z'].sum() for m in range(self.nabins)])
        else:
            for m in range(self.nabins):
                df = self.mats[m]
                pairs = np.array([df['x'], df['y']]).T
                
                locations = []
                for c in cut:
                    matches = (c == pairs).T
                    condition = np.logical_or(matches[0], matches[1])
                    for i in np.where(condition)[0]:
                        if i in locations:
                            pass
                        else:
                            locations.append(i)
                signal.append(df.drop(index=locations)['z'].sum())
         
        gc.collect()
        
        return np.diff(signal)/norm

    def pix_area(self):
        """
        Figure out the total amount of data area (i.e. percentage of max data
        per pixel) there is in each pixel for both data sources
        
        @returns a list of length n pix where each float value is the fraction
                 of total data area the pixel has
        """
        pop1_norm = np.sum(self.pop1)
        pop2_norm = np.sum(self.pop2)
        return self.pop1/pop1_norm, self.pop2/pop2_norm
        
    def partition(self, num=16):
        """
        Try to make num equal data area (as specified by pix_area()) groups of 
        pixels, to effectively divide the catalog area into num groups. 
        
        @params  num  -  the number of areas to divide the data into
        
        @returns a list of length = num of the int pixel numbers that comprise
                 each group. Ignores pixels with no data
                 Also returns a list saying the average area per group returned
        
        Will raise an exception if it cannot infer this many groups, due to not
        enough pixels (which requires that you run everything again with higher 
        nside)
        """
        #Prepare the output
        groups = [[] for i in range(num)]
        g_area = np.zeros(num)
        
        #Get the area per pixel for both sources and find the average
        a1, a2 = self.pix_area()
        aa = ( a1 + a2 ) / 2.0
        
        #Order the pixels by area (lowest to highest)
        order = np.argsort(aa).tolist()
        
        #Remove all the pixels with no area in them
        while aa[order[0]] == 0.0:
            order.pop(0)
        
        #Count the number of pixels with nonzero area and compare to desired
        #number of regions. If it is fewer, raise an exception
        if len(order) < num:
            raise ValueError("Asked for "+str(num)+" groups of data, but "+
                             "only "+str(len(order))+" pixels have data.\n"+
                             "CountMatrix: file= "+self.file+" and name"+
                             self.name)
        
        #Otherwise take the num largest pixels and start the groups with them
        else:
            for n in range(num):
                pixel_number = order.pop()
                groups[n].append(pixel_number)
                g_area[n] = g_area[n] + aa[pixel_number]
        
        #While there are still pixels with data that aren't in a group...
        while len(order) > 0:
            #Get the smallest pixel
            next_pixel = order.pop(0)
            
            #Figure out which group is smallest (the first if multiple)
            smallest = np.where(g_area == np.min(g_area))[0][0]
                
            #Add the pixel to the group 
            groups[smallest].append(next_pixel)
            
            #and add the pixel's area to its area
            g_area[smallest] = g_area[smallest] = aa[next_pixel]
            
        g_area = g_area / np.sum(g_area)
        return groups, g_area
    
    def load(self):
        #Load the hdf matrix from the filename and hdf table tabname
        st = pd.HDFStore(self.file+"_meta", mode='r')
        
        #angular bin information
        self.abins = pd.read_hdf(st, key=(self.name+"_abins"))['abins'].tolist()
        self.nabins = len(self.abins)
        if self.nabins == 0: self.nabins = 1
        
        #population information
        df_pop = pd.read_hdf(st, key=(self.name+"_pop"))
        self.pop1 = np.array(df_pop['pop1'], dtype=int)
        self.pop2 = np.array(df_pop['pop2'], dtype=int)
        
        st.close()
        gc.collect()
        
        self.mats = []
        gc.collect()
        
        for m in range(self.nabins):
            df = pd.read_hdf(self.file+"_abin"+str(m),
                             key=self.name+"_abin"+str(m))
            self.mats.append(df)
            gc.collect()


    
    def save(self, dump=False):
        #Save the hdf matrix to the filename and hdf table tabname
        st = pd.HDFStore(self.file+"_meta", mode='w')
        
        #Angular bin information
        df_abins = pd.DataFrame.from_dict({'abins':self.abins})
        df_abins.to_hdf(st, key=(self.name+"_abins"), format='fixed')

        #Population information
        df_pop = pd.DataFrame.from_dict({'pop1':self.pop1, 'pop2':self.pop2})
        df_pop.to_hdf(st, key=(self.name+"_pop"), format="fixed")
        
        st.flush()
        st.close()
        gc.collect()
        
        #Count information        
        for m in range(self.nabins):
            self.mats[m].to_hdf(self.file+"_abin"+str(m), key=(self.name+"_abin"+str(m)),
                                format="fixed", mode='w')
            if dump: self.mats[m] = None
            gc.collect()
        
        gc.collect()
        
def mat_df2index_df(path, key):
    """
    Function for converting old numpy matrix data structured CountMatrix objects
    into pseudo-matrix CountMatrix objects which were stored as Pandas DataFrames HDFs.
    """
    try:
        mat = pd.read_hdf(path, key=key)['mat'].as_matrix()
    except KeyError:
        pass
    df = mat2df(mat)
    df.to_hdf(path, key=key)
    gc.collect()

def mat2df(mat):
    """
    Converts the numpy matrix into a DataFrame pseudomatrix
    """
    x, y, z = [], [], []
    indices = np.where(mat > 0.0)
    x, y = indices[0], indices[1]
    z = [mat[x[i], y[i]] for i in range(len(x))]
    return pd.DataFrame.from_dict({'x':x, 'y':y, 'z':z})

def df2mat(df, shape):
    """
    Convert a DataFrame pseudomatrix into a numpy matrix 
    """
    mat = np.zeros(shape, dtype=float)
    for i in range(len(df)):
        mat[df['x'][i], df['y'][i]] = df['z'][i]
    return mat

"""
Originally I wanted the count matrix object to be able to generate its own models
but it ended up making more sense to handle that stuff separately

The following method was used for that
"""
#    def get_mod(self, method='assume', min_z = None, max_z = None, 
#                angular_range = np.logspace(np.log10(.0025),
#                                    np.log10(1.5), num=cm.res).tolist()):
#        """
#        Compute the expected dark matter halo model profile for the associated 
#        redshift distributions of this CountMatrix.
#        
#        @params
#            method            How to determine what kind of model is being made
#                'assume'   -  if both z histograms are available, use both
#                'single'   -  use the first z histogram twice (autocorrelation)
#                'both'     -  use both z histograms (cross correlation if 
#                                                     they are different)
#        """
#        if (len(self.z_hist1)  == 0 or len(self.z_hist2)  == 0   or
#            len(self.z_edges1) == 0 or len(self.z_edges2) == 0):
#            return [], []
#        
#        
#        if min_z == None:
#            min_z = self.min_z
#        if max_z == None:
#            max_z = self.max_z
#        
#        method_to_use = ''
#        if method == 'assume':
#            if len(self.z_hist2) > 0:
#                method_to_use = 'both'
#            else:
#                method_to_use = 'single'
#        else:
#            method_to_use = method
#            
#        if method_to_use == 'single':    
#            return (cm.mod_from_zhist(self.z_hist1, self.z_hist1,
#                                     self.z_edges1, self.z_edges1,
#                                     min_z = min_z, max_z = max_z,
#                                     angular_range = angular_range)[0],
#                    angular_range)
#        else:
#            return (cm.mod_from_zhist(self.z_hist1, self.z_hist2,
#                                     self.z_edges1, self.z_edges2,
#                                     min_z = min_z, max_z = max_z,
#                                     angular_range = angular_range)[0],
#                    angular_range)