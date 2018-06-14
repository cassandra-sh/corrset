#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 17:02:19 2018

@author: csh4
"""
import numpy as np
import healpy as hp
import cProfile

def pixpairs(nside):
    """
    Get a list of all HEALPIX neighbor-neighbor pixel-pixel pairs.
    
    No repeats, as in (2, 1) will not appear if (1, 2) appears
    
    @ params nside to be used
    @ returns list of 2 integer lists of (self, other) healpix number. 
    
    performance: probably as fast as possible without cython
    """
    n_pix = hp.nside2npix(nside)
    
    #First do the self-self combos
    self_self = []
    for i in range(n_pix):
        self_self.append([i, i])
        
    #Now do the self-other combos
    self_neighbor = []
    pixel_pairs_groups = [[] for pixel in range(n_pix)]
    for i in range(n_pix):
        to_add = hp.pixelfunc.get_all_neighbours(nside, i)
        for j in to_add:
            if j == -1:
                pass
            else:
                pair = []
                if j > i:
                    pair = [i, j]
                else:
                    pair = [j, i]
                    
                if (pair in pixel_pairs_groups[pair[0]]):
                    pass
                else:
                    pixel_pairs_groups[pair[0]].append(pair)
    for pixel_pairs_group in pixel_pairs_groups:
        for pixel_pair in pixel_pairs_group:
            self_neighbor.append(pixel_pair)
    
    #Finally do the 2 separation combos if nside > 16
    self_separated = []
    if nside > 16:
        pixel_pairs_groups = [[] for pixel in range(n_pix)]
        for pixel in range(n_pix):
            neighbors = hp.pixelfunc.get_all_neighbours(nside, pixel)
            neighbor_neighbor_groups = []
            for neighbor in neighbors:
                if neighbor == -1:
                    pass
                else:
                    group = hp.pixelfunc.get_all_neighbours(nside, neighbor)
                    neighbor_neighbor_groups.append(group)
            
            separated_neighbors = []
            for neighbor_neighbor_group in neighbor_neighbor_groups:
                for neighbor_neighbor in neighbor_neighbor_group:
                    if neighbor_neighbor in neighbors:
                        pass
                    elif neighbor_neighbor == pixel:
                        pass
                    elif neighbor_neighbor == -1:
                        pass
                    elif neighbor_neighbor in separated_neighbors:
                        pass
                    else:
                        separated_neighbors.append(neighbor_neighbor)
            
            pairs = []
            for separated_neighbor in separated_neighbors:
                if separated_neighbor > pixel:
                    pairs.append([pixel, separated_neighbor])
                else:
                    pairs.append([separated_neighbor, pixel])
            
            
            for pair in pairs:
                if pair in pixel_pairs_groups[pair[0]]:
                    pass
                else:
                    pixel_pairs_groups[pair[0]].append(pair)
            
        for pixel_pairs_group in pixel_pairs_groups:
            for pixel_pair in pixel_pairs_group:
                self_separated.append(pixel_pair)
    
    final_pairs =  self_self + self_neighbor + self_separated
    
    print("pixpairs("+str(nside)+") output shape is "+
          str(np.shape(final_pairs)))
    
    return final_pairs

    
def profile():
    cProfile.runctx('pixpairs(4)',  globals(), locals())
    cProfile.runctx('pixpairs(8)',  globals(), locals())
    cProfile.runctx('pixpairs(16)', globals(), locals())
    cProfile.runctx('pixpairs(32)', globals(), locals())
    cProfile.runctx('pixpairs(64)', globals(), locals())

def main():
    profile()

if __name__ == "__main__":
    main()