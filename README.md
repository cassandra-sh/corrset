# corrset
Cassandra's correlation astronomy code

This code is divided into several parts
1. builder.py
    This part takes the input files (spec-z catalog, hsc catalog, agn catalog, sdss t1 agn spectroscopy catalog) and generates complete cross-match catalogs from them. These catalogs will be suitable for running whole-catalogs statistics    
    a. cross_match.py
        This part does the actual cross matching for builder.py
2. qualifier.py
    This part uses the data from the cross matched catalogs to produce the streamlined catalogs which will be used for correlation statistics
3. corrset.py
    This part makes a corrset, which can be used to produce jackknife resampled auto and cross correlations quickly. The computation time will be frontloaded in making the corrset, and producing different products with it should be fast. 

There may eventually be a file called director.py which interfaces will all of these for science results.
