# corrset
Cassandra's correlation astronomy code

This version uses astropy for all the file handling. The new version uses astropy very minimally. 

In the most general terms, this code is, for now, divided into three parts
1. builder.py
    This module takes the input files (spec-z catalog, hsc catalog, agn catalog, sdss t1 agn spectroscopy catalog) and generates complete cross-match catalogs from them. These catalogs will be suitable for running whole-catalogs statistics
2. qualifier.py
    This module uses the data from the cross matched catalogs to produce the streamlined catalogs which will be used for correlation statistics
3. corrset.py
    This module makes a corrset, which can be used to produce jackknife resampled auto and cross correlations quickly. The computation time will be frontloaded in making the corrset, and producing different products with it should be fast. 

There will also be a file called director.py which interfaces will all of these for science results.
