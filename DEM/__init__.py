#coding: utf-8

"""
        * * * Package DEM * * *
 
This package contains 4 subpackages :
        * mesh_related
        * reconstructions
        * DEM
        * miscellaneous
 
* mesh_related contains all functions that have to do with the mesh and the links between facets, cells and vertices.
* reconstructions contains the computation of the reconstruction matrices and the interpolation for the facet reconstruction.
* DEM contains the definition of the DEMproblem class.
* miscellaneous contains the interpolation of a function in a DEM array and other useful functions like the penalty matrices.
 
"""

# info
__version__ = "1.0"
__author__  = "Frédéric Marazzato"
__date__    = "Friday September 4th 2020"

# import sub modules
from DEM.mesh_related import *
from DEM.reconstructions import *
from DEM.miscellaneous import *
