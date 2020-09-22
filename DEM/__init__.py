#coding: utf-8

#Copyright 2020 Frédéric Marazzato

#This file is part of VariaDEM.

#VariaDEM is free software: you can redistribute it and/or modify
#it under the terms of the GNU Lesser General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#VariaDEM is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#Lesser GNU General Public License for more details.

#You should have received a copy of the Lesser GNU General Public License
#along with VariaDEM.  If not, see <http://www.gnu.org/licenses/>.


"""
        * * * Package VariaDEM * * *
 
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
__version__ = "0.1"
__author__  = "Frédéric Marazzato"
__date__    = "Friday September 22th 2020"

# import sub modules
from DEM.mesh_related import *
from DEM.reconstructions import *
from DEM.miscellaneous import *
