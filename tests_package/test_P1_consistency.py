# coding: utf-8

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

import sys
sys.path.append('../')
from DEM.DEM import *
from DEM.miscellaneous import DEM_interpolation,local_project
import pytest

#Size of mesh and number of elements
L = 0.5
nb_elt = 3

@pytest.mark.parametrize("mesh", [RectangleMesh(Point(-L,-L),Point(L,L),nb_elt,nb_elt,"crossed"), BoxMesh(Point(-L, -L, -L), Point(L, L, L), nb_elt, nb_elt, nb_elt)])
def test_reconstruction(mesh):
    dim = mesh.geometric_dimension()
    d = dim #scalar problem #dim #vectorial problem

    #DEM problem creation with reconstruction matrices
    problem = DEMProblem(mesh, d, 0.)

    #Testing P1 consistency and that's all
    x = SpatialCoordinate(mesh)
    u = DEM_interpolation(x, problem)
    assert round(max(u),15) == L
    assert round(min(u),15) == -L

    #CR interpolation
    test_CR = Function(problem.CR)
    reco_CR = problem.DEM_to_CR * u
    test_CR.vector().set_local(reco_CR)
    assert round(max(reco_CR),14) == L
    assert round(min(reco_CR),14)  == -L

    #Test on gradient
    gradient = local_project(grad(test_CR), problem.W)
    gradient_vec = gradient.vector().get_local()
    gradient_vec  = gradient_vec.reshape((problem.DG_0.dim() // d,d,dim))
    assert round(min(gradient_vec[:,0,0]),12) == 1. and round(max(gradient_vec[:,0,0]),12) == 1.
    assert round(min(gradient_vec[:,0,1]),12) == 0. and round(max(gradient_vec[:,0,1]),12) == 0.
    assert round(min(gradient_vec[:,1,0]),12) == 0. and round(max(gradient_vec[:,1,0]),12) == 0.
    assert round(min(gradient_vec[:,1,1]),12) == 1. and round(max(gradient_vec[:,1,1]),12) == 1.
    #More tests for 3d functions
    if d == 3:
        assert round(min(gradient_vec[:,0,2]),12) == 0. and round(max(gradient_vec[:,0,2]),12) == 0.
        assert round(min(gradient_vec[:,2,0]),12) == 0. and round(max(gradient_vec[:,2,0]),12) == 0.
        assert round(min(gradient_vec[:,1,2]),12) == 0. and round(max(gradient_vec[:,1,2]),12) == 0.
        assert round(min(gradient_vec[:,2,1]),12) == 0. and round(max(gradient_vec[:,2,1]),12) == 0.
        assert round(min(gradient_vec[:,2,2]),12) == 1. and round(max(gradient_vec[:,2,2]),12) == 1.
        

    #Outputfile
    #file = File('P1_consistency.pvd')
    #file.write(test_CR)
    #file.write(gradient)

    #P1-discontinuous reconstruction
    test_DG_1 = Function(problem.DG_1)
    test_DG_1.vector().set_local(problem.DEM_to_DG_1 * u)
    assert round(max(test_DG_1.vector().get_local()),14) == L
    assert round(min(test_DG_1.vector().get_local()),14) ==  -L

    #Test on gradient
    gradient_DG = local_project(grad(test_DG_1), problem.W)
    gradient_vec = gradient_DG.vector().get_local()
    gradient_vec  = gradient_vec.reshape((problem.DG_0.dim() // d,d,dim))
    assert round(min(gradient_vec[:,0,0]),12) == 1. and round(max(gradient_vec[:,0,0]),12) == 1.
    assert round(min(gradient_vec[:,0,1]),12) == 0. and round(max(gradient_vec[:,0,1]),12) == 0.
    assert round(min(gradient_vec[:,1,0]),12) == 0. and round(max(gradient_vec[:,1,0]),12) == 0.
    assert round(min(gradient_vec[:,1,1]),12) == 1. and round(max(gradient_vec[:,1,1]),12) == 1. 
    #More tests for 3d functions
    if d == 3:
        assert round(min(gradient_vec[:,0,2]),12) == 0. and round(max(gradient_vec[:,0,2]),12) == 0.
        assert round(min(gradient_vec[:,2,0]),12) == 0. and round(max(gradient_vec[:,2,0]),12) == 0.
        assert round(min(gradient_vec[:,1,2]),12) == 0. and round(max(gradient_vec[:,1,2]),12) == 0.
        assert round(min(gradient_vec[:,2,1]),12) == 0. and round(max(gradient_vec[:,2,1]),12) == 0.
        assert round(min(gradient_vec[:,2,2]),12) == 1. and round(max(gradient_vec[:,2,2]),12) == 1.


    #Outputfile
    #file.write(test_DG_1)
    #file.write(gradient_DG)
