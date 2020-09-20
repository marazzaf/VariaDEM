# coding: utf-8
import sys
sys.path.append('../')
from DEM.DEM import *
from DEM.miscellaneous import DEM_interpolation,local_project
import pytest

#import pytest #for unit tests
eps = 2e-15 #constant to compare floats. Possible to find a Python constant with that value ?
eps_2 = 1e-12 #constant to compare gradients to zero

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
    assert abs(max(u) - L) < eps
    assert abs(min(u) + L) < eps

    #CR interpolation
    test_CR = Function(problem.CR)
    reco_CR = problem.DEM_to_CR * u
    test_CR.vector().set_local(reco_CR)
    assert abs(max(reco_CR) - L) < eps
    assert abs(min(reco_CR) + L) < eps

    #Test on gradient
    gradient = local_project(grad(test_CR), problem.W)
    gradient_vec = gradient.vector().get_local()
    gradient_vec  = gradient_vec.reshape((U_DG.dim() // d,d,dim))
    assert abs(min(gradient_vec[:,0,0]) - 1.) < eps_2 and abs(max(gradient_vec[:,0,0]) - 1.) < eps_2
    assert abs(min(gradient_vec[:,0,1])) < eps_2 and abs(max(gradient_vec[:,0,1])) < eps_2
    assert abs(min(gradient_vec[:,1,0])) < eps_2 and abs(max(gradient_vec[:,1,0])) < eps_2
    assert abs(min(gradient_vec[:,1,1]) - 1.) < eps_2 and abs(max(gradient_vec[:,1,1]) - 1.) < eps_2
    #More tests for 3d functions
    if d == 3:
        assert abs(min(gradient_vec[:,0,2])) < eps_2 and abs(max(gradient_vec[:,0,2])) < eps_2
        assert abs(min(gradient_vec[:,2,0])) < eps_2 and abs(max(gradient_vec[:,2,0])) < eps_2
        assert abs(min(gradient_vec[:,1,2])) < eps_2 and abs(max(gradient_vec[:,1,2])) < eps_2
        assert abs(min(gradient_vec[:,2,1])) < eps_2 and abs(max(gradient_vec[:,2,1])) < eps_2
        assert abs(min(gradient_vec[:,2,2]) - 1.) < eps_2 and abs(max(gradient_vec[:,2,2]) - 1.) < eps_2
        

    #Outputfile
    #file = File('P1_consistency.pvd')
    #file.write(test_CR)
    #file.write(gradient)

    #P1-discontinuous reconstruction
    test_DG_1 = Function(problem.DG_1)
    test_DG_1.vector().set_local(problem.DEM_to_DG_1 * u)
    assert abs(max(test_DG_1.vector().get_local()) - L) < eps
    assert abs(min(test_DG_1.vector().get_local()) + L) < eps

    #Test on gradient
    gradient_DG = local_project(grad(test_DG_1), W)
    gradient_vec = gradient_DG.vector().get_local()
    gradient_vec  = gradient_vec.reshape((U_DG.dim() // d,d,dim))
    assert abs(min(gradient_vec[:,0,0]) - 1.) < eps_2 and abs(max(gradient_vec[:,0,0]) - 1.) < eps_2
    assert abs(min(gradient_vec[:,0,1])) < eps_2 and abs(max(gradient_vec[:,0,1])) < eps_2
    assert abs(min(gradient_vec[:,1,0])) < eps_2 and abs(max(gradient_vec[:,1,0])) < eps_2
    assert abs(min(gradient_vec[:,1,1]) - 1.) < eps_2 and abs(max(gradient_vec[:,1,1]) - 1.) < eps_2
    #More tests for 3d functions
    if d == 3:
        assert abs(min(gradient_vec[:,0,2])) < eps_2 and abs(max(gradient_vec[:,0,2])) < eps_2
        assert abs(min(gradient_vec[:,2,0])) < eps_2 and abs(max(gradient_vec[:,2,0])) < eps_2
        assert abs(min(gradient_vec[:,1,2])) < eps_2 and abs(max(gradient_vec[:,1,2])) < eps_2
        assert abs(min(gradient_vec[:,2,1])) < eps_2 and abs(max(gradient_vec[:,2,1])) < eps_2
        assert abs(min(gradient_vec[:,2,2]) - 1.) < eps_2 and abs(max(gradient_vec[:,2,2]) - 1.) < eps_2


    #Outputfile
    #file.write(test_DG_1)
    #file.write(gradient_DG)
