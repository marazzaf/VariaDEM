# coding: utf-8
import sys
sys.path.append('../')
from DEM import *
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

    #DEM reconstruction
    DEM_to_DG, DEM_to_CG, DEM_to_CR, DEM_to_DG_1, nb_dof_DEM = compute_all_reconstruction_matrices(mesh, d)
    #print('nb dof DEM: %i' % nb_dof_DEM)

    #Testing P1 consistency and that's all
    x = SpatialCoordinate(mesh)
    u = DEM_interpolation(x, mesh, d, DEM_to_CG, DEM_to_DG)
    assert abs(max(u) - L) < eps
    assert abs(min(u) + L) < eps
    #Ajouter des test unitaires là-dessus.

    #Functional Spaces
    U_DG = VectorFunctionSpace(mesh, 'DG', 0) #Pour délacement dans cellules
    U_DG_1 = VectorFunctionSpace(mesh, 'DG', 1)
    U_CR = VectorFunctionSpace(mesh, 'CR', 1) #Pour interpollation dans les faces
    U_CG = VectorFunctionSpace(mesh, 'CG', 1) #Pour bc
    W = TensorFunctionSpace(mesh, 'DG', 0)

    #CR interpolation
    test_CR = Function(U_CR)
    reco_CR = DEM_to_CR * u
    test_CR.vector().set_local(reco_CR)
    assert abs(max(reco_CR) - L) < eps
    assert abs(min(reco_CR) + L) < eps

    #Test on gradient
    gradient = local_project(grad(test_CR), W)
    gradient_vec = gradient.vector().get_local()
    gradient_vec  = gradient_vec.reshape((U_DG.dim() // d,d,dim))
    assert abs(min(gradient_vec[:,0,0]) - 1.) < eps_2 and abs(max(gradient_vec[:,0,0]) - 1.) < eps_2
    assert abs(min(gradient_vec[:,0,1])) < eps_2 and abs(max(gradient_vec[:,0,1])) < eps_2
    assert abs(min(gradient_vec[:,1,0])) < eps_2 and abs(max(gradient_vec[:,1,0])) < eps_2
    assert abs(min(gradient_vec[:,1,1]) - 1.) < eps_2 and abs(max(gradient_vec[:,1,1]) - 1.) < eps_2
    #Ajouter cas pour 3d

    #Outputfile
    #file = File('P1_consistency.pvd')
    #file.write(test_CR)
    #file.write(gradient)

    #P1-discontinuous reconstruction
    test_DG_1 = Function(U_DG_1)
    test_DG_1.vector().set_local(DEM_to_DG_1 * u)
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
    #Ajouter cas pour 3d


    #Outputfile
    #file.write(test_DG_1)
    #file.write(gradient_DG)


##Testing reconstructions in both 2d and 3d
#@pytest.fixture
#def mesh_for_test(): #2d mesh
#    #mesh = RectangleMesh(Point(-L,-L),Point(L,L),nb_elt,nb_elt,"crossed")
#    #assert test_reconstruction(mesh) == 0
#    return RectangleMesh(Point(-L,-L),Point(L,L),nb_elt,nb_elt,"crossed")
#
#@pytest.fixture
#def mesh_for_test(): #3d mesh
#    #mesh = BoxMesh(Point(-L, -L, -L), Point(L, L, L), nb_elt, nb_elt, nb_elt)
#    #assert test_reconstruction(mesh) == 0
#    return BoxMesh(Point(-L, -L, -L), Point(L, L, L), nb_elt, nb_elt, nb_elt)
