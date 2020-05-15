# coding: utf-8
from miscellaneous import *
from reconstructions import *
from mesh_related import *

print(facet_neighborhood.__doc__)

L = 0.5
nb_elt = 5
mesh = RectangleMesh(Point(-L,-L),Point(L,L),nb_elt,nb_elt,"crossed")
#mesh = mesh = BoxMesh(Point(0., 0., 0.), Point(L, L, L), nb_elt, nb_elt, nb_elt)
dim = mesh.geometric_dimension()
d = dim #vectorial problem

#DEM reconstruction
DEM_to_DG, DEM_to_CG, DEM_to_CR, DEM_to_DG_1, nb_dof_DEM = compute_all_reconstruction_matrices(mesh, d)
print('nb dof DEM: %i' % nb_dof_DEM)

#Functional Spaces
U_DG = VectorFunctionSpace(mesh, 'DG', 0) #Pour délacement dans cellules
U_DG_1 = VectorFunctionSpace(mesh, 'DG', 1)
U_CR = VectorFunctionSpace(mesh, 'CR', 1) #Pour interpollation dans les faces
U_CG = VectorFunctionSpace(mesh, 'CG', 1) #Pour bc
W = TensorFunctionSpace(mesh, 'DG', 0)

#Testing P1 consistency and that's all
x = SpatialCoordinate(mesh)
#u = passage_ccG_to_DG.T * local_project(x, U_DG).vector().get_local() + passage_ccG_to_CG.T * local_project(x, U_CG).vector().get_local()
u = DEM_interpolation(x, mesh, d, DEM_to_CG, DEM_to_DG)
#u += 0.5 * np.ones(nb_dof_DEM)

#CR interpolation
test_CR = Function(U_CR)
test_CR.vector().set_local(DEM_to_CR * u)

#Outputfile
file = File('test.pvd')
file.write(test_CR)
file.write(local_project(grad(test_CR), W))

#test_DG_1 = Function(U_DG_1)
#test_DG_1.vector().set_local(passage_ccG_to_DG_1 * u)
#print(min(u))
#print(min(test_DG_1.vector().get_local()))
#print(max(u))
#print(max(test_DG_1.vector().get_local()))
#file.write(test_DG_1)
#file.write(local_project(grad(test_DG_1), W))




