# coding: utf-8
import sys
sys.path.append('../')
from DEM import *

L = 0.5
nb_elt = 3
mesh = RectangleMesh(Point(-L,-L),Point(L,L),nb_elt,nb_elt,"crossed")
#mesh = mesh = BoxMesh(Point(0., 0., 0.), Point(L, L, L), nb_elt, nb_elt, nb_elt)
dim = mesh.geometric_dimension()
d = dim #scalar problem #dim #vectorial problem

#DEM reconstruction
DEM_to_DG, DEM_to_CG, DEM_to_CR, DEM_to_DG_1, nb_dof_DEM = compute_all_reconstruction_matrices(mesh, d)
#print('nb dof DEM: %i' % nb_dof_DEM)

#Testing P1 consistency and that's all
x = SpatialCoordinate(mesh)
u = DEM_interpolation(x, mesh, d, DEM_to_CG, DEM_to_DG)
#u = DEM_interpolation(x[0]+x[1], mesh, d, DEM_to_CG, DEM_to_DG)
#u = DEM_interpolation(as_vector((x[0]+x[1],0.)), mesh, d, DEM_to_CG, DEM_to_DG)
print(max(u))
print(min(u))
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
print(min(reco_CR))
print(max(reco_CR))
#Ajouter des test unitaires là-dessus.

#Outputfile
file = File('P1_consistency.pvd')
file.write(test_CR)
file.write(local_project(grad(test_CR), W))

test_DG_1 = Function(U_DG_1)
test_DG_1.vector().set_local(DEM_to_DG_1 * u)
print(min(test_DG_1.vector().get_local()))
print(max(test_DG_1.vector().get_local()))
file.write(test_DG_1)
file.write(local_project(grad(test_DG_1), W))




