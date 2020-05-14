# coding: utf-8
from facets import *
from reconstructions import *
from mesh_related import *

print(facet_neighborhood.__doc__)

L = 0.5
nb_elt = 10
mesh = RectangleMesh(Point(-L,-L),Point(L,L),nb_elt,nb_elt,"crossed")
dim = mesh.geometric_dimension()
d = dim #vectorial problem

#Function spaces
U_DG = VectorFunctionSpace(mesh, 'DG', 0) #Pour délacement dans cellules
U_DG_1 = VectorFunctionSpace(mesh, 'DG', 1)
U_CR = VectorFunctionSpace(mesh, 'CR', 1) #Pour interpollation dans les faces
U_CG = VectorFunctionSpace(mesh, 'CG', 1) #Pour bc
W = TensorFunctionSpace(mesh, 'DG', 0)

#DEM reconstruction
nb_dof_cells = U_DG.dofmap().global_dimension()
facet_num = facet_neighborhood(mesh)
dico_pos_bary_faces = dico_position_bary_face(mesh,d)
pos_bary_cells = position_cell_dofs(mesh,d)
vertex_associe_face,pos_ddl_vertex,num_ddl_vertex_ccG,nb_dof_ccG = dico_position_vertex_bord(mesh, facet_num, d, dim)
print('nb dof ccG : %i' % nb_dof_ccG)

convexe_num,convexe_coord = smallest_convexe_bary_coord_bis(facet_num,pos_bary_cells,pos_ddl_vertex,dico_pos_bary_faces,dim,d)
print('Convexe ok !')
passage_ccG_to_CR = matrice_passage_ccG_CR(mesh, nb_dof_ccG, convexe_num, convexe_coord, vertex_associe_face, num_ddl_vertex_ccG, d, dim)
passage_ccG_to_CG = DEM_to_CG_matrix(mesh, nb_dof_ccG,num_ddl_vertex_ccG,d,dim)
passage_ccG_to_DG = DEM_to_DG_matrix(nb_dof_cells,nb_dof_ccG)
print('matrices passage ok !')

#mat_grad = gradient_matrix(mesh, d)
passage_ccG_to_DG_1 = DEM_to_DG_1_matrix(mesh, nb_dof_ccG, d, dim, passage_ccG_to_CR)

#test P1 consistency and that's all
x = SpatialCoordinate(mesh)
func = Expression(('x[0]', 'x[1]'), degree = 1)
u = passage_ccG_to_DG.T * interpolate(func, U_DG).vector().get_local() + passage_ccG_to_CG.T * interpolate(func, U_CG).vector().get_local()
u += np.ones(nb_dof_ccG)

test_DG_1 = Function(U_DG_1)
test_DG_1.vector().set_local(passage_ccG_to_DG_1 * u)
print(min(u))
print(min(test_DG_1.vector().get_local()))
print(max(u))
print(max(test_DG_1.vector().get_local()))

file = File('test.pvd')
file.write(test_DG_1)
file.write(local_project(grad(test_DG_1), W))

test_CR = Function(U_CR)
test_CR.vector().set_local(passage_ccG_to_CR * u)

file.write(test_CR)
file.write(local_project(grad(test_CR), W))
