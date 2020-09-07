# coding: utf-8
import sys
sys.path.append('../')
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve,cg
from DEM.DEM import *
from DEM.reconstructions import compute_all_reconstruction_matrices,gradient_matrix
from DEM.miscellaneous import penalty_FV

# Form compiler options
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True
parameters["allow_extrapolation"] = True

# elastic parameters
E = Constant(70e3)
nu = Constant(0.3)
lmbda = E*nu/(1+nu)/(1-2*nu)
mu = E/2./(1+nu)
penalty = mu #1.e-4 * mu

Ll, l = 1., 1. #0.1   # sizes in rectangular mesh
a = 0.8 #rapport pour déplacement max...
mesh = Mesh("./mesh/square_1.xml")
facets = MeshFunction("size_t", mesh, 1)
ds = Measure('ds')(subdomain_data=facets)
h_max = mesh.hmax() #Taille du maillage.

# Mesh-related functions
h = CellVolume(mesh) #Pour volume des particules voisines
hF = FacetArea(mesh)
h_avg = (h('+') + h('-'))/ (2. * hF('+'))
nF = FacetNormal(mesh)

#Function spaces
U_DG = VectorFunctionSpace(mesh, 'DG', 0) #Pour délacement dans cellules
U_DG_1 = VectorFunctionSpace(mesh, 'DG', 1) #Pour reconstruction ccG
U_CR = VectorFunctionSpace(mesh, 'CR', 1) #Pour interpollation dans les faces
U_CG = VectorFunctionSpace(mesh, 'CG', 1) #Pour bc
W0 = FunctionSpace(mesh, 'DG', 0) 
W = TensorFunctionSpace(mesh, 'DG', 0)

#Functions for computation
solution_u_CR = Function(U_CR,  name="CR")
solution_u_DG_1 = Function(U_DG_1,  name="DG 1")
error = Function(U_DG_1, name='Error')
dim = solution_u_DG_1.geometric_dimension()  # space dimension
d = dim #pb vectoriel

#reference solution
x = SpatialCoordinate(mesh)
u_ref = Expression(('0.5 * a * (pow(x[0],2) + pow(x[1],2))', '0.5 * a * (pow(x[0],2) + pow(x[1],2))'), a=a, degree=2)
#Du_ref = Expression(( ('a * x[0]', 'a * (x[0] + x[1])'), ('a * (x[0] + x[1])', 'a * x[1]')), a=a, degree=1)
Du_ref = Expression(( ('a * x[0]', 'a * x[1]'), ('a * x[0]', 'a * x[1]')), a=a, degree=1)


n = FacetNormal(mesh)
volume_load = -a * (lmbda + 3.*mu) * as_vector([1., 1.])

def F_ext_2(v):
    return inner(volume_load, v) * dx

def eps(v):
    return sym(grad(v))

def sigma(v):
    return lmbda * div(v) * Identity(dim) + 2. * mu * eps(v)

##Cell-centre Galerkin reconstruction
#dofm = U_DG.dofmap()
#nb_ddl_cells = len(dofm.dofs())
#elt = U_DG.element()
#dofmap_CG = U_CG.dofmap()
#nb_ddl_CG = len(dofmap_CG.dofs())
#face_num = facet_neighborhood(mesh)
#dofmap_CR = U_CR.dofmap()
#nb_ddl_CR = len(dofmap_CR.dofs())
#elt_bis = U_CR.element()
#dico_pos_bary_faces = dico_position_bary_face(mesh,dofmap_CR,elt_bis)
#pos_bary_cells = position_ddl_cells(mesh,elt)
#vertex_associe_face,pos_ddl_vertex,num_ddl_vertex_ccG = dico_position_vertex_bord(mesh, face_num, nb_ddl_cells, d, dim)
#nb_ddl_ccG = nb_ddl_cells + d * len(pos_ddl_vertex)
#print('nb dof ccG: %i' % nb_ddl_ccG)
#convexe_num,convexe_coord = smallest_convexe_bary_coord_bis(face_num,pos_bary_cells,pos_ddl_vertex,dico_pos_bary_faces,dim,d)
#print('Convexe ok !')
#passage_ccG_to_CR = matrice_passage_ccG_CR(mesh, nb_ddl_ccG, convexe_num, convexe_coord, vertex_associe_face, num_ddl_vertex_ccG, d, dim)
#passage_ccG_to_CG = matrice_passage_ccG_CG(mesh, nb_ddl_ccG,num_ddl_vertex_ccG,d,dim)
#passage_ccG_to_DG = matrice_passage_ccG_DG(nb_ddl_cells,nb_ddl_ccG)

#DEM reconstruction
DEM_to_DG, DEM_to_CG, DEM_to_CR, DEM_to_DG_1, nb_dof_DEM = compute_all_reconstruction_matrices(mesh, d)
print('matrices passage ok !')

#Elastic bilinear form
AA1 = elastic_bilinear_form(mesh, d, DEM_to_CR, sigma, eps)

#gradient matrix
mat_grad = gradient_matrix(mesh, d)

#making the penalty term by hand... See if better...
mat_pen = penalty_FV(penalty, nb_ddl_ccG, mesh, face_num, d, dim, mat_grad, dico_pos_bary_faces, passage_ccG_to_CR)
#print(mat_pen.shape)*

# Define variational problem
u_CR = TrialFunction(U_CR)
v_CR = TestFunction(U_CR)
v_DG = TestFunction(U_DG)
u_CG = TrialFunction(U_CG)
v_CG = TestFunction(U_CG)
u_DG_1 = TrialFunction(U_DG_1)
v_DG_1 = TestFunction(U_DG_1)

#cell-boundary
a3 =  penalty * hF / h * inner(u_CG, v_DG_1) * ds
A3 = assemble(a3)
row,col,val = as_backend_type(A3).mat().getValuesCSR()
A32 = sp.csr_matrix((val, col, row))

a3 =  penalty * hF / h * inner(u_CG, v_CG) * ds
A3 = assemble(a3)
row,col,val = as_backend_type(A3).mat().getValuesCSR()
A33 = sp.csr_matrix((val, col, row))

a3 = penalty * hF / h * inner(v_DG_1('+'), u_DG_1('+')) * ds
A3 = assemble(a3)
row,col,val = as_backend_type(A3).mat().getValuesCSR()
A34 = sp.csr_matrix((val, col, row))
A34.resize((A32.shape[0],A32.shape[0]))

A_pen_bis = passage_ccG_to_CG.T * A33 * passage_ccG_to_CG -passage_ccG_to_CG.T * A32.T * passage_ccG_to_DG_1 - passage_ccG_to_DG_1.T * A32 * passage_ccG_to_CG + passage_ccG_to_DG_1.T * A34 * passage_ccG_to_DG_1

#Imposition des conditions de Dirichlet Homogène
a4 = inner(v_CG('+'),as_vector((1.,1.))) / hF * ds
A4 = assemble(a4)
A_BC = passage_ccG_to_CG.T * A4.get_local()

A = AA1 + mat_pen + A_pen_bis

#Prise en compte du chargement volumique à l'ancienne... Pour tester...
L2 = F_ext_2(v_DG) #forme linéaire pour chargement volumique
#L2 = F_ext_2(v_DG_1)
LL2 = assemble(L2)
b2 = LL2.get_local() #en format DG
bb2 = passage_ccG_to_DG.T * b2
#bb2 = passage_ccG_to_DG_1.T * b2

L = bb2

#Imposing strongly Dirichlet BC
mat_not_D,mat_D = schur(A_BC)
print('matrices passage Schur ok')
A_D = mat_D * A * mat_D.T
A_not_D = mat_not_D * A * mat_not_D.T
B = mat_not_D * A * mat_D.T

L_not_D = mat_not_D * L

#interpolation of Dirichlet BC...
F = interpolate(u_ref, U_CG).vector().get_local()
F = mat_D * passage_ccG_to_CG.T * F
L_not_D = L_not_D - B * F


file_results = File("ccG_4_.pvd")
res = open('res.txt', 'a')

#solve
print('Solve !')
#u_reduced,info = cg(A_not_D, L_not_D)
#assert(info == 0)
u_reduced = spsolve(A_not_D, L_not_D)
u = mat_not_D.T * u_reduced + mat_D.T * F

solution_u_CR.vector().set_local(passage_ccG_to_CR * u)
solution_u_DG_1.vector().set_local(passage_ccG_to_DG_1 * u)


#sorties paraview
file_results.write(solution_u_DG_1)

#computation of errors
#u_ref_DG_1 = interpolate(u_ref, U_DG_1)
u_ref_ccG = passage_ccG_to_DG.T * interpolate(u_ref, U_DG).vector().get_local() + passage_ccG_to_CG.T * interpolate(u_ref, U_CG).vector().get_local()
u_ref_DG_1 = Function(U_DG_1)
u_ref_DG_1.vector().set_local(passage_ccG_to_DG_1 * u_ref_ccG)
error.vector().set_local(passage_ccG_to_DG_1 * u - u_ref_DG_1.vector().get_local())
file_results.write(error)
err_u_L2_DG_1 = errornorm(solution_u_DG_1, u_ref_DG_1, 'L2')
print('err L2 DG 1: %.5e' % err_u_L2_DG_1)
u_ref_CR = interpolate(u_ref, U_CR)
err_u_L2_CR = errornorm(solution_u_CR, u_ref_CR, 'L2')
print('err L2 CR: %.5e' % err_u_L2_CR)

err_Du_L2 = errornorm(solution_u_CR, u_ref_CR, 'H10')

u_ref_ccG = passage_ccG_to_DG.T * interpolate(u_ref, U_DG).vector().get_local() + passage_ccG_to_CG.T * interpolate(u_ref, U_CG).vector().get_local()
diff = u - u_ref_ccG
#err_energy = np.dot(diff, AA1 * diff) + np.dot(diff, A_pen * diff) + np.dot(diff, A_pen_bis * diff)
err_energy = np.dot(diff, A * diff)

err_energy = np.sqrt(0.5 * err_energy)
print('error energy: %.5e' % err_energy)
print('discrete grad : %.5e' % err_Du_L2)
print('error elastic energy: %.5e' % (0.5 * np.dot(diff, AA1 * diff)))
#print('autres termes: %.5e' % (-0.5 * np.dot(diff, AA11 * diff)))
print('pen DG : %.5e' % (0.5 * np.dot(diff, mat_pen * diff)))
print('pen cell-vertex : %.5e' % (0.5 * np.dot( diff, A_pen_bis * diff)))

res.write('%.5e %i %.5e %.5e %.5e\n' % (h_max, nb_ddl_ccG, err_u_L2_DG_1, err_energy, err_Du_L2) )

#close file
res.close()


#plot_err_CR = abs((solution_u_CR - u_ref_CR)[0])
#fig = plot(plot_err_CR)
#plt.colorbar(fig, shrink=0.5, aspect=10)
#plt.title('err x')
#plt.show()
#
#plot_err_CR = abs((solution_u_CR - u_ref_CR)[1])
#fig = plot(plot_err_CR)
#plt.colorbar(fig, shrink=0.5, aspect=10)
#plt.title('err y')
#plt.show()
#
###plot errors
##plot_err_DG = abs((solution_u_DG - u_ref_DG)[0]) # - mean_u
##fig = plot(plot_err_DG)
##plt.colorbar(fig, shrink=0.5, aspect=10)
##plt.title('x')
##plt.show()
##
##plot_err_DG = abs((solution_u_DG - u_ref_DG)[1]) # - mean_u
##fig = plot(plot_err_DG)
##plt.colorbar(fig, shrink=0.5, aspect=10)
##plt.title('y')
##plt.show()
#
#plot_err_grad = abs((grad(solution_u_CR) - Du_reff)[0,0])
#fig = plot(plot_err_grad)
#plt.colorbar(fig, shrink=0.5, aspect=10)
#plt.title('err Du')
#plt.show()
#
#
##computed solution versus ref
#
#plot_CR = sqrt((solution_u_CR)**2) # - mean_u
#fig = plot(plot_CR)
#plt.colorbar(fig, shrink=0.5, aspect=10)
#plt.savefig('disp.pdf')
#plt.title('Norm of displacement ')
#plt.show()
#
#plot_CR = sqrt((u_ref_CR)**2) # - mean_u
#fig = plot(plot_CR)
#plt.colorbar(fig, shrink=0.5, aspect=10)
#plt.title('Reference norm of displacement')
#plt.savefig('ref_disp.pdf')
#plt.show()
#
##plot_CR = (solution_u_CR)[0] # - mean_u
##fig = plot(plot_CR)
##plt.colorbar(fig, shrink=0.5, aspect=10)
##plt.title('x')
##plt.show()
##
##fig = plot(u_ref_CR[0])
##plt.colorbar(fig, shrink=0.5, aspect=10)
##plt.title('ref x')
##plt.show()
##
##plot_CR = (solution_u_CR)[1] # - mean_u
##fig = plot(plot_CR)
##plt.colorbar(fig, shrink=0.5, aspect=10)
##plt.title('y')
##plt.show()
##
##fig = plot(u_ref_CR[1])
##plt.colorbar(fig, shrink=0.5, aspect=10)
##plt.title('ref y')
##plt.show()
##
##fig = plot(grad(solution_u_CR)[0,0])
##plt.colorbar(fig, shrink=0.5, aspect=10)
##plt.title('Du')
##plt.savefig('Du.pdf')
##plt.show()
##
##fig = plot(Du_reff[0,0])
##plt.colorbar(fig, shrink=0.5, aspect=10)
##plt.title('Du ref')
##plt.savefig('Du_ref.pdf')
##plt.show()
