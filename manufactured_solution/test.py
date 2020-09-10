# coding: utf-8
import sys
sys.path.append('../')
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve,cg
from DEM.DEM import *
from DEM.miscellaneous import *
from dolfin import *

# Form compiler options
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True
parameters["allow_extrapolation"] = True

# elastic parameters
E = Constant(70e3)
nu = Constant(0.3)
lmbda = E*nu/(1+nu)/(1-2*nu)
mu = E/2./(1+nu)
penalty = mu

Ll, l = 1., 1. #0.1   # sizes in rectangular mesh
a = 0.8 #rapport pour déplacement max...
mesh = Mesh("./mesh/square_1.xml")
facets = MeshFunction("size_t", mesh, 1)
ds = Measure('ds')(subdomain_data=facets)
h_max = mesh.hmax() #Taille du maillage.
d = mesh.geometric_dimension() #vectorial problem

#Creating the DEM problem
problem = DEMProblem(mesh, d, penalty)

#Function spaces
U_DG = VectorFunctionSpace(mesh, 'DG', 0) #Pour délacement dans cellules
U_DG_1 = VectorFunctionSpace(mesh, 'DG', 1) #Pour reconstruction ccG
U_CR = VectorFunctionSpace(mesh, 'CR', 1) #Pour interpollation dans les faces
U_CG = VectorFunctionSpace(mesh, 'CG', 1) #Pour bc

#Functions for computation
solution_u_CR = Function(U_CR,  name="CR")
solution_u_DG_1 = Function(U_DG_1,  name="DG 1")
error = Function(U_DG_1, name='Error')
dim = solution_u_DG_1.geometric_dimension()  # space dimension

#reference solution
x = SpatialCoordinate(mesh)
u_ref = Expression(('0.5 * a * (pow(x[0],2) + pow(x[1],2))', '0.5 * a * (pow(x[0],2) + pow(x[1],2))'), a=a, degree=2)
Du_ref = Expression(( ('a * x[0]', 'a * x[1]'), ('a * x[0]', 'a * x[1]')), a=a, degree=1)

def eps(v):
    return sym(grad(v))

def sigma(v):
    return lmbda * div(v) * Identity(dim) + 2. * mu * eps(v)

#Elastic bilinear form
AA1 = elastic_bilinear_form(problem.mesh, problem.d, problem.DEM_to_CR, sigma, eps)

#making the penalty term by hand... See if better...
mat_pen = penalties(problem)

#Assembling rigidity matrix
A = AA1 + mat_pen

## Define variational problem
#u_CR = TrialFunction(U_CR)
#v_CR = TestFunction(U_CR)
#v_DG = TestFunction(U_DG)
#u_CG = TrialFunction(U_CG)
#v_CG = TestFunction(U_CG)
#u_DG_1 = TrialFunction(U_DG_1)
#v_DG_1 = TestFunction(U_DG_1)

#Imposition des conditions de Dirichlet Homogène
A_not_D,B = problem.for_dirichlet(A)

#Volume load for rhs
volume_load = -a * (lmbda + 3.*mu) * as_vector([1., 1.])
#Assembling the rhs
L = assemble_volume_load(volume_load, problem)

##Schur complement to solve with Dirichlet BC
#L_not_D = mat_not_D * L
#
##interpolation of Dirichlet BC...
#u_BC = interpolate(u_ref, U_CG).vector().get_local()
#u_BC = mat_D * passage_ccG_to_CG.T * u_BC
#L_not_D = L_not_D - B * u_BC

L_not_D,u_BC = schur_complement(L, u_ref, B, problem)

file_results = File("ccG_4_.pvd")
res = open('res.txt', 'a')

#solve
print('Solve !')
u_reduced = spsolve(A_not_D, L_not_D) #exact solve to have the right convergence order
#u = mat_not_D.T * u_reduced + mat_D.T * u_BC
u = complete_solution(u_reduced, u_BC, problem)

solution_u_CR.vector().set_local(problem.DEM_to_CR * u)
solution_u_DG_1.vector().set_local(problem.DEM_to_DG_1 * u)


#sorties paraview
file_results.write(solution_u_DG_1)

#computation of errors
u_ref_DEM = DEM_interpolation(u_ref, problem)
u_ref_DG_1 = Function(U_DG_1)
u_ref_DG_1.vector().set_local(problem.DEM_to_DG_1 * u_ref_DEM)
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
