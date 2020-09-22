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
penalty = 2*mu
#Length of square domaine
L = 1.
a = 0.8 #Parameter of analytical solution

#Vectorial problem
def eps(v):
    return sym(grad(v))

def sigma(v):
    return lmbda * div(v) * Identity(2) + 2. * mu * eps(v)

def manufactured_solution_computation(nb_elt,L):
    mesh = RectangleMesh(Point(-L,-L),Point(L,L),nb_elt,nb_elt,"crossed")
    facets = MeshFunction("size_t", mesh, 1)
    ds = Measure('ds')(subdomain_data=facets)
    h_max = mesh.hmax() #Taille du maillage.
    d = mesh.geometric_dimension() #vectorial problem

    #Creating the DEM problem
    problem = DEMProblem(mesh, d, penalty)

    #Functions for computation
    solution_u_CR = Function(problem.CR,  name="CR")
    solution_u_DG_1 = Function(problem.DG_1,  name="DG 1")
    error = Function(problem.DG_1, name='Error')

    #reference solution
    x = SpatialCoordinate(mesh)
    u_ref = Expression(('0.5 * a * (pow(x[0],2) + pow(x[1],2))', '0.5 * a * (pow(x[0],2) + pow(x[1],2))'), a=a, degree=2)
    Du_ref = Expression(( ('a * x[0]', 'a * x[1]'), ('a * x[0]', 'a * x[1]')), a=a, degree=1)

    #Elastic bilinear form
    AA1 = elastic_bilinear_form(problem.mesh, problem.d, problem.DEM_to_CR, sigma, eps)

    #making the penalty term by hand... See if better...
    mat_pen = penalties(problem)

    #Assembling rigidity matrix
    A = AA1 + mat_pen

    #Imposition des conditions de Dirichlet Homogène
    A_not_D,B = problem.for_dirichlet(A)

    #Volume load for rhs
    volume_load = -a * (lmbda + 3.*mu) * as_vector([1., 1.])
    #Assembling the rhs
    L = assemble_volume_load(volume_load, problem)

    L_not_D,u_BC = schur_complement(L, u_ref, B, problem)

    #file_results = File("ccG_4_.pvd")
    res = open('res.txt', 'a')

    #solve
    print('Solve !')
    u_reduced = spsolve(A_not_D, L_not_D) #exact solve to have the right convergence order
    #u = mat_not_D.T * u_reduced + mat_D.T * u_BC
    u = complete_solution(u_reduced, u_BC, problem)

    solution_u_CR.vector().set_local(problem.DEM_to_CR * u)
    solution_u_DG_1.vector().set_local(problem.DEM_to_DG_1 * u)


    ##sorties paraview
    #file_results.write(solution_u_DG_1)

    #computation of errors
    u_ref_DEM = DEM_interpolation(u_ref, problem)
    u_ref_DG_1 = Function(problem.DG_1)
    u_ref_DG_1.vector().set_local(problem.DEM_to_DG_1 * u_ref_DEM)
    error.vector().set_local(problem.DEM_to_DG_1 * u - u_ref_DG_1.vector().get_local())
    #file_results.write(error)
    err_u_L2_DG_1 = errornorm(solution_u_DG_1, u_ref_DG_1, 'L2')
    print('err L2 DG 1: %.5e' % err_u_L2_DG_1)
    u_ref_CR = interpolate(u_ref, problem.CR)
    err_u_L2_CR = errornorm(solution_u_CR, u_ref_CR, 'L2')
    print('err L2 CR: %.5e' % err_u_L2_CR)

    err_Du_L2 = errornorm(solution_u_CR, u_ref_CR, 'H10')

    diff = u - u_ref_DEM
    err_energy = np.dot(diff, A * diff)

    err_energy = np.sqrt(0.5 * err_energy)
    print('error energy: %.5e' % err_energy)
    print('discrete grad : %.5e' % err_Du_L2)
    print('error elastic energy: %.5e' % (0.5 * np.dot(diff, AA1 * diff)))

    res.write('%.5e %i %.5e %.5e %.5e\n' % (h_max, problem.nb_dof_DEM, err_u_L2_DG_1, err_energy, err_Du_L2) )

    #close file
    res.close()

#Running computation
refining_mesh = [5, 10, 20, 40]
for nb_elt in refining_mesh:
    manufactured_solution_computation(nb_elt,L)
