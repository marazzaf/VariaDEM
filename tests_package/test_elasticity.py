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
from scipy.sparse.linalg import spsolve
from DEM.DEM import *
from DEM.miscellaneous import *
from dolfin import *
import pytest

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

@pytest.mark.parametrize("mesh", [Mesh('mesh_test_elasticity.xml')])
def test_elasticity(mesh):
    #mesh = Mesh('mesh_test_elasticity.xml')
    a = 0.8 #rapport pour déplacement max...
    facets = MeshFunction("size_t", mesh, 1)
    ds = Measure('ds')(subdomain_data=facets)
    d = mesh.geometric_dimension() #vectorial problem

    #Creating the DEM problem
    problem = DEMProblem(mesh, d, penalty)

    #Function spaces
    U_DG_1 = VectorFunctionSpace(mesh, 'DG', 1) #Pour reconstruction ccG
    U_CR = VectorFunctionSpace(mesh, 'CR', 1) #Pour interpollation dans les faces
    W = TensorFunctionSpace(mesh, 'DG', 0)

    #Functions for computation
    solution_u_CR = Function(U_CR,  name="CR")
    solution_u_DG_1 = Function(U_DG_1,  name="DG 1")
    solution_Du = Function(W, name="grad")
    
    #reference solution
    x = SpatialCoordinate(mesh)
    u_ref = Expression(('0.5 * a * (pow(x[0],2) + pow(x[1],2))', '0.5 * a * (pow(x[0],2) + pow(x[1],2))'), a=a, degree=2)
    Du_ref = Expression(( ('a * x[0]', 'a * x[1]'), ('a * x[0]', 'a * x[1]')), a=a, degree=1)

    def eps(v):
        return sym(grad(v))

    def sigma(v):
        return lmbda * div(v) * Identity(problem.dim) + 2. * mu * eps(v)

    #Elastic bilinear form
    AA1 = elastic_bilinear_form(problem, sigma, eps)

    #making the penalty term by hand... See if better...
    mat_pen = penalties(problem)

    #Assembling rigidity matrix
    A = AA1 + mat_pen

    #Imposing strongly Dirichlet boundary conditions
    A_not_D,B = problem.for_dirichlet(A)

    #Volume load for rhs
    volume_load = -a * (lmbda + 3.*mu) * as_vector([1., 1.])
    #Assembling the rhs
    L = assemble_volume_load(volume_load, problem)

    #Getting the rhs without the Drichlet components
    L_not_D,u_BC = schur_complement(L, u_ref, B, problem)


    #solve
    u_reduced = spsolve(A_not_D, L_not_D) #exact solve to have the right convergence order
    u = complete_solution(u_reduced, u_BC, problem)

    #Solutions
    solution_u_CR.vector().set_local(problem.DEM_to_CR * u)
    solution_u_DG_1.vector().set_local(problem.DEM_to_DG_1 * u)

    #computation of errors on solution
    u_ref_DEM = DEM_interpolation(u_ref, problem)
    u_ref_DG_1 = Function(U_DG_1)
    u_ref_DG_1.vector().set_local(problem.DEM_to_DG_1 * u_ref_DEM)
    value = u_ref_DG_1((0.,0.))
    value_ref = solution_u_DG_1((0.,0.))
    assert round(value[0],3) ==  round(value_ref[0],3) and round(value[1],3) ==  round(value_ref[1],3)

    #computation of errors on gradient
    solution_Du = local_project(grad(solution_u_CR),W)
    value_Du = solution_Du((0.,0.))
    value_ref_Du = Du_ref((0.,0.))
    assert round(value_Du[0],1) ==  round(value_ref_Du[0],1) and round(value_Du[1],1) ==  round(value_ref_Du[1],1) and round(value_Du[2],1) ==  round(value_ref_Du[2],1) and round(value_Du[3],1) ==  round(value_ref_Du[3],1)

    

