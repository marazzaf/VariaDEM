# coding: utf-8
from dolfin import *
import numpy as np
from numpy.linalg import norm
from scipy.sparse import dok_matrix

def lumped_mass_matrix(mesh_, nb_ddl_ccG_, pos_ddl_vertex, num_ddl_vertex_ccG, d_, rho_):
    """Creates the ditributed diagonal mass matrix used to compute dynamic evolution with the DEM."""
    dim = mesh_.geometric_dimension()
    if d_ == dim: #vectorial problem
        U_DG = VectorFunctionSpace(mesh_, "DG", 0)
    elif d_ == 1: #scalar problem
        U_DG = FunctionSpace(mesh_, "DG", 0)

    #original mass matrix
    u_DG = TrialFunction(U_DG)
    v_DG = TestFunction(U_DG)
    M = rho_ * inner(u_DG,v_DG) * dx
    one = Constant(np.ones(d_)) #to get the diagonal
    res = assemble(action(M, one)).get_local()
    res = res.resize(nb_ddl_ccG_)
            
    for v in vertices(mesh_):
        if v.index() in pos_ddl_vertex: #vertex is indeed on the boundary
            total_mass = 0. #total mass to be allocated to the vertex
            for c in cells(v):     
                #Computation of edge barycentres only in 3d
                pos_bary_edge = []
                if dim == 3: #facet is enought is 2d
                    for e in edges(c):
                        test = False #Testing if edge of the cell contains v
                        pos = np.array([0.,0.,0.])
                        for vp in vertices(e):
                            pos += np.array([vp.x(0),vp.x(1),vp.x(2)]) / 2.
                            if vp.index() == v.index():
                                test = True
                        if test:
                            pos_bary_edge.append(pos)
                elif dim == 2:
                    for f in facets(c):
                        test = False #Testing if edge of the cell contains v
                        pos = np.array([0.,0.])
                        for vp in vertices(f):
                            pos += np.array([vp.x(0),vp.x(1)]) / 2.
                            if vp.index() == v.index():
                                test = True
                        if test:
                            pos_bary_edge.append(pos)
                            
                #computation of the volume given to the vertex
                tet = [pos_ddl_vertex.get(v.index())] + pos_bary_edge # + pos_bary_facets + [pos_bary_cell]
                vec1 = tet[1] - tet[0]
                vec2 = tet[2] - tet[0]
                if dim == 3: #rho_ is a volumic mass in that case
                    vec3 = tet[3] - tet[0]
                    mass = np.absolute(np.dot(np.cross(vec1,vec2),vec3)) / 6. * rho_
                elif dim == 2: #rho_ must be a surfacic mass in that case
                    mass = 0.5 * norm(np.cross(vec1,vec2)) * rho_
                total_mass += mass
                #remove the mass given to boundary vertices from cells in ccG mass vector
                res[c.index() * d_] += -mass
                if d_ >= 2:
                    res[c.index() * d_ + 1] += -mass
                if d_ == 3:
                    res[c.index() * d_ + 2] += -mass
            #allocates summed mass coming from cells containing vertex v to dofs of vertex v
            for i in num_ddl_vertex_ccG.get(v.index()):
                res[i] = total_mass
            #print('Total mass : %f' % total_mass
            
    return res

def local_project(v, V, u=None):
    """Element-wise projection using LocalSolver"""
    dv = TrialFunction(V)
    v_ = TestFunction(V)
    a_proj = inner(dv, v_)*dx
    b_proj = inner(v, v_)*dx
    solver = LocalSolver(a_proj, b_proj)
    solver.factorize()
    if u is None:
        u = Function(V)
        solver.solve_local_rhs(u)
        return u
    else:
        solver.solve_local_rhs(u)
        return

def DEM_interpolation(func, mesh_, d_, DEM_to_CG_, DEM_to_DG_):
    """Interpolates a function or expression to return a DEM vector containg the interpolation."""
    dim = mesh_.geometric_dimension()
    if d_ == dim:
        U_DG = VectorFunctionSpace(mesh_, 'DG', 0)
        U_CG = VectorFunctionSpace(mesh_, 'CG', 1)
    elif d_ == 1:
        U_DG = FunctionSpace(mesh_, 'DG', 0)
        U_CG = FunctionSpace(mesh_, 'CG', 1)

    return DEM_to_DG_.T * local_project(func, U_DG).vector().get_local() + DEM_to_CG_.T * local_project(func, U_CG).vector().get_local()

def Dirichlet_BC(form, DEM_to_CG):
    L = assemble(form)
    return DEM_to_CG.T * L.get_local()

def volume_load(form, DEM_to_DG):
    L = assemble(form)
    return DEM_to_DG.T * L


def schur(A_BC):
    nb_ddl_ccG = A_BC.shape[0]
    l = A_BC.nonzero()[0]
    aux = set(l) #contains number of Dirichlet dof
    nb_ddl_Dirichlet = len(aux)
    aux_bis = set(range(nb_ddl_ccG))
    aux_bis = aux_bis.difference(aux) #contains number of vertex non Dirichlet dof
    sorted(aux_bis) #sort the set

    #Get non Dirichlet values
    mat_not_D = dok_matrix((nb_ddl_ccG - nb_ddl_Dirichlet, nb_ddl_ccG))
    for (i,j) in zip(range(mat_not_D.shape[0]),aux_bis):
        mat_not_D[i,j] = 1.

    #Get Dirichlet boundary conditions
    mat_D = dok_matrix((nb_ddl_Dirichlet, nb_ddl_ccG))
    for (i,j) in zip(range(mat_D.shape[0]),aux):
        mat_D[i,j] = 1.
    return mat_not_D.tocsr(), mat_D.tocsr()
