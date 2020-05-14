# coding: utf-8
from fenics import *
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

def penalty_FV(penalty_, nb_ddl_ccG_, mesh_, face_num, d_, dim_, mat_grad_, dico_pos_bary_facet, passage_ccG_CR_):
    if d_ >= 2:
        U_CR = VectorFunctionSpace(mesh_, 'CR', 1)
        U_DG = VectorFunctionSpace(mesh_, 'DG', 0)
        tens_DG_0 = TensorFunctionSpace(mesh_, 'DG', 0)
    else:
        U_CR = FunctionSpace(mesh_, 'CR', 1)
        U_DG = FunctionSpace(mesh_, 'DG', 0)
        tens_DG_0 = VectorFunctionSpace(mesh_, 'DG', 0)
        
    dofmap_CR = U_CR.dofmap()
    elt_CR = U_CR.element()
    elt_DG = U_DG.element()
    nb_ddl_CR = len(dofmap_CR.dofs())
    dofmap_tens_DG_0 = tens_DG_0.dofmap()
    nb_ddl_grad = len(dofmap_tens_DG_0.dofs())

    #assembling penalty factor
    vol = CellVolume(mesh_)
    hF = FacetArea(mesh_)
    testt = TestFunction(U_CR)
    helpp = Function(U_CR)
    helpp.vector().set_local(np.ones_like(helpp.vector().get_local()))
    a_aux = penalty_ * (2.*hF('+'))/ (vol('+') + vol('-')) * inner(helpp('+'), testt('+')) * dS
    mat = assemble(a_aux).get_local()

    #creating jump matrix
    mat_jump_1 = dok_matrix((nb_ddl_CR,nb_ddl_ccG_))
    mat_jump_2 = dok_matrix((nb_ddl_CR,nb_ddl_grad))
    for f in facets(mesh_):
        if len(face_num.get(f.index())) == 2: #Face interne
            num_global_face = f.index()
            num_global_ddl = dofmap_CR.entity_dofs(mesh_, dim_ - 1, np.array([num_global_face], dtype="uintp"))
            coeff_pen = mat[num_global_ddl][0]
            pos_bary_facet = dico_pos_bary_facet[f.index()] #position barycentre of facet
            #print(pos_bary_facet)
            for c,num_cell,sign in zip(cells(f),face_num.get(num_global_face),[1., -1.]):
                #filling-in the DG 0 part of the jump
                mat_jump_1[num_global_ddl[0]:num_global_ddl[-1]+1,d_ * num_cell : (num_cell+1) * d_] = sign*np.sqrt(coeff_pen)*np.eye(d_)
                
                #filling-in the DG 1 part of the jump...
                pos_bary_cell = elt_DG.tabulate_dof_coordinates(c)[0]
                diff = pos_bary_facet - pos_bary_cell
                pen_diff = np.sqrt(coeff_pen)*diff
                tens_dof_position = dofmap_tens_DG_0.cell_dofs(num_cell)
                for num,dof_CR in enumerate(num_global_ddl):
                    for i in range(dim_):
                        mat_jump_2[dof_CR,tens_dof_position[(num % d_)*d_ + i]] = sign*pen_diff[i]
            
    mat_jump = mat_jump_1 + mat_jump_2 * mat_grad_ * passage_ccG_CR_
    return (mat_jump.T * mat_jump).tocsr()

def penalty_boundary(penalty_, nb_ddl_ccG_, mesh_, face_num, d_, dim_, num_ddl_vertex_ccG, mat_grad_, dico_pos_bary_facet, passage_ccG_CR_, pos_bary_cells):
    if d_ >= 2:
        U_CR = VectorFunctionSpace(mesh_, 'CR', 1)
        tens_DG_0 = TensorFunctionSpace(mesh_, 'DG', 0)
    else:
        U_CR = FunctionSpace(mesh_, 'CR', 1) #Pour interpollation dans les faces
        tens_DG_0 = VectorFunctionSpace(mesh_, 'DG', 0)
        
    dofmap_CR = U_CR.dofmap()
    nb_ddl_CR = len(dofmap_CR.dofs())
    dofmap_tens_DG_0 = tens_DG_0.dofmap()
    nb_ddl_grad = len(dofmap_tens_DG_0.dofs())

    #assembling penalty factor
    vol = CellVolume(mesh_)
    hF = FacetArea(mesh_)
    testt = TestFunction(U_CR)
    helpp = Function(U_CR)
    helpp.vector().set_local(np.ones_like(helpp.vector().get_local()))
    a_aux = penalty_ * hF / vol * inner(helpp, testt) * ds
    mat = assemble(a_aux).get_local()

    #creating jump matrix
    mat_jump_1 = dok_matrix((nb_ddl_CR,nb_ddl_ccG_))
    mat_jump_2 = dok_matrix((nb_ddl_CR,nb_ddl_grad))
    for f in facets(mesh_):
        if len(face_num.get(f.index())) == 1: #Face sur le bord car n'a qu'un voisin
            num_global_face = f.index()
            num_global_ddl = dofmap_CR.entity_dofs(mesh_, dim_ - 1, np.array([num_global_face], dtype="uintp"))
            coeff_pen = mat[num_global_ddl][0]
            pos_bary_facet = dico_pos_bary_facet[f.index()] #position barycentre of facet

            #cell part
            #filling-in the DG 0 part of the jump
            num_cell = face_num.get(f.index())[0]
            mat_jump_1[num_global_ddl[0]:num_global_ddl[-1]+1,d_ * num_cell : (num_cell+1) * d_] = np.sqrt(coeff_pen)*np.eye(d_)

            #filling-in the DG 1 part of the jump
            pos_bary_cell = pos_bary_cells.get(num_cell)
            diff = pos_bary_facet - pos_bary_cell
            pen_diff = np.sqrt(coeff_pen)*diff
            tens_dof_position = dofmap_tens_DG_0.cell_dofs(num_cell)
            for num,dof_CR in enumerate(num_global_ddl):
                for i in range(dim_):
                    mat_jump_2[dof_CR,tens_dof_position[(num % d_)*d_ + i]] = pen_diff[i]

            #boundary facet part
            for v in vertices(f):
                dof_vert = num_ddl_vertex_ccG.get(v.index())
                mat_jump_1[num_global_ddl[0]:num_global_ddl[-1]+1,dof_vert[0]:dof_vert[-1]+1] = -np.sqrt(coeff_pen)*np.eye(d_) / d_

    mat_jump_bnd = mat_jump_1 + mat_jump_2 * mat_grad_ * passage_ccG_CR_
    return (mat_jump_bnd.T * mat_jump_bnd).tocsr()
