# coding: utf-8

from dolfin import *
from scipy.sparse import csr_matrix
from DEM.errors import *

def elastic_bilinear_form(mesh_, d_, DEM_to_CR_matrix, sigma=grad, eps=grad):
    dim = mesh_.geometric_dimension()
    if d_ == 1:
        U_CR = FunctionSpace(mesh_, 'CR', 1)
    elif d_ == dim:
        U_CR = VectorFunctionSpace(mesh_, 'CR', 1)
    else:
        raise ValueError('Problem is either scalar or vectorial (in 2d and 3d)')

    u_CR = TrialFunction(U_CR)
    v_CR = TestFunction(U_CR)

    #Mettre eps et sigma en arguments de la fonction ?
    if d_ == 1:
        a1 = eps(u_CR) * sigma(v_CR) * dx
    elif d_ == dim:
        a1 = inner(eps(u_CR), sigma(v_CR)) * dx
    else:
        raise ValueError('Problem is either scalar or vectorial (in 2d and 3d)')
    
    A1 = assemble(a1)
    row,col,val = as_backend_type(A1).mat().getValuesCSR()
    A1 = csr_matrix((val, col, row))
    return DEM_to_CR_matrix.T * A1 * DEM_to_CR_matrix

def penalty_FV(penalty_, nb_ddl_ccG_, mesh_, face_num, d_, dim_, mat_grad_, dico_pos_bary_facet, passage_ccG_CR_):
    """Creates the penalty matrix to stabilize the DEM."""
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

def penalty_boundary(penalty_, nb_ddl_ccG_, mesh_, face_num, d_, num_ddl_vertex_ccG, mat_grad_, dico_pos_bary_facet, passage_ccG_CR_, pos_bary_cells):
    dim = mesh_.geometric_dimension()
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
            num_global_ddl = dofmap_CR.entity_dofs(mesh_, dim - 1, np.array([num_global_face], dtype="uintp"))
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
                for i in range(dim):
                    mat_jump_2[dof_CR,tens_dof_position[(num % d_)*d_ + i]] = pen_diff[i]

            #boundary facet part
            for v in vertices(f):
                dof_vert = num_ddl_vertex_ccG.get(v.index())
                mat_jump_1[num_global_ddl[0]:num_global_ddl[-1]+1,dof_vert[0]:dof_vert[-1]+1] = -np.sqrt(coeff_pen)*np.eye(d_) / d_

    mat_jump_bnd = mat_jump_1 + mat_jump_2 * mat_grad_ * passage_ccG_CR_
    return (mat_jump_bnd.T * mat_jump_bnd).tocsr()

def penalty_term(nb_ddl_ccG_, mesh_, d_, dim_, mat_grad_, passage_ccG_CR_, G_, nb_ddl_CR_, nz_vec_BC):
    if d_ >= 2:
        U_DG = VectorFunctionSpace(mesh_, 'DG', 0)
        tens_DG_0 = TensorFunctionSpace(mesh_, 'DG', 0)
    else:
        U_DG = FunctionSpace(mesh_, 'DG', 0)
        tens_DG_0 = VectorFunctionSpace(mesh_, 'DG', 0)
        
    nb_ddl_cells = U_DG.dofmap().global_dimension()
    dofmap_tens_DG_0 = tens_DG_0.dofmap()
    nb_ddl_grad = dofmap_tens_DG_0.global_dimension()

    #creating jump matrix
    mat_jump_1 = sp.dok_matrix((nb_ddl_CR_,nb_ddl_ccG_))
    mat_jump_2 = sp.dok_matrix((nb_ddl_CR_,nb_ddl_grad))
    for (x,y) in G_.edges():
        num_global_ddl = G_[x][y]['dof_CR']
        coeff_pen = G_[x][y]['pen_factor']
        pos_bary_facet = G_[x][y]['barycentre'] #position barycentre of facet
        if abs(x) < nb_ddl_cells // d_ and abs(y) < nb_ddl_cells // d_: #Inner facet
            c1,c2 = x,y
            #filling-in the DG 0 part of the jump
            mat_jump_1[num_global_ddl[0]:num_global_ddl[-1]+1,d_ * c1 : (c1+1) * d_] = np.sqrt(coeff_pen)*np.eye(d_)
            mat_jump_1[num_global_ddl[0]:num_global_ddl[-1]+1,d_ * c2 : (c2+1) * d_] = -np.sqrt(coeff_pen)*np.eye(d_)

            for num_cell,sign in zip([c1,c2],[1., -1.]):
                #filling-in the DG 1 part of the jump...
                pos_bary_cell = G_.node[num_cell]['pos']
                diff = pos_bary_facet - pos_bary_cell
                pen_diff = np.sqrt(coeff_pen)*diff
                tens_dof_position = dofmap_tens_DG_0.cell_dofs(num_cell)
                for num,dof_CR in enumerate(num_global_ddl):
                    for i in range(dim_):
                        mat_jump_2[dof_CR,tens_dof_position[num*d_ + i]] = sign*pen_diff[i]

        #Penalty between facet reconstruction and cell value
        elif abs(x) >= nb_ddl_cells // d_ or abs(y) >= nb_ddl_cells // d_: #Outer facet
        
            if x >= 0 and y >= 0:
                num_cell = min(x,y)
                other = max(x,y)
            elif x <= 0 or y <= 0:
                num_cell = max(x,y)
                other = min(x,y)
        
            #selection dofs with Dirichlet BC
            coeff_pen = np.sqrt(coeff_pen)
            
            #cell part
            #filling-in the DG 0 part of the jump
            for pos,num_CR in enumerate(num_global_ddl):
                mat_jump_1[num_CR,d_ * num_cell + pos] = coeff_pen
        
            #filling-in the DG 1 part of the jump
            pos_bary_cell = G_.node[num_cell]['pos']
            diff = pos_bary_facet - pos_bary_cell
            pen_diff = coeff_pen*diff
            tens_dof_position = dofmap_tens_DG_0.cell_dofs(num_cell)
            for num,dof_CR in enumerate(num_global_ddl):
                if dof_CR in nz_vec_BC:
                    for i in range(dim_):
                        mat_jump_2[dof_CR,tens_dof_position[num*d_ + i]] = pen_diff[i]

            #boundary facet part
            dof = G_.node[other]['dof']
            count = 0
            for pos,num_CR in enumerate(num_global_ddl):
                mat_jump_1[num_CR,dof[pos]] = -coeff_pen
                        

    mat_jump_1 = mat_jump_1.tocsr()
    mat_jump_2 = mat_jump_2.tocsr()
    mat_jump = mat_jump_1 + mat_jump_2 * mat_grad_ * passage_ccG_CR_
    return mat_jump.T * mat_jump
