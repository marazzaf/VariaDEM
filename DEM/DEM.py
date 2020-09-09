# coding: utf-8

from dolfin import *
from scipy.sparse import csr_matrix
from DEM.errors import *
from DEM.reconstructions import compute_all_reconstruction_matrices,gradient_matrix
from DEM.mesh_related import facet_neighborhood,dico_position_bary_face

class DEMProblem:
    """ Class that will contain the basics of a DEM problem from the mesh and the dimension of the problem to reconstrucion matrices and gradient matrix."""
    def __init__(self, mesh, d, penalty):
        self.mesh = mesh
        self.dim = self.mesh.geometric_dimension()
        self.d = d
        self.penalty = penalty

        #Define the necessary functionnal spaces depending on d
        if self.d == 1:
            self.CR = FunctionSpace(self.mesh, 'CR', 1)
            self.W = VectorFunctionSpace(self.mesh, 'DG', 0)
            self.DG_0 = FunctionSpace(self.mesh, 'DG', 0)
            self.DG_1 = FunctionSpace(self.mesh, 'DG', 1)
            self.CG = FunctionSpace(self.mesh,'CG', 1)
        elif self.d == self.dim:
            self.CR = VectorFunctionSpace(self.mesh, 'CR', 1)
            self.W = TensorFunctionSpace(self.mesh, 'DG', 0)
            self.DG_0 = VectorFunctionSpace(self.mesh, 'DG', 0)
            self.DG_1 = VectorFunctionSpace(self.mesh, 'DG', 1)
            self.CG = VectorFunctionSpace(self.mesh,'CG', 1)
        else:
            raise ValueError('Problem is whether scalar or vectorial')

        #gradient
        self.mat_grad = gradient_matrix(self)

        #useful
        self.facet_num = facet_neighborhood(self.mesh)
        self.bary_facets = dico_position_bary_face(self.mesh, self.d)

        #DEM reconstructions
        self.DEM_to_DG, self.DEM_to_CG, self.DEM_to_CR, self.DEM_to_DG_1, self.nb_dof_DEM = compute_all_reconstruction_matrices(self)
        print('Reconstruction matrices ok!')


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

def penalty_FV(problem):
    #penalty_, nb_ddl_ccG_, mesh_, face_num, d_, dim_, mat_grad_, dico_pos_bary_facet, passage_ccG_CR_):
    """Creates the penalty matrix to stabilize the DEM."""

    if problem.d == problem.dim:
        tens_DG_0 = TensorFunctionSpace(problem.mesh, 'DG', 0)
    elif problem.d == 1:
        tens_DG_0 = VectorFunctionSpace(problem.mesh, 'DG', 0)
    else:
        raise ValueError
        
    dofmap_CR = problem.CR.dofmap()
    elt_DG = problem.DG.element()
    nb_ddl_CR = dofmap_CR.global_dimension()
    dofmap_tens_DG_0 = tens_DG_0.dofmap()
    nb_ddl_grad = dofmap_tens_DG_0.global_dimension()

    #assembling penalty factor
    vol = CellVolume(problem.mesh)
    hF = FacetArea(problem.mesh)
    testt = TestFunction(problem.CR)
    helpp = Function(problem.CR)
    helpp.vector().set_local(np.ones_like(helpp.vector().get_local()))
    a_aux = penalty_ * (2.*hF('+'))/ (vol('+') + vol('-')) * inner(helpp('+'), testt('+')) * dS
    mat = assemble(a_aux).get_local()

    #creating jump matrix
    mat_jump_1 = dok_matrix((nb_ddl_CR,problem.nb_dof_DEM))
    mat_jump_2 = dok_matrix((nb_ddl_CR,nb_ddl_grad))
    for f in facets(problem.mesh):
        if len(problem.facet_num.get(f.index())) == 2: #Face interne
            num_global_face = f.index()
            num_global_ddl = dofmap_CR.entity_dofs(mesh_, dim_ - 1, np.array([num_global_face], dtype="uintp"))
            coeff_pen = mat[num_global_ddl][0]
            pos_bary_facet = bary_facets[f.index()] #position barycentre of facet
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
            
    mat_jump = mat_jump_1.tocsr() + mat_jump_2.tocsr() * problem.mat_grad * problem.DEM_to_CR
    return mat_jump.T * mat_jump

def penalty_boundary(penalty_, nb_ddl_ccG_, mesh_, face_num, d_, num_ddl_vertex_ccG, mat_grad_, dico_pos_bary_facet, passage_ccG_CR_, pos_bary_cells):
    dim = mesh_.geometric_dimension()
    if d_ >= 2:
        U_CR = VectorFunctionSpace(mesh_, 'CR', 1)
        tens_DG_0 = TensorFunctionSpace(mesh_, 'DG', 0)
    else:
        U_CR = FunctionSpace(mesh_, 'CR', 1) #Pour interpollation dans les faces
        tens_DG_0 = VectorFunctionSpace(mesh_, 'DG', 0)
        
    dofmap_CR = U_CR.dofmap()
    nb_ddl_CR = dofmap_CR.global_dimension()
    dofmap_tens_DG_0 = tens_DG_0.dofmap()
    nb_ddl_grad = dofmap_tens_DG_0.global_dimension()

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

    mat_jump_bnd = mat_jump_1.tocsr() + mat_jump_2.tocsr() * mat_grad_ * passage_ccG_CR_
    return mat_jump_bnd.T * mat_jump_bnd
