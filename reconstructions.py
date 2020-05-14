# coding: utf-8
import scipy.sparse as sp
from dolfin import *

def DEM_to_DG_matrix(nb_dof_cells_,nb_dof_ccG_):
    """Creates a csr companion matrix to get the cells values of a DEM vector."""
    return sp.eye(nb_dof_cells_, n = nb_dof_ccG_, format='csr')

def DEM_to_CG_matrix(mesh_, nb_ddl_ccG_,num_vert_ccG,d_,dim):
    """Creates a csr companion matrix to get the boundary vertex values of a DEM vector."""
    if d_ == 1:
        U_CG = FunctionSpace(mesh_,'CG', 1)
    elif d_ == dim:
        U_CG = VectorFunctionSpace(mesh_,'CG', 1)
    nb_ddl_CG = U_CG.dofmap().global_dimension()
    matrice_resultat = sp.dok_matrix((nb_ddl_CG,nb_ddl_ccG_)) #Matrice vide.
    vert_to_dof = vertex_to_dof_map(U_CG)

    #mettre des 1 à la bonne place dans la matrice...
    for (i,j),k in zip(num_vert_ccG.items(),range(nb_ddl_ccG_)): #On boucle sur le numéro des vertex
        matrice_resultat[vert_to_dof[i*d_], j[0]] = 1.
        if d_ >= 2:
            matrice_resultat[vert_to_dof[i*d_]+1, j[1]] = 1.
        if d_ == 3:
            matrice_resultat[vert_to_dof[i*d_]+2, j[2]] = 1.

    return matrice_resultat.tocsr()

def DEM_to_DG_1_matrix(mesh_, nb_ddl_ccG_, d_, dim_, passage_ccG_CR):
    mat_grad = gradient_matrix(mesh_,d_)
    if d_ == 1:
        EDG_0 = FunctionSpace(mesh_, 'DG', 0)
        EDG_1 = FunctionSpace(mesh_, 'DG', 1)
        tens_DG_0 = VectorFunctionSpace(mesh_, 'DG', 0)
    else:
        EDG_0 = VectorFunctionSpace(mesh_, 'DG', 0)
        EDG_1 = VectorFunctionSpace(mesh_, 'DG', 1)
        tens_DG_0 = TensorFunctionSpace(mesh_, 'DG', 0)
    dofmap_DG_0 = EDG_0.dofmap()
    dofmap_DG_1 = EDG_1.dofmap()
    dofmap_tens_DG_0 = tens_DG_0.dofmap()
    elt_0 = EDG_0.element()
    elt_1 = EDG_1.element()
    nb_total_dof_DG_1 = len(dofmap_DG_1.dofs())
    nb_ddl_grad = len(dofmap_tens_DG_0.dofs())
    matrice_resultat_1 = sp.dok_matrix((nb_total_dof_DG_1,nb_ddl_ccG_)) #Empty matrix
    matrice_resultat_2 = sp.dok_matrix((nb_total_dof_DG_1,nb_ddl_grad)) #Empty matrix
    
    for c in cells(mesh_):
        index_cell = c.index()
        dof_position = dofmap_DG_1.cell_dofs(index_cell)

        #filling-in the matrix to have the constant cell value
        DG_0_dofs = dofmap_DG_0.cell_dofs(index_cell)
        for dof in dof_position:
            #print(dof,dof % d_)
            matrice_resultat_1[dof, DG_0_dofs[dof % d_]] = 1.

        #filling-in part to add the gradient term
        position_barycentre = elt_0.tabulate_dof_coordinates(c)[0]
        pos_dof_DG_1 = elt_1.tabulate_dof_coordinates(c)
        tens_dof_position = dofmap_tens_DG_0.cell_dofs(index_cell)
        for dof,pos in zip(dof_position,pos_dof_DG_1): #loop on quadrature points
            diff = pos - position_barycentre
            for i in range(dim_):
                matrice_resultat_2[dof, tens_dof_position[(dof % d_)*d_ + i]] = diff[i]
        
    return matrice_resultat_1 +  matrice_resultat_2 * mat_grad * passage_ccG_CR

def gradient_matrix(mesh_, d_):
    """Creates a matrix computing the cell-wise gradient from the facet values stored in a Crouzeix-raviart FE vector."""
    dim = mesh_.geometric_dimension()
    if d_ == 1:
        W = VectorFunctionSpace(mesh_, 'DG', 0)
        U_CR = FunctionSpace(mesh_, 'CR', 1)
    elif d_ == dim:
        W = TensorFunctionSpace(mesh_, 'DG', 0)
        U_CR = VectorFunctionSpace(mesh_, 'CR', 1)

    vol = CellVolume(mesh_)

    #variational form gradient
    u_CR = TrialFunction(U_CR)
    Dv_DG = TestFunction(W)
    a = inner(grad(u_CR), Dv_DG) / vol * dx
    A = assemble(a)
    row,col,val = as_backend_type(A).mat().getValuesCSR()
    return sp.csr_matrix((val, col, row))
