# coding: utf-8
import scipy.sparse as sp
from dolfin import *

def matrice_passage_ccG_DG(nb_dof_cells_,nb_dof_ccG_):
    """Creates a csr companion matrix to get the cells values of a DEM vector."""
    return sp.eye(nb_dof_cells_, n = nb_dof_ccG_, format='csr')

def matrice_passage_ccG_CG(mesh_, nb_ddl_ccG_,num_vert_ccG,d_,dim):
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
