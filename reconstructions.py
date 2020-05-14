# coding: utf-8
import scipy.sparse as sp

def matrice_passage_ccG_DG(nb_dof_cells_,nb_dof_ccG_):
    """Creates a csr companion matrix to get the cells values of a DEM vector."""
    return sp.eye(nb_dof_cells_, n = nb_dof_ccG_, format='csr')
