# coding: utf-8
from fenics import *

def facet_neighborhood(mesh_):
    """Returns a dictionnary containing as key the index of the facets and as values the list of indices of the cells (or cell) containing the facet. """
    indices = dict([])

    for f in facets(mesh_):
        voisins_num = []
        for c in cells(f):
            voisins_num.append(c.index())

        indices[f.index()] = voisins_num
    return indices

def position_cell_dofs(mesh_,d_):
    """Returns a dictionnary having as key the index of a cell and as value the position of its dof."""
    dim_ = mesh_.geometric_dimension()
    
    if d_ == 1:
        U_DG = FunctionSpace(mesh_, 'DG', 0) #scalar case
    elif d_ == dim_:
        U_DG = VectorFunctionSpace(mesh_, 'DG', 0) #vectorial case
    elt = U_DG.element()
    
    cell_pos = dict([])
    for c in cells(mesh_):
        cell_pos[c.index()] = elt.tabulate_dof_coordinates(c)[0]
    return cell_pos
