# coding: utf-8
from dolfin import *
from numpy import array

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

def dico_position_bary_face(mesh_, d_):
    """Creates a dictionary whose keys are the index of the facets of the mesh and the values the positions of the barycentre of the facets.""" 
    dim = mesh_.geometric_dimension()
    if d_ == dim:
        U_CR = VectorFunctionSpace(mesh_, 'CR', 1) #vectorial case
    elif d_ == 1:
        U_CR = FunctionSpace(mesh_, 'CR', 1) #scalar case
    elt = U_CR.element()
        
    result = dict([])
    for cell in cells(mesh_):
        local_num_facet = -1
        for f in facets(cell):
            local_num_facet += 1
            pos_dof_cell = elt.tabulate_dof_coordinates(cell)
            result[f.index()] = pos_dof_cell[local_num_facet]
    return result

def dico_position_vertex_bord(mesh_, face_num, d_):
    """truc"""
    dim = mesh_.geometric_dimension()
    if d_ == dim:
        U_DG = VectorFunctionSpace(mesh_, 'DG', 0) #vectorial case
    elif d_ == 1:
        U_DG = FunctionSpace(mesh_, 'DG', 0)
    nb_cell_dofs = U_DG.dofmap().global_dimension()
    
    vertex_associe_face = dict([])
    pos_ddl_vertex = dict([])
    num_ddl_vertex_ccG = dict([])
    compteur = nb_cell_dofs - 1 #va être à la fin le nbr de dof ccG
    for f in facets(mesh_):
        if len(face_num.get(f.index())) == 1: #Face sur le bord car n'a qu'un voisin
            vertex_associe_face[f.index()] = []
    for v in vertices(mesh_):
        test_bord = False #pour voir si vertex est sur le bord
        for f in facets(v):
            if len(face_num.get(f.index())) == 1: #Face sur le bord car n'a qu'un voisin
                vertex_associe_face[f.index()] = vertex_associe_face[f.index()] + [v.index()]
                test_bord = True
        if test_bord: #vertex est bien sur le bord. Donc devient un ddl de ccG
            if d_ == 2: #pour num du vertex dans vec ccG
                num_ddl_vertex_ccG[v.index()] = [compteur+1, compteur+2]
                compteur += 2
            elif d_ == 3:
                num_ddl_vertex_ccG[v.index()] = [compteur+1, compteur+2, compteur+3]
                compteur += 3
            elif d_ == 1:
                num_ddl_vertex_ccG[v.index()] = [compteur+1]
                compteur += 1
            if dim == 2: #pour position du vertex dans l'espace
                pos_ddl_vertex[v.index()] = array([v.x(0),v.x(1)])
            elif dim == 3:
                pos_ddl_vertex[v.index()] = array([v.x(0),v.x(1),v.x(2)])

    nb_DEM_dofs = nb_cell_dofs + d_ * len(pos_ddl_vertex)

    return vertex_associe_face,pos_ddl_vertex,num_ddl_vertex_ccG,nb_DEM_dofs
