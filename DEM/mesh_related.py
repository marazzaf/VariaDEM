# coding: utf-8
from dolfin import *
from numpy import array
import networkx as nx
from DEM.errors import *

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

def connectivity_graph(mesh_, d_): #Contains all necessary mesh information
    G = nx.Graph()

    #useful mesh entities
    dim = mesh_.topology().dim()
    if d_ == 1:
        U_CR = FunctionSpace(mesh_, 'CR', 1)
        U_DG = FunctionSpace(mesh_, 'DG', 0)
    elif d_ == 2 or d_ == 3:
        U_CR = VectorFunctionSpace(mesh_, 'CR', 1)
        U_DG = VectorFunctionSpace(mesh_, 'DG', 0)
    else:
        raise DimensionError
    nb_ddl_cells = U_DG.dofmap().global_dimension()
    dofmap_CR = U_CR.dofmap()
    nb_dof_CR = dofmap_CR.global_dimension()

    #useful auxiliary functions
    vol_c = CellVolume(mesh_) #Pour volume des particules voisines
    hF = FacetArea(mesh_)
    scalar_DG = FunctionSpace(mesh_, 'DG', 0) #for volumes
    f_DG = TestFunction(scalar_DG)
    scalar_CR = FunctionSpace(mesh_, 'CR', 1) #for surfaces
    f_CR = TestFunction(scalar_CR)

    #computation of volumes, surfaces and normals
    volumes = assemble(f_DG * dx).get_local()
    assert(volumes.min() > 0.)
    areas = assemble(f_CR('+') * (dS + ds)).get_local()
    assert(areas.min() > 0.)

    #importing cell dofs
    for c in cells(mesh_): #Importing cells
        aux = list(np.arange(count, count+d_))
        count += d_
        #computing volume and barycentre of the cell
        vert = []
        vert_ind = []
        for v in vertices(c):
            vert.append( np.array(v.point()[:])[:dim] )
            vert_ind.append(v.index())
        vol = volumes[c.index()]
        vert = np.array(vert)
        bary = vert.sum(axis=0) / vert.shape[0]
        #adding node to the graph
        G.add_node(c.index(), dof=aux, pos=bary, measure=vol, vertices=vert, bnd=False) #bnd=True if cell is on boundary of the domain

    #importing connectivity and facet dofs
    for f in facets(mesh_):
        aux_bis = [] #number of the cells
        for c in cells(f):
            aux_bis.append(c.index())
        num_global_ddl_facet = dofmap_CR.entity_dofs(mesh_, dim - 1, np.array([f.index()], dtype="uintp")) #number of the dofs in CR
        #computing quantites related to the facets
        vert = []
        for v in vertices(f):
            vert.append( np.array(v.point()[:])[:dim] )
        normal = normals[num_global_ddl_facet[0] // d_, :]
        area = areas[num_global_ddl_facet[0] // d_]
        #facet barycentre computation
        vert = np.array(vert)
        bary = vert.sum(axis=0) / vert.shape[0]

        #adding the facets to the graph
        if len(aux_bis) == 2: #add the link between two cell dofs     
            #adding edge
            G.add_edge(aux_bis[0],aux_bis[1], num=num_global_ddl_facet[0] // d_, dof_CR=num_global_ddl_facet, measure=area, barycentre=bary, vertices=vert, pen_factor=pen_factor[num_global_ddl_facet[0] // d_], breakable=True)

        elif len(aux_bis) == 1: #add the link between a cell dof and a boundary facet dof
            for c in cells(f): #only one cell contains the boundary facet
                bary_cell = G.node[c.index()]['pos']

            #checking if adding "dofs" for Dirichlet BC
            nb_dofs = len(dirichlet_dofs & set(num_global_ddl_facet))
            aux = list(np.arange(count, count+nb_dofs))
            count += nb_dofs
            components = sorted(list(dirichlet_dofs & set(num_global_ddl_facet)))
            components = np.array(components) % d_
            
            #number of the dof is total number of cells + num of the facet
            G.add_node(nb_ddl_cells // d_ + num_global_ddl_facet[0] // d_, pos=bary, dof=aux)
            G.add_edge(aux_bis[0], nb_ddl_cells // d_ + num_global_ddl_facet[0] // d_, num=num_global_ddl_facet[0] // d_, dof_CR=num_global_ddl_facet, measure=area, barycentre=bary, vertices=vert, pen_factor=pen_factor[num_global_ddl_facet[0] // d_])
            G.node[aux_bis[0]]['bnd'] = True #Cell is on the boundary of the domain
            #Ajouter des noeuds pour les vertex de bord ?

        else:
            raise ValueError('A facet cannot have more than 2 neighbours!')
                
    return G
