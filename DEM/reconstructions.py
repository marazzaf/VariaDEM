# coding: utf-8
import scipy.sparse as sp
from dolfin import *
from numpy import array,arange,append
from scipy.spatial import ConvexHull, Delaunay, KDTree
from DEM.mesh_related import *
from DEM.errors import *

def DEM_to_DG_matrix(problem,nb_dof_ccG_):
    """Creates a csr companion matrix to get the cells values of a DEM vector."""
    nb_cell_dofs = problem.DG_0.dofmap().global_dimension()
    return sp.eye(nb_cell_dofs, n = nb_dof_ccG_, format='csr')

def DEM_to_CG_matrix(problem, num_vert_ccG, nb_dof_DEM):
    """Creates a csr companion matrix to get the boundary vertex values of a DEM vector."""
    nb_dof_CG = problem.CG.dofmap().global_dimension()
    matrice_resultat = sp.dok_matrix((nb_dof_CG,nb_dof_DEM)) #Empty matrix
    vert_to_dof = vertex_to_dof_map(problem.CG)

    #mettre des 1 à la bonne place dans la matrice...
    for (i,j),k in zip(num_vert_ccG.items(),range(nb_dof_DEM)): #On boucle sur le numéro des vertex
        matrice_resultat[vert_to_dof[i*problem.d], j[0]] = 1.
        if problem.d >= 2:
            matrice_resultat[vert_to_dof[i*problem.d]+1, j[1]] = 1.
        if problem.d == 3:
            matrice_resultat[vert_to_dof[i*problem.d]+2, j[2]] = 1.

    return matrice_resultat.tocsr()

def DEM_to_DG_1_matrix(problem, nb_dof_ccG_, DEM_to_CR):
    EDG_0 = problem.DG_0
    EDG_1 = problem.DG_1
    tens_DG_0 = problem.W
        
    dofmap_DG_0 = EDG_0.dofmap()
    dofmap_DG_1 = EDG_1.dofmap()
    dofmap_tens_DG_0 = tens_DG_0.dofmap()
    elt_0 = EDG_0.element()
    elt_1 = EDG_1.element()
    nb_total_dof_DG_1 = dofmap_DG_1.global_dimension()
    nb_dof_grad = dofmap_tens_DG_0.global_dimension()
    matrice_resultat_1 = sp.dok_matrix((nb_total_dof_DG_1,nb_dof_ccG_)) #Empty matrix
    matrice_resultat_2 = sp.dok_matrix((nb_total_dof_DG_1,nb_dof_grad)) #Empty matrix
    
    for c in cells(problem.mesh):
        index_cell = c.index()
        dof_position = dofmap_DG_1.cell_dofs(index_cell)

        #filling-in the matrix to have the constant cell value
        DG_0_dofs = dofmap_DG_0.cell_dofs(index_cell)
        for dof in dof_position:
            matrice_resultat_1[dof, DG_0_dofs[dof % problem.d]] = 1.

        #filling-in part to add the gradient term
        position_barycentre = elt_0.tabulate_dof_coordinates(c)[0]
        pos_dof_DG_1 = elt_1.tabulate_dof_coordinates(c)
        tens_dof_position = dofmap_tens_DG_0.cell_dofs(index_cell)
        for dof,pos in zip(dof_position,pos_dof_DG_1): #loop on quadrature points
            diff = pos - position_barycentre
            for i in range(problem.dim):
                matrice_resultat_2[dof, tens_dof_position[(dof % problem.d)*problem.d + i]] = diff[i]
        
    return matrice_resultat_1.tocsr() +  matrice_resultat_2.tocsr() * problem.mat_grad * DEM_to_CR

def gradient_matrix(problem):
    """Creates a matrix computing the cell-wise gradient from the facet values stored in a Crouzeix-raviart FE vector."""
    vol = CellVolume(problem.mesh)

    #variational form gradient
    u_CR = TrialFunction(problem.CR)
    Dv_DG = TestFunction(problem.W)
    a = inner(grad(u_CR), Dv_DG) / vol * dx
    A = assemble(a)
    row,col,val = as_backend_type(A).mat().getValuesCSR()
    return sp.csr_matrix((val, col, row))

def facet_interpolation(facet_num,pos_bary_cells,pos_vert,pos_bary_facets,dim_,d_, I=10):
    """Computes the reconstruction in the facets of the meh from the dofs of the DEM."""
    
    toutes_pos_ddl = [] #ordre : d'abord tous les ddl de cellules puis tous ceux des vertex au bord
    for i in pos_bary_cells.values():
        if dim_ == 2:
            toutes_pos_ddl.append(i)
        elif dim_ == 3:
            toutes_pos_ddl.append(i)
    for i in pos_vert.values():
        toutes_pos_ddl.append(i)
    toutes_pos_ddl = array(toutes_pos_ddl)
    #calcul des voisinages par arbre
    tree = KDTree(toutes_pos_ddl)
    #num de tous les ddl
    tous_num_ddl = arange(len(toutes_pos_ddl) * d_)

    #Fixing limits to the number of dofs used in the search for an interpolating simplex
    if dim_ == 3 and I < 25:
        I = 25 #comes from experience as default but can be changed
    if dim_ == 2 and I < 10:
        I = 10 #idem
    
    #calcul du convexe associé à chaque face
    res_num = dict([])
    res_pos = dict([])
    res_coord = dict([])
    for f,neigh in facet_num.items():
        if len(neigh) > 1: #Inner facet
            aux_num = []
            aux_pos = []
            x = pos_bary_facets.get(f) #position du barycentre de la face
            distance,pos_voisins = tree.query(x, I)
            
            #adding points to compute the convex hull
            data_convex_hull = [x]
            for k in range(I):
                data_convex_hull.append(tree.data[pos_voisins[k]])

            #computing the convex with the points in the list
            convex = ConvexHull(data_convex_hull,qhull_options='Qc QJ Pp')
            if 0 not in convex.vertices: #convex strictly contains the facet barycentre
                #Faire une triangulation de Delaunay des points du tree
                list_points = convex.points
                delau = Delaunay(list_points[1:]) #on retire le barycentre de la face de ces données. On ne le veut pas dans le Delaunay
                #Ensuite, utiliser le transform sur le Delaunay pour avoir les coords bary du bay de la face. Il reste seulement à trouver dans quel tétra est-ce que toutes les coords sont toutes positives !
                trans = delau.transform
                num_simplex = delau.find_simplex(x)
                coord_bary = delau.transform[num_simplex,:dim_].dot(x - delau.transform[num_simplex,dim_])
                coord_bary = append(coord_bary, 1. - coord_bary.sum())

                res_coord[f] = coord_bary
                for k in delau.simplices[num_simplex]:
                    num = pos_voisins[k] #not k-1 because the barycentre of the face x has been removed from the Delaunay triangulation
                    if d_ == 1:
                        aux_num.append([num]) #numéro dans vecteur ccG
                    elif d_ == 2:
                        aux_num.append([num * d_, num * d_ + 1])
                    elif d_ == 3:
                        aux_num.append([num * d_, num * d_ + 1, num * d_ + 2])

                #On associe le tetra à la face 
                res_num[f] = aux_num
                res_pos[f] = aux_pos
            
            else:
                raise ConvexError('Not possible to find a convex containing the barycenter of the facet.\n')
                                
    return res_num,res_coord

def DEM_to_CR_matrix(problem, nb_dof_ccG, facet_num, vertex_associe_face, num_ddl_vertex, pos_ddl_vertex):
    dofmap_CR = problem.CR.dofmap()
    nb_total_dof_CR = dofmap_CR.global_dimension()

    #computing the useful mesh quantities
    pos_bary_cells = position_cell_dofs(problem.mesh,problem.d)
    dico_pos_bary_faces = dico_position_bary_face(problem.mesh,problem.d)
    
    #Computing the facet reconstructions
    convex_num,convex_coord = facet_interpolation(facet_num,pos_bary_cells,pos_ddl_vertex,dico_pos_bary_faces,problem.dim,problem.d)

    #Storing the facet reconstructions in a matrix
    matrice_resultat = sp.dok_matrix((nb_total_dof_CR,nb_dof_ccG)) #Matrice vide.
    for f in facets(problem.mesh):
        num_global_face = f.index()
        num_global_ddl = dofmap_CR.entity_dofs(problem.mesh, problem.dim - 1, array([num_global_face], dtype="uintp"))
        convexe_f = convex_num.get(num_global_face)
        convexe_c = convex_coord.get(num_global_face)

        if convexe_f != None: #Face interne, on interpolle la valeur !
            for i,j in zip(convexe_f,convexe_c):
                matrice_resultat[num_global_ddl[0],i[0]] = j
                if problem.d >= 2:
                    matrice_resultat[num_global_ddl[1],i[1]] = j
                if problem.d == 3:
                    matrice_resultat[num_global_ddl[2],i[2]] = j
        else: #Face sur le bord, on interpolle la valeur avec les valeurs aux vertex
            pos_init = vertex_associe_face.get(num_global_face)
            v1 = num_ddl_vertex[pos_init[0]]
            v2 = num_ddl_vertex[pos_init[1]]
            if problem.dim == 2:
                matrice_resultat[num_global_ddl[0], v1[0]] = 0.5
                matrice_resultat[num_global_ddl[0], v2[0]] = 0.5
                if problem.d == 2: #pb vectoriel
                    matrice_resultat[num_global_ddl[1], v1[1]] = 0.5
                    matrice_resultat[num_global_ddl[1], v2[1]] = 0.5
            if problem.dim == 3:
                v3 = num_ddl_vertex[pos_init[2]]
                matrice_resultat[num_global_ddl[0], v1[0]] = 1./3.
                matrice_resultat[num_global_ddl[0], v2[0]] = 1./3.
                matrice_resultat[num_global_ddl[0], v3[0]] = 1./3.
                if problem.d >= 2: #deuxième ligne
                    matrice_resultat[num_global_ddl[1], v1[1]] = 1./3.
                    matrice_resultat[num_global_ddl[1], v2[1]] = 1./3.
                    matrice_resultat[num_global_ddl[1], v3[1]] = 1./3.
                if problem.d == 3: #troisième ligne
                    matrice_resultat[num_global_ddl[2], v1[2]] = 1./3.
                    matrice_resultat[num_global_ddl[2], v2[2]] = 1./3.
                    matrice_resultat[num_global_ddl[2], v3[2]] = 1./3.
        
    return matrice_resultat.tocsr()

def compute_all_reconstruction_matrices(problem):
    """Computes all the required reconstruction matrices."""

    #calling functions to construct the matrices
    DEM_to_DG = DEM_to_DG_matrix(problem, problem.nb_dof_DEM)
    DEM_to_CG = DEM_to_CG_matrix(problem, problem.num_ddl_vertex, problem.nb_dof_DEM)
    DEM_to_CR = DEM_to_CR_matrix(problem, problem.nb_dof_DEM, problem.facet_num, problem.vertex_associe_face, problem.num_ddl_vertex, problem.pos_ddl_vertex)
    DEM_to_DG_1 = DEM_to_DG_1_matrix(problem, problem.nb_dof_DEM, DEM_to_CR)

    return DEM_to_DG, DEM_to_CG, DEM_to_CR, DEM_to_DG_1
