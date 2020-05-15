# coding: utf-8
import scipy.sparse as sp
from dolfin import *
from numpy import array,arange,append
from scipy.spatial import ConvexHull, Delaunay, KDTree
from scipy.spatial.qhull import QhullError
import sys
from mesh_related import *

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

def facet_interpolation(face_n_num,pos_bary_cells,pos_vert,pos_bary_facets,dim_,d_): #pos_vert contient les positions des barycentres sur les faces au bord
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
    #calcul du convexe associé à chaque face
    res_num = dict([])
    res_pos = dict([])
    res_coord = dict([])
    for i,j in face_n_num.items():
        #print(i
        if len(j) > 1: #cad face pas au bord
            aux_num = []
            aux_pos = []
            x = pos_bary_facets.get(i) #position du barycentre de la face
            if dim_ == 2:
                nb_voisins = 10
            elif dim_ == 3:
                nb_voisins = 25 #20 avant de faire la torsion avec maillage fin...
            distance,pos_voisins = tree.query(x, nb_voisins)
            
            #adding points to compute the convex hull
            data_convex_hull = [x]
            for k in range(nb_voisins):
                data_convex_hull.append(tree.data[pos_voisins[k]])

            #computing the convex with the points in the list
            convexe = ConvexHull(data_convex_hull,qhull_options='Qc QJ Pp')
            if 0 not in convexe.vertices: #cad qu'on a un convexe qui contient strictement x
                #Faire une triangulation de Delaunay des points du tree
                list_points = convexe.points
                delau = Delaunay(list_points[1:]) #on retire le barycentre de la face de ces données. On ne le veut pas dans le Delaunay
                #Ensuite, utiliser le transform sur le Delaunay pour avoir les coords bary du bay de la face. Il reste seulement à trouver dans quel tétra est-ce que toutes les coords sont toutes positives !
                trans = delau.transform
                num_simplex = delau.find_simplex(x)
                #print(num_simplex
                coord_bary = delau.transform[num_simplex,:dim_].dot(x - delau.transform[num_simplex,dim_])
                coord_bary = append(coord_bary, 1. - coord_bary.sum())

                res_coord[i] = coord_bary
                for k in delau.simplices[num_simplex]:
                    num = pos_voisins[k] #not k-1 because the barycentre of the face x has been removed from the Delaunay triangulation
                    #print(num
                    if d_ == 1:
                        aux_num.append([num]) #numéro dans vecteur ccG
                    elif d_ == 2:
                        aux_num.append([num * d_, num * d_ + 1])
                    elif d_ == 3:
                        aux_num.append([num * d_, num * d_ + 1, num * d_ + 2])

                #On associe le tetra à la face 
                res_num[i] = aux_num
                res_pos[i] = aux_pos
            
            else:
                print('Not possible to find a convex containing the barycenter of the facet.\n')
                print('Ending computation !')
                sys.exit()
                                
    return res_num,res_coord

def matrice_passage_ccG_CR(mesh_, nb_ddl_ccG, facet_num, vertex_associe_face, num_ddl_vertex, d_, pos_ddl_vertex):
    dim = mesh_.geometric_dimension()
    if d_ == 1:
        ECR = FunctionSpace(mesh_, 'CR', 1)
        EDG = FunctionSpace(mesh_, 'DG', 0)
    elif d_ == dim:
        ECR = VectorFunctionSpace(mesh_, 'CR', 1)
        EDG = VectorFunctionSpace(mesh_, 'DG', 0)
    dofmap_CR = ECR.dofmap()
    nb_total_dof_CR = dofmap_CR.global_dimension()

    #computing the useful mesh quantities
    pos_bary_cells = position_cell_dofs(mesh_,d_)
    dico_pos_bary_faces = dico_position_bary_face(mesh_,d_)
    
    #Computing the facet reconstructions
    convex_num,convex_coord = facet_interpolation(facet_num,pos_bary_cells,pos_ddl_vertex,dico_pos_bary_faces,dim,d_)

    #Storing the facet reconstructions in a matrix
    matrice_resultat = sp.dok_matrix((nb_total_dof_CR,nb_ddl_ccG)) #Matrice vide.
    for f in facets(mesh_):
        num_global_face = f.index()
        num_global_ddl = dofmap_CR.entity_dofs(mesh_, dim - 1, array([num_global_face], dtype="uintp"))
        convexe_f = convex_num.get(num_global_face)
        convexe_c = convex_coord.get(num_global_face)

        if convexe_f != None: #Face interne, on interpolle la valeur !
            for i,j in zip(convexe_f,convexe_c):
                matrice_resultat[num_global_ddl[0],i[0]] = j
                if d_ >= 2:
                    matrice_resultat[num_global_ddl[1],i[1]] = j
                if d_ == 3:
                    matrice_resultat[num_global_ddl[2],i[2]] = j
        else: #Face sur le bord, on interpolle la valeur avec les valeurs aux vertex
            pos_init = vertex_associe_face.get(num_global_face)
            #print(pos_init
            v1 = num_ddl_vertex[pos_init[0]]
            v2 = num_ddl_vertex[pos_init[1]]
            if dim == 2:
                matrice_resultat[num_global_ddl[0], v1[0]] = 0.5
                matrice_resultat[num_global_ddl[0], v2[0]] = 0.5
                if d_ == 2: #pb vectoriel
                    matrice_resultat[num_global_ddl[1], v1[1]] = 0.5
                    matrice_resultat[num_global_ddl[1], v2[1]] = 0.5
            if dim == 3:
                v3 = num_ddl_vertex[pos_init[2]]
                matrice_resultat[num_global_ddl[0], v1[0]] = 1./3.
                matrice_resultat[num_global_ddl[0], v2[0]] = 1./3.
                matrice_resultat[num_global_ddl[0], v3[0]] = 1./3.
                if d_ >= 2: #deuxième ligne
                    matrice_resultat[num_global_ddl[1], v1[1]] = 1./3.
                    matrice_resultat[num_global_ddl[1], v2[1]] = 1./3.
                    matrice_resultat[num_global_ddl[1], v3[1]] = 1./3.
                if d_ == 3: #troisième ligne
                    matrice_resultat[num_global_ddl[2], v1[2]] = 1./3.
                    matrice_resultat[num_global_ddl[2], v2[2]] = 1./3.
                    matrice_resultat[num_global_ddl[2], v3[2]] = 1./3.
                
        assert(abs(sp.lil_matrix.sum(matrice_resultat[num_global_ddl[0],:]) - 1.) < 1.e-10)
        if d_ == 2:
            assert(abs(sp.lil_matrix.sum(matrice_resultat[num_global_ddl[1],:]) - 1.) < 1.e-10)
        #if d_ == 3:
        #    assert(abs(sp.lil_matrix.sum(matrice_resultat[num_global_ddl[2],:]) - 1.) < 1.e-10)
        
    return matrice_resultat.tocsr()
