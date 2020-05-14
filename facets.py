# coding: utf-8
from fenics import *
import numpy as np
from numpy.linalg import norm
import scipy.sparse as sp
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from scipy.spatial import KDTree
from scipy.spatial.qhull import QhullError
import matplotlib.pyplot as plt
import sys

def simplex_facet(facet_n_num,facet_n_pos,facet_nn_num,facet_nn_pos, h_): #On va aller chercher nn pour le triangle
    result_num = dict([])
    result_pos = dict([])
    #for (f, voisins_num),(g,voisins_pos) in zip(face_num.items(),face_pos.items()):
    for f,voisins_num in facet_n_num.items():
        #print('face : %i' % f
        if(len(voisins_num) == 1): #cad face sur le bord
            result_num[f] = []
            result_pos[f] = []
        else: #face interne
            voi1 = voisins_num[0]
            voi2 = voisins_num[1]
            for i,j in zip(facet_nn_num.get(f),facet_nn_pos.get(f)):
                if i != voi1 and i != voi2:
                    pos_test = facet_n_pos.get(f) + [j]
                    if not( tri_applati(pos_test, h_) ):
                        result_num[f] = [voi1, voi2, i]
                        result_pos[f] = pos_test
                        break
            assert( len(result_num.get(f)) == 3)
    return result_num,result_pos

def vertex_cellule(mesh_, face_num, nb_ddl_cells, d_): #Donne la position des vertex sur le bord pour chaque cellule
    res_num = dict([])
    res_pos = dict([])
    pos_ccG = dict([])
    pos = int(nb_ddl_cells)
    for c in cells(mesh_):
        aux_num = []
        aux_pos = []
        for f in facets(c):
            if len(face_num.get(f.index())) == 1:
                for v in vertices(f):
                    aux_num.append(v.index())
                    aux_pos.append([v.x(0),v.x(1),v.x(2)])
                    if d_ == 1:
                        num_ddl_ccG = [pos]
                        pos += 1
                    elif d_ == 2:
                        num_ddl_ccG = [pos*d_, pos*d_+1]
                        pos += 2
                    elif d_ == 3:
                        num_ddl_ccG = [pos*d_, pos*d_+1, pos*d_+2]
                        pos += 3
                    pos_ccG[v.index()] = num_ddl_ccG
        if aux_num != []:
            res_num[c.index()] = aux_num
            res_pos[c.index()] = aux_pos
    return res_num,res_pos,pos_ccG
                

def add_boundary_dof(mesh_,facet_n_num,facet_nn_num,facet_nn_pos,dico_pos_bary_faces,dico_faces_bord, d_): #va renvoyer des numéros de ddl dans vecteur ccG. Sera plus simple...
    aux_num = dict([])
    res_num = dict([])
    res_pos = dict([])
    for i,j in facet_nn_num.items():
        new_num = []
        for k in j: #On va changer les numéros de cellules par les positions des ddl dans vecteur ccG
            num_ddl_ccG = [k] #k est le num de la cellule. On met les positions des ddl de la face dans vecteur ccG.
            if d_ == 2:
                num_ddl_ccG = [k*d_, k*d_+1]
            elif d_ == 3:
                num_ddl_ccG = [k*d_, k*d_+1, k*d_+2]
            new_num.append(num_ddl_ccG)
        aux_num[i] = new_num #On réassocie les ddl voisins de ceux de la face de num i
    #On va ajouter les ddl des faces sur le bord
    for f in facets(mesh_):
        num_faces = []
        pos_faces = []
        for c in cells(f):
            for g in facets(c): #facets liées à la facet considérée
                if g != f and len(facet_n_num.get(g.index())) == 1: #cad que la facet liée est sur le bord
                    num_faces.append(dico_faces_bord.get(g.index()))
                    pos_faces.append(dico_pos_bary_faces.get(g.index()))
        deja_present_num = aux_num.get(f.index())
        deja_present_pos = facet_nn_pos.get(f.index())
        if num_faces == []:
            res_num[f.index()] = deja_present_num
        else:
            res_num[f.index()] = deja_present_num + num_faces
        if pos_faces == []:
            res_pos[f.index()] = deja_present_pos
        else:
            res_pos[f.index()] = deja_present_pos + pos_faces
    return res_num,res_pos #On va prendre en compte en plus ces données danns le calcul du convexe entourant chaque face

def barycentric_coord_2d(points, x): #On évalue dans tetra de points les coordonnées de x
    mat = np.array([np.append(points[0], 1.), np.append(points[1], 1.), np.append(points[2], 1.)]).T
    if np.abs(np.linalg.det(mat)) < 1.e-10:
        print('attention utilisation de la condition foireuse !')
        return np.array([0.,0.5,0.5]) #Garder cette condition ???
    else:
        b = np.array([x[0], x[1], 1.]).T
        res = np.linalg.solve(mat,b)
        assert(np.abs(np.sum(res) - 1.0) < 1.e-14) #valeur ici bien choisie ? Prendre valeur relativ
        return res

def dico_position_bary_face(mesh_, d_):
    """Creates a dictionary whose keys are the index of the facets of the mesh and the values the positions of the barycentre of the facets.""" 
    dim = mesh_.geometric_dimension()
    if dim == d_:
        U_CR = VectorFunctionSpace(mesh_, 'CR', 1) #vectorial case
    elif dim == 1:
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

def smallest_convexe_bary_coord(face_n_num,pos_bary_cells,pos_vert,pos_bary_facets,dim,d_,h_,h_min,Tetra=True): #pos_vert contient les positions des barycentres sur les faces au bord
    stencil = 0. #distance max utilisée pour reconstruire une valeur
    toutes_pos_ddl = [] #ordre : d'abord tous les ddl de cellules puis tous ceux des vertex au bord
    for i in pos_bary_cells.values():
        if dim == 2:
            toutes_pos_ddl.append(i)#[i[0],i[1]])
        elif dim == 3:
            toutes_pos_ddl.append(i) #[i[0],i[1],i[2]])
    for i in pos_vert.values():
        toutes_pos_ddl.append(i)
    toutes_pos_ddl = np.array(toutes_pos_ddl)
    #calcul des voisinages par arbre
    tree = KDTree(toutes_pos_ddl)
    #num de tous les ddl
    tous_num_ddl = np.arange(len(toutes_pos_ddl) * d_)
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
            if dim == 2:
                nb_voisins = 10
            elif dim == 3:
                nb_voisins = 25 #20 avant de faire la torsion avec maillage fin...
            test = True
            #while not(test_coord) and nb_voisins < 100:
            distance,pos_voisins = tree.query(x, nb_voisins)
            if distance.max() > stencil:
                stencil = distance.max()
            compteur = dim+1
            data_convex_hull = [x]
            for k in range(compteur):
                data_convex_hull.append(tree.data[pos_voisins[k]])
            try:
                convexe = ConvexHull(data_convex_hull,incremental=True,qhull_options='Qc QJ Pp') # p C-1.e-4') #Tv #QJ
            except QhullError:
                data_convex_hull.append(tree.data[pos_voisins[compteur]])
                compteur += 1
                convexe = ConvexHull(data_convex_hull,incremental=True,qhull_options='Qc QJ Pp')
            while test:
                try:
                    convexe.add_points([tree.data[pos_voisins[compteur]]])
                    compteur += 1
                except QhullError:
                    convexe.add_points([tree.data[pos_voisins[compteur]]])
                    compteur += 1
                #print(convexe.points
                #print(convexe.vertices
                if not( 0 in convexe.vertices ): #cad qu'on a un convexe qui contient strictement x
                    if not(Tetra): #On revoie tout le convexe qu'on vient de calculer
                        for k in convexe.vertices:
                            aux_pos.append(convexe.points[k]) #coord spatiale des points du convexe
                        if dim == 3:
                            coord_bary = generalised_barycentric_coord_3d(aux_pos,x,np.zeros_like(aux_pos),h_,0) #ça en 2d ou autre ?
                        elif dim == 2:
                            coord_bary = generalised_barycentric_coord(aux_pos,x, h_) #,np.zeros_like(aux_pos),h_,0) #ça en 2d ou autre ?
                        if np.abs(coord_bary.sum() - 1.) < 1.e-15 and coord_bary.min() > 0. and coord_bary.size == len(aux_pos):
                            test = False #Coord bary sont ok. Fin de la recherche
                            res_coord[i] = coord_bary
                            for k in convexe.vertices: #refaire ça ici ???
                                #print(k
                                num = pos_voisins[k-1] #k-1 because the barycentre of the face x is not in the KDtree
                                #print(num
                                if d_ == 1:
                                    aux_num.append([num]) #numéro dans vecteur ccG
                                elif d_ == 2:
                                    aux_num.append([num * d_, num * d_ + 1])
                                elif d_ == 3:
                                    aux_num.append([num * d_, num * d_ + 1, num * d_ + 2])
                        else:
                            aux_num = []
                            aux_pos = []
                    else:  #On renvoie un tétra qui contient strictement le bary de la face
                        #Faire une triangulation de Delaunay des points du tree
                        list_points = convexe.points
                        delau = Delaunay(list_points[1:]) #on retire le barycentre de la face de ces données. On ne le veut pas dans le Delaunay
                        #Ensuite, utiliser le transform sur le Delaunay pour avoir les coords bary du bay de la face. Il reste seulement à trouver dans quel tétra est-ce que toutes les coords sont toutes positives !
                        trans = delau.transform
                        num_simplex = delau.find_simplex(x)
                        #print(num_simplex
                        coord_bary = delau.transform[num_simplex,:dim].dot(x - delau.transform[num_simplex,dim])
                        coord_bary = np.append(coord_bary, 1. - coord_bary.sum())
                        try :
                            assert( np.abs(coord_bary.sum() - 1.) < 1.e-15 and coord_bary.min() > 1.e-14 and coord_bary.size == dim+1 ) #cad coord bary sont bien ok pour interpollation
                        except AssertionError:
                            print('Pb avec coord bary dans une face !')
                            print('%.15e' % (np.abs(coord_bary.sum() - 1.)))
                            print(coord_bary)
                        test = False #Coord bary sont ok. Fin de la recherche
                        res_coord[i] = coord_bary
                        for k in delau.simplices[num_simplex]:
                            #print(k
                            num = pos_voisins[k] #not k-1 because the barycentre of the face x has been removed from the Delaunay triangulation
                            #print(num
                            if d_ == 1:
                                aux_num.append([num]) #numéro dans vecteur ccG
                            elif d_ == 2:
                                aux_num.append([num * d_, num * d_ + 1])
                            elif d_ == 3:
                                aux_num.append([num * d_, num * d_ + 1, num * d_ + 2])

            convexe.close() #On enlève les ressources pour l'ajout incrémental
            #On associe le tetra à la face 
            res_num[i] = aux_num
            #print(aux_num
            res_pos[i] = aux_pos
            #print(aux_pos
            #res_coord[i] = aux_coord.append(iter_coord_bary_3d(aux_pos,x,h_))

    return res_num,res_pos,res_coord,stencil #res_pos utile ?    

def convexe_generalised_bary_coord(face_n_num,face_nn_num,face_nn_pos,pos_bary_facets,dim,h_):
    res_num = dict([])
    res_pos = dict([])
    res_coord = dict([])
    for (f,num),(g,pos) in zip(face_nn_num.items(),face_nn_pos.items()):
        print(g)
        #print(pos
        if len(face_n_num.get(g)) == 1: #face au bord
            res_num[g] = []
            res_pos[g] = []
            res_coord[g] = []
        else: #face pas au bord
            assert(len(pos) >= dim + 1)
            hull = ConvexHull(pos,qhull_options='Po')
            vert = hull.vertices #position des vertex qui forment le convexe dans la liste des pos
            convexe_num = []
            convexe_pos = []
            for i in vert:
                convexe_num.append(num[i])
                convexe_pos.append(pos[i])
            #print(convexe_num
            #print(convexe_pos
            res_num[g] = convexe_num
            res_pos[g] = convexe_pos
            if dim == 2:
                res_coord[g] = generalised_barycentric_coord(convexe_pos, pos_bary_facets[g]) #les coords bary généralisées du centre de la face de num f dans le convexe qu'on vient de calculer
            elif dim == 3:
                #res_coord[g] = generalised_barycentric_coord_3d(convexe_pos, pos_bary_facets[g])
                res_coord[g] = iter_coord_bary_3d(convexe_pos, pos_bary_facets[g],h_)
    return res_num,res_pos, res_coord

#Dictionnaire avec numéro de la face donne sa position dans le vecteur ccG !
def dic_position_facets_bord(mesh_, facet_n_num,nb_cellules, d_):
    resultat = dict([])
    compteur = nb_cellules-1
    for num_face,voisins in facet_n_num.items():
        if len(voisins) == 1: #Face sur le bord car n'a qu'un voisin
            if d_ == 2:
                resultat[num_face] = [compteur+1, compteur+2]
                compteur += 2
            elif d_ == 3:
                resultat[num_face] = [compteur+1, compteur+2, compteur+3]
                compteur += 3
            elif d_ == 1:
                resultat[num_face] = [compteur+1]
                compteur += 1
    return resultat

def dico_position_vertex_bord(mesh_, face_num, nb_ddl_cellules, d_, dim):
    vertex_associe_face = dict([])
    pos_ddl_vertex = dict([])
    num_ddl_vertex_ccG = dict([])
    compteur = nb_ddl_cellules-1 #-1 ? #va être à la fin le nbr de dof ccG
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
                pos_ddl_vertex[v.index()] = np.array([v.x(0),v.x(1)])
            elif dim == 3:
                pos_ddl_vertex[v.index()] = np.array([v.x(0),v.x(1),v.x(2)])

    return vertex_associe_face,pos_ddl_vertex,num_ddl_vertex_ccG

def mass_vertex_dofs(mesh_, nb_ddl_ccG_, pos_ddl_vertex, num_ddl_vertex_ccG, d_, dim, rho_):
    res = np.zeros(nb_ddl_ccG_)
    if d_ == dim: #vectorial problem
        U_DG = VectorFunctionSpace(mesh_, "DG", 0)
    elif d_ == 1: #scalar problem
        U_DG = FunctionSpace(mesh_, "DG", 0)
    elt_DG = U_DG.element()
    nb_ddl_cells = len(U_DG.dofmap().dofs())
            
    for v in vertices(mesh_):
        if v.index() in pos_ddl_vertex: #vertex is indeed on the boundary
            total_mass = 0. #total mass to be allocated to the vertex
            for c in cells(v):
                #mass = 0. #to be computed      
                #Computation of edge barycentres only in 3d
                pos_bary_edge = []
                if dim == 3: #facet is enought is 2d
                    for e in edges(c):
                        test = False #Testing if edge of the cell contains v
                        pos = np.array([0.,0.,0.])
                        for vp in vertices(e):
                            pos += np.array([vp.x(0),vp.x(1),vp.x(2)]) / 2.
                            if vp.index() == v.index():
                                test = True
                        if test:
                            pos_bary_edge.append(pos)
                elif dim == 2:
                    for f in facets(c):
                        test = False #Testing if edge of the cell contains v
                        pos = np.array([0.,0.])
                        for vp in vertices(f):
                            pos += np.array([vp.x(0),vp.x(1)]) / 2.
                            if vp.index() == v.index():
                                test = True
                        if test:
                            pos_bary_edge.append(pos)
                            
                #computation of the volume given to the vertex
                tet = [pos_ddl_vertex.get(v.index())] + pos_bary_edge # + pos_bary_facets + [pos_bary_cell]
                vec1 = tet[1] - tet[0]
                vec2 = tet[2] - tet[0]
                if dim == 3: #rho_ is a volumic mass in that case
                    vec3 = tet[3] - tet[0]
                    mass = np.absolute(np.dot(np.cross(vec1,vec2),vec3)) / 6. * rho_
                elif dim == 2: #rho_ must be a surfacic mass in that case
                    mass = 0.5 * norm(np.cross(vec1,vec2)) * rho_
                total_mass += mass
                #remove the mass given to boundary vertices from cells in ccG mass vector
                res[c.index() * d_] += -mass
                if d_ >= 2:
                    res[c.index() * d_ + 1] += -mass
                if d_ == 3:
                    res[c.index() * d_ + 2] += -mass
            #allocates summed mass coming from cells containing vertex v to dofs of vertex v
            for i in num_ddl_vertex_ccG.get(v.index()):
                res[i] = total_mass
            #print('Total mass : %f' % total_mass
            
    return res

def matrice_passage_ccG_CR(mesh_, nb_ddl_ccG, conv_num, conv_coord, vertex_associe_face, num_ddl_vertex, d_, dim):
    if d_ == 1:
        ECR = FunctionSpace(mesh_, 'CR', 1)
        EDG = FunctionSpace(mesh_, 'DG', 0)
    else:
        ECR = VectorFunctionSpace(mesh_, 'CR', 1)
        EDG = VectorFunctionSpace(mesh_, 'DG', 0)
    dofmap_CR = ECR.dofmap()
    nb_total_dof_CR = len(dofmap_CR.dofs())
    matrice_resultat = sp.dok_matrix((nb_total_dof_CR,nb_ddl_ccG)) #Matrice vide.
    for f in facets(mesh_):
        #print(dofmap_CR.cell_dofs(cell.index())
        #num_local_face += 1
        num_global_face = f.index()
        #num_global_ddl = local_to_global_dof(num_local_face, dofmap_CR, num_global_cellule, d_)
        num_global_ddl = dofmap_CR.entity_dofs(mesh_, dim - 1, np.array([num_global_face], dtype="uintp"))
        #print(num_global_ddl
        #print(num_global_face
        convexe_f = conv_num.get(num_global_face)
        convexe_c = conv_coord.get(num_global_face)
        #print(tri_p
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
                
        #assert(abs(sp.lil_matrix.sum(matrice_resultat[num_global_ddl[0],:]) - 1.) < 1.e-10)
        #if d_ == 2:
        #    assert(abs(sp.lil_matrix.sum(matrice_resultat[num_global_ddl[1],:]) - 1.) < 1.e-10)
        #if d_ == 3:
        #    assert(abs(sp.lil_matrix.sum(matrice_resultat[num_global_ddl[2],:]) - 1.) < 1.e-10)
        
    return matrice_resultat.tocsr()


def penalty_boundary_old(penalty_, nb_ddl_ccG_, mesh_, face_num, d_, dim, num_ddl_vertex_ccG):
    if d_ >= 2:
        U_CR = VectorFunctionSpace(mesh_, 'CR', 1)
    else:
        U_CR = FunctionSpace(mesh_, 'CR', 1) #Pour interpollation dans les faces
    dofmap_ = U_CR.dofmap()
    nb_ddl_CR = len(dofmap_.dofs())

    #assembling penalty factor
    vol = CellVolume(mesh_)
    hF = FacetArea(mesh_)
    testt = TestFunction(U_CR)
    helpp = Function(U_CR)
    helpp.vector().set_local(np.ones_like(helpp.vector().get_local()))
    a_aux = penalty_ * hF / vol * inner(helpp, testt) * ds
    mat = assemble(a_aux).get_local()

    #creating jump matrix
    mat_jump_bnd = sp.dok_matrix((nb_ddl_CR,nb_ddl_ccG_))
    for v in vertices(mesh_):
        test_bord = False #pour voir si vertex est sur le bord
        for f in facets(v):
            if len(face_num.get(f.index())) == 1: #Face sur le bord car n'a qu'un voisin
                num_global_face = f.index()
                num_global_ddl = dofmap_.entity_dofs(mesh_, dim - 1, np.array([num_global_face], dtype="uintp"))
                coeff_pen = mat[num_global_ddl][0]
                #diagg = sp.diags(np.sqrt(coeff_pen))
                num_cell = face_num.get(f.index())[0]
                mat_jump_bnd[num_global_ddl[0]:num_global_ddl[-1]+1,d_ * num_cell : (num_cell+1) * d_] = np.sqrt(coeff_pen)*np.eye(d_) #diagg #np.eye(d_)
                dof_vert = num_ddl_vertex_ccG.get(v.index())
                mat_jump_bnd[num_global_ddl[0]:num_global_ddl[-1]+1,dof_vert[0]:dof_vert[-1]+1] = -np.sqrt(coeff_pen)*np.eye(d_) / d_ #-diagg / d_ #-np.eye(d_) / d_

    return (mat_jump_bnd.T * mat_jump_bnd).tocsr()

def smallest_convexe_bary_coord_bis(face_n_num,pos_bary_cells,pos_vert,pos_bary_facets,dim_,d_): #pos_vert contient les positions des barycentres sur les faces au bord
    toutes_pos_ddl = [] #ordre : d'abord tous les ddl de cellules puis tous ceux des vertex au bord
    for i in pos_bary_cells.values():
        if dim_ == 2:
            toutes_pos_ddl.append(i)
        elif dim_ == 3:
            toutes_pos_ddl.append(i)
    for i in pos_vert.values():
        toutes_pos_ddl.append(i)
    toutes_pos_ddl = np.array(toutes_pos_ddl)
    #calcul des voisinages par arbre
    tree = KDTree(toutes_pos_ddl)
    #num de tous les ddl
    tous_num_ddl = np.arange(len(toutes_pos_ddl) * d_)
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
                coord_bary = np.append(coord_bary, 1. - coord_bary.sum())

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

def local_project(v, V, u=None):
    dv = TrialFunction(V)
    v_ = TestFunction(V)
    a_proj = inner(dv, v_)*dx
    b_proj = inner(v, v_)*dx
    solver = LocalSolver(a_proj, b_proj)
    solver.factorize()
    if u is None:
        u = Function(V)
        solver.solve_local_rhs(u)
        return u
    else:
        solver.solve_local_rhs(u)
        return

def penalty_FV(penalty_, nb_ddl_ccG_, mesh_, face_num, d_, dim_, mat_grad_, dico_pos_bary_facet, passage_ccG_CR_):
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
    mat_jump_1 = sp.dok_matrix((nb_ddl_CR,nb_ddl_ccG_))
    mat_jump_2 = sp.dok_matrix((nb_ddl_CR,nb_ddl_grad))
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
                #mat_jump_1[num_global_ddl[0]:num_global_ddl[-1]+1,d_ * num_cell : (num_cell+1) * d_] = sign*np.eye(d_)
                
                #filling-in the DG 1 part of the jump...
                pos_bary_cell = elt_DG.tabulate_dof_coordinates(c)[0]
                #print(pos_bary_cell)
                diff = pos_bary_facet - pos_bary_cell
                #print(diff)
                pen_diff = np.sqrt(coeff_pen)*diff
                #pen_diff = diff
                tens_dof_position = dofmap_tens_DG_0.cell_dofs(num_cell)
                for num,dof_CR in enumerate(num_global_ddl):
                    for i in range(dim_):
                        #print((num % d_)*d_ + i)
                        mat_jump_2[dof_CR,tens_dof_position[(num % d_)*d_ + i]] = sign*pen_diff[i]
                #print(mat_jump_2)
                #print('\n')
            #sys.exit()
            
    mat_jump = mat_jump_1 + mat_jump_2 * mat_grad_ * passage_ccG_CR_
    return (mat_jump.T * mat_jump).tocsr()

def penalty_boundary(penalty_, nb_ddl_ccG_, mesh_, face_num, d_, dim_, num_ddl_vertex_ccG, mat_grad_, dico_pos_bary_facet, passage_ccG_CR_, pos_bary_cells):
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
    mat_jump_1 = sp.dok_matrix((nb_ddl_CR,nb_ddl_ccG_))
    mat_jump_2 = sp.dok_matrix((nb_ddl_CR,nb_ddl_grad))
    for f in facets(mesh_):
        if len(face_num.get(f.index())) == 1: #Face sur le bord car n'a qu'un voisin
            num_global_face = f.index()
            num_global_ddl = dofmap_CR.entity_dofs(mesh_, dim_ - 1, np.array([num_global_face], dtype="uintp"))
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
                for i in range(dim_):
                    mat_jump_2[dof_CR,tens_dof_position[(num % d_)*d_ + i]] = pen_diff[i]

            #boundary facet part
            for v in vertices(f):
                dof_vert = num_ddl_vertex_ccG.get(v.index())
                mat_jump_1[num_global_ddl[0]:num_global_ddl[-1]+1,dof_vert[0]:dof_vert[-1]+1] = -np.sqrt(coeff_pen)*np.eye(d_) / d_

    mat_jump_bnd = mat_jump_1 + mat_jump_2 * mat_grad_ * passage_ccG_CR_
    return (mat_jump_bnd.T * mat_jump_bnd).tocsr()
    #return mat_jump_bnd.tocsr() #just for test

def comparison_bary_coord(face_n_num,pos_bary_cells,pos_vert,pos_bary_facets,dim,d_,h_,I=6): #pos_vert contient les positions des barycentres sur les faces au bord
    toutes_pos_ddl = [] #ordre : d'abord tous les ddl de cellules puis tous ceux des vertex au bord
    for i in pos_bary_cells.values():
        toutes_pos_ddl.append(i)
    for i in pos_vert.values():
        toutes_pos_ddl.append(i)
    toutes_pos_ddl = np.array(toutes_pos_ddl)
    tree = KDTree(toutes_pos_ddl)
    #calcul du convexe associé à chaque face
    res_num = dict([])
    res_coord = dict([])

    #outputs
    nb_inner_facets = 0
    extrapolating_facets = 0
    for i,j in face_n_num.items():
        if len(j) > 1: #cad facette pas au bord
            nb_inner_facets += 1
            aux_num = []
            x = pos_bary_facets.get(i) #position du barycentre de la face
            distance,pos_voisins = tree.query(x, I) #valeur fixée ici (pour etre large !)
            data_convex_hull = [x] #[] #[x]
            for k in range(I):
                data_convex_hull.append(tree.data[pos_voisins[k]])

            #computation of convex hull
            convex = ConvexHull(data_convex_hull,qhull_options='Qc QJ Pp')    
            assert(convex.volume > 1.e-5 * h_**3) #otherwise convex hull is degenerate !

            #convex is not degenerate. Computing interpolation or extrapolation
            if 0 not in convex.vertices: #interpolation
                #computing the barycentric coordinates
                list_points = convex.points
                delau = Delaunay(list_points[1:])
                trans = delau.transform
                num_simplex = delau.find_simplex(x)
                coord_bary = delau.transform[num_simplex,:dim].dot(x - delau.transform[num_simplex,dim])
                coord_bary = np.append(coord_bary, 1. - coord_bary.sum())
                res_coord[i] = coord_bary
                for k in delau.simplices[num_simplex]:
                    num = pos_voisins[k] #not k-1 because the barycentre of the face x has been removed from the Delaunay triangulation
                    if d_ == 1:
                        aux_num.append([num]) #numéro dans vecteur ccG
                    elif d_ == 2:
                        aux_num.append([num * d_, num * d_ + 1])
                    elif d_ == 3:
                        aux_num.append([num * d_, num * d_ + 1, num * d_ + 2])
            else: #extrapolation with closest smallest non-degenerate simplex
                extrapolating_facets += 1
                #taking the last point in the hull and the dim first !
                mat = []
                mat.append(convex.points[0] - convex.points[-1])
                if d_ == 3: #to have a second vector not colinear with the first
                    for k in np.arange(len(convex.points))+1:
                        test = convex.points[k] - convex.points[-1]
                        test /= np.linalg.norm(test)
                        if np.linalg.norm(np.cross(test, mat[0] / np.linalg.norm(mat[0]))) > 1.e-5:
                            mat.append(convex.points[k] - convex.points[-1])
                            break

                    
                for k in np.arange(len(convex.points))+1: #to have the last points to have an invertible matrix
                    test = convex.points[k] - convex.points[-1]
                    test /= np.linalg.norm(test)
                    if np.absolute(np.linalg.det(np.array([mat[0]/np.linalg.norm(mat[0]),mat[1]/np.linalg.norm(mat[1])] + [test]))) >  1.e-5:
                            mat.append(convex.points[k] - convex.points[-1])
                            break
                        

                mat = np.array(mat)
                assert(mat.shape == (dim,dim))
                assert(np.absolute(np.linalg.det(mat)) > 1.e-10) #no degenerate barycentric coordinates !
                rhs = np.array(x - convex.points[-1])
                coord_bary = np.linalg.solve(mat, rhs)
                coord_bary = np.append(coord_bary, 1. - coord_bary.sum())
                res_coord[i] = coord_bary
                for k in range(dim):
                    num = pos_voisins[k] #not k-1 because the barycentre of the face x has been removed from the Delaunay triangulation
                    if d_ == 1:
                        aux_num.append([num]) #numéro dans vecteur ccG
                    elif d_ == 2:
                        aux_num.append([num * d_, num * d_ + 1])
                    elif d_ == 3:
                        aux_num.append([num * d_, num * d_ + 1, num * d_ + 2])
                num = pos_voisins[I-1] #taking the last point. Reasonable ?
                if d_ == 1:
                    aux_num.append([num]) #numéro dans vecteur ccG
                elif d_ == 2:
                    aux_num.append([num * d_, num * d_ + 1])
                elif d_ == 3:
                    aux_num.append([num * d_, num * d_ + 1, num * d_ + 2])
            
            try: #checking barycentric coordinates are ok !
                assert(np.absolute(coord_bary).max() < 1.e4)
            except AssertionError:
                print(test)
                print('Coord bary')
                print(coord_bary)
                print('Stopping computation...')
                sys.exit()

            #On associe le tetra à la face
            assert(len(aux_num) > 0)
            res_num[i] = aux_num
            #res_pos[i] = aux_pos

    print('Nb inner facets: %i' % nb_inner_facets)
    print('Nb extrapolation on inner facets: %i' % extrapolating_facets)
    print('Percentage related: %.5e%%' % (extrapolating_facets / nb_inner_facets * 100))

    return res_num,res_coord
