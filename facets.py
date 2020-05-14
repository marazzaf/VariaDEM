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

def facet_neighborhood(mesh_,elt_):
    """Returns a dictionnary containing as key the index of the facets and as values the list of indices of the cells (or cell) containing the facet. """
    indices = dict([])

    for f in facets(mesh_):
        voisins_num = []
        for c in cells(f):
            voisins_num.append(c.index())

        indices[f.index()] = voisins_num
    return indices

def position_ddl_cells(mesh_,elt_): #Renvoie les cellules voisines d'une cellule donnée et leurs positions
    cell_pos = dict([])
    for c in cells(mesh_):
        cell_pos[c.index()] = elt_.tabulate_dof_coordinates(c)[0]
    return cell_pos

def position_vertex_bord(mesh_,face_num):
    vertex_pos = set()
    for f in facets(mesh_):
        if len(face_num.get(f.index())) == 1: #face sur le bord
            for v in vertices(f):
                vertex_pos.add( (v.x(0),v.x(1),v.x(2)) )
    return vertex_pos

def cell_neighbors(mesh_,elt_): #Renvoie les cellules voisines d'une cellule donnée et leurs positions
    cell_num = dict([])
    cell_pos = dict([])
    for cell in cells(mesh_):
        num = []
        pos = []
        for f in facets(cell):
            for c in cells(f):
                if c.index() != cell.index():
                    num.append(c.index())
                    pos.append(elt_.tabulate_dof_coordinates(c)[0])
        cell_num[cell.index()] = num
        cell_pos[cell.index()] = pos
    return cell_num,cell_pos

def four_set(dico):
    res = []
    for i,j in dico.items():
        for k,l in dico.items():
            if k > i:
                for m,n in dico.items():
                    if m > k and m > i:
                        for o,p in dico.items():
                            if o > m and o > k and o > i:
                                dico = {i:j, k:l, m:n, o:p}
                                res.append(dico)
    return res

def tetra_face(face_num,face_pos,cell_num,cell_pos):
    result_num = dict([])
    result_pos = dict([])
    #for (f, voisins_num),(g,voisins_pos) in zip(face_num.items(),face_pos.items()):
    for (f,voisins_num),(g,voisins_pos) in zip(face_num.items(),face_pos.items()):
        #print('face : %i' % f
        if(len(voisins_num) == 1): #cad face sur le bord
            result_num[f] = []
            result_pos[f] = []
        else: #face interne
            #dico_cellules_pot_num ={cell_num[voisins_num[0]]:cell_pos.get(voisins_num[0]), cell_num[voisins_num[1]]:cell_pos.get(voisins_num[1])}
            cellules_pot_num = [voisins_num[0],voisins_num[1]] + cell_num[voisins_num[0]] + cell_num[voisins_num[1]]
            cellules_pot_pos = [voisins_pos[0],voisins_pos[1]] + cell_pos.get(voisins_num[0]) + cell_pos.get(voisins_num[1])
            dico_cellules_pot = dict([])
            for i,j in zip(cellules_pot_num,cellules_pot_pos):
                dico_cellules_pot[i] = j #création du dico pour avoir les sous-ensembles à 4 éléments ensuite
            #print(dico_cellules_pot
            liste_sous_ensembles = four_set(dico_cellules_pot)
            #print(len(liste_sous_ensembles)
            for i in liste_sous_ensembles:
                #print(i
                if not( tetra_applati(i.values()) ):
                    result_num[f] = i.keys()
                    result_pos[f] = i.values
                    break
            assert( len(result_num.get(f)) == 4)
    return result_num,result_pos

def tetra_face_bis(face_num,face_pos,face_nnn_num,face_nnn_pos, h_): #On va aller chercher nnn pour le tetra
    result_num = dict([])
    result_pos = dict([])
    #for (f, voisins_num),(g,voisins_pos) in zip(face_num.items(),face_pos.items()):
    for f,voisins_num in face_num.items():
        #print('face : %i' % f
        if(len(voisins_num) == 1): #cad face sur le bord
            result_num[f] = []
            result_pos[f] = []
        else: #face interne
            dico_cellules_pot = dict([])
            for i,j in zip(face_nnn_num.get(f),face_nnn_pos.get(f)):
                dico_cellules_pot[i] = j #création du dico pour avoir les sous-ensembles à 4 éléments ensuite
            #print(dico_cellules_pot
            liste_sous_ensembles = four_set(dico_cellules_pot)
            #print(len(liste_sous_ensembles)
            for i in liste_sous_ensembles:
                #print(i
                if not( tetra_applati(i.values(), h_) ):
                    result_num[f] = i.keys()
                    result_pos[f] = i.values()
                    break
            if result_num.get(f) == None:
                #Faire quoi ?
                result_num[f] = [voisins_num[0],voisins_num[1],voisins_num[0],voisins_num[1]]
                voisins_pos = face_pos.get(f)
                result_pos[f] = [voisins_pos[0],voisins_pos[1],voisins_pos[0],voisins_pos[1]]
            #assert( len(result_num.get(f)) == 4) #Enlevé car on ne trouvait pas de voisin dans certains cas...
    return result_num,result_pos

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

def tetra_applati(tetra_face_pos, h_):
    #print(tetra_face_pos
    vec_test = np.array([tetra_face_pos[1] - tetra_face_pos[0], tetra_face_pos[2] - tetra_face_pos[0], tetra_face_pos[3] - tetra_face_pos[0]])
    test = np.dot(vec_test[0], np.cross(vec_test[1], vec_test[2]))
    #print(test
    if np.abs(test) < 1.e-5 * h_ * h_ * h_: #1.e-5: #1.e-10:
        return True
    else:
        return False

def tri_applati(tri_pos, h_):
    #print(tetra_face_pos
    vec_test = np.array([tri_pos[1] - tri_pos[0], tri_pos[2] - tri_pos[0]])
    test = np.abs(np.cross(vec_test[0], vec_test[1]))
    #print(test
    if test < 1.e-5 * h_ * h_: #1.e-5: #1.e-10: #Pour avoir une erreur relative
        return True
    else:
        return False

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
                

def facet_n_neighborhood(face_num,face_pos,cell_num,cell_pos, dim): #dimension 2d ou 3d
    result_num = dict([])
    result_pos = dict([])
    for (f,voisins_num),(g,voisins_pos) in zip(face_num.items(),face_pos.items()):
        if(len(voisins_num) == 1): #cad face sur le bord
            result_num[f] = [voisins_num[0]] + cell_num[voisins_num[0]]
            result_pos[f] = [voisins_pos[0]] + cell_pos[voisins_num[0]]
        else: #face interne
            #result_num[f] = [voisins_num[0],voisins_num[1]] + cell_num[voisins_num[0]] + cell_num[voisins_num[1]]
            #result_pos[f] = [voisins_pos[0],voisins_pos[1]] + cell_pos[voisins_num[0]] + cell_pos[voisins_num[1]]
            result_num[f] = cell_num[voisins_num[0]] + cell_num[voisins_num[1]]
            result_pos[f] = cell_pos[voisins_num[0]] + cell_pos[voisins_num[1]]
            assert( len(result_num.get(f)) >= dim+1 )
    return result_num,result_pos #renvoi les numéros des cellules autour de la face considérée puis la même avec les coodonnées

def facet_nn_neighborhood(face_n_num,face_n_pos,cell_num,cell_pos, dim): #Préparation pour version avec coord bary génréalisées
    result_num = dict([])
    result_pos = dict([])
    for (f,voisins_num),(g,voisins_pos) in zip(face_n_num.items(),face_n_pos.items()):
        for i in voisins_num:
            result_num[f] = voisins_num + cell_num[i]
            result_pos[f] = voisins_pos + cell_pos[i]
        assert( len(result_num.get(f)) >= dim+1 )
            #if( len(result_num.get(f)) < 4 ):
                #print('num face : %i, nb voisins % i' % (f,len(result_num.get(f)))
    return result_num,result_pos #renvoi les numéros des cellules autour de la face considérée puis la même avec les coodonnées

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

#def add_vertex_dof(mesh_,facet_n_num,facet_nn_num,facet_nn_pos,dico_pos_bary_faces,dico_vertex_bord, pos_vertex_bord, d_, dico_vert_cell, dico_pos_vert_ccG): #va renvoyer des numéros de ddl dans vecteur ccG. Sera plus simple...
def add_vertex_dof(facet_nn_num,facet_nn_pos, d_, vertex_num, vertex_pos, vertex_ccG):
    aux_num = dict([])
    res_pos = dict(facet_nn_pos)
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
    #On va ajouter les ddl des vertex sur le bord
    #Il faut chercher dans les faces voisines de cet
    for (i,j),k in zip(facet_nn_num.items(),facet_nn_pos.values()):
    #Ajouter le numéro des vertex au bord (dans vecteur ccG)
        for l in j: #l est le numéro des cellules nn de i
            if vertex_num.get(l) != None:
                aux_aux_num = []
                aux_pos = []
                for v,w in zip(vertex_num.get(l),vertex_pos.get(l)):
                    aux_aux_num.append(vertex_ccG.get(v))
                    aux_pos.append(np.array(w))
                aux_num[i] = aux_num[i] + aux_aux_num
                res_pos[i] = facet_nn_pos[i] + aux_pos
    
    return aux_num,res_pos #On va prendre en compte en plus ces connées danns le calcul du convexe entourant chaque face

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

def dico_position_bary_face(mesh_,dofmap_,elt_):
    resultat = dict([])
    for cell in cells(mesh_):
        num_local_face = -1
        for f in facets(cell):
            num_local_face += 1
            pos_dof_cell = elt_.tabulate_dof_coordinates(cell)
            resultat[f.index()] = pos_dof_cell[num_local_face]
    return resultat

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

def min_I_mesh(face_n_num,pos_bary_cells,pos_vert,pos_bary_facets,dim,d_,h_): #pos_vert contient les positions des barycentres sur les faces au bord
    toutes_pos_ddl = [] #ordre : d'abord tous les ddl de cellules puis tous ceux des vertex au bord
    for i in pos_bary_cells.values():
        toutes_pos_ddl.append(i)
    for i in pos_vert.values():
        toutes_pos_ddl.append(i)
    toutes_pos_ddl = np.array(toutes_pos_ddl)
    tree = KDTree(toutes_pos_ddl)

    #outputs
    nb_inner_facets = 0
    extrapolating_facets = 0
    I=0 #on cherche sa valeur
    J = dict()
    for i,j in face_n_num.items():
        if len(j) > 1: #cad facette pas au bord
            nb_inner_facets += 1
            x = pos_bary_facets.get(i) #position du barycentre de la face
            nb_voisins = dim+1
            distance,pos_voisins = tree.query(x, 20) #valeur fixée ici (pour etre large !)
            data_convex_hull = [] #[] #[x]
            for k in range(nb_voisins):
                data_convex_hull.append(tree.data[pos_voisins[k]])

            #computation of closest non-degenerate simplex
            degenerate = True
            convex_aux = ConvexHull(data_convex_hull,qhull_options='Qc QJ Pp', incremental=True)    

            while degenerate:
                if convex_aux.volume < 1.e-5 * h_**3: #ajouter un point et recommencer
                    nb_voisins += 1
                    I = max(I,nb_voisins)
                    convex_aux.add_points([tree.data[pos_voisins[nb_voisins-1]]])
                else: #fini
                    if nb_voisins not in J:
                        J[nb_voisins] = 1
                    else:
                        J[nb_voisins] += 1
                    degenerate = False
                    convex_aux.close()
            
    return I,J #J est le bon. Sortir les valeurs propres par contre...

def comparison_bary_coord_old(face_n_num,pos_bary_cells,pos_vert,pos_bary_facets,dim,d_,h_): #pos_vert contient les positions des barycentres sur les faces au bord
    toutes_pos_ddl = [] #ordre : d'abord tous les ddl de cellules puis tous ceux des vertex au bord
    for i in pos_bary_cells.values():
        toutes_pos_ddl.append(i)
    for i in pos_vert.values():
        toutes_pos_ddl.append(i)
    toutes_pos_ddl = np.array(toutes_pos_ddl)
    tree = KDTree(toutes_pos_ddl)
    tous_num_ddl = np.arange(len(toutes_pos_ddl) * d_)
    #calcul du convexe associé à chaque face
    res_num = dict([])
    res_pos = dict([])
    res_coord = dict([])

    #outputs
    nb_inner_facets = 0
    extrapolating_facets = 0
    I=6 #paramètre à faire bouger
    J = dict()
    for i,j in face_n_num.items():
        if len(j) > 1: #cad face pas au bord
            nb_inner_facets += 1
            aux_num = []
            aux_pos = []
            x = pos_bary_facets.get(i) #position du barycentre de la face      
            nb_voisins = max(dim+1,I) #dim+1 #max(dim+1,I) #4 #5 #6 #9 #12 #15
            distance,pos_voisins = tree.query(x, 20) #valeur fixée ici (pour etre large !)
            #print('Distances:')
            #print(distance)
            data_convex_hull = [x]
            for k in range(nb_voisins):
                data_convex_hull.append(tree.data[pos_voisins[k]])

            #computation of closest non-degenerate simplex
            degenerate = True
            test_simplex = False #True
            simplex = data_convex_hull[1:dim+2]
            if test_simplex:
                nb_points = dim+1
                convex_aux = ConvexHull(simplex,qhull_options='Qc QJ Pp', incremental=True)
            else:
                nb_points = nb_voisins
                convex_aux = ConvexHull(data_convex_hull,qhull_options='Qc QJ Pp')    

            while degenerate:
                if convex_aux.volume < 1.e-5 * h_**3: #ajouter un point et recommencer
                    #print(convex_aux.volume)
                    nb_points += 1
                    I = max(I,nb_points)
                    print(nb_points)
                    convex_aux.add_points([tree.data[pos_voisins[nb_points]]])
                    #print(convex_aux.volume)
                    simplex.append(tree.data[pos_voisins[nb_points]])
                else: #finie
                    if nb_points not in J:
                        J[nb_points] = 1
                    else:
                        J[nb_points] += 1
                    #print(convex_aux.volume)
                    #print(len(convex_aux.vertices))
                    degenerate = False
                    convex_aux.close()

            if test_simplex:
                convexe = ConvexHull([x]+simplex,qhull_options='Qc QJ Pp')
                #print(len([x]+simplex))
                #print(len(data_convex_hull))
            else:
                convexe = ConvexHull(data_convex_hull,qhull_options='Qc QJ Pp') # p C-1.e-4') #Tv #QJ
            #print('Nb points in cloud: %i' % len(convexe.points))

            #test=False
            if 0 not in convexe.vertices: #interpolation
                test=True
                #Faire une triangulation de Delaunay des points du convexe
                list_points = convexe.points
                delau = Delaunay(list_points[1:]) #on retire le barycentre de la face de ces données. On ne le veut pas dans le Delaunay
                #Ensuite, utiliser le transform sur le Delaunay pour avoir les coords bary du bay de la face. Il reste seulement à trouver dans quel tétra est-ce que toutes les coords sont toutes positives !
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
                test=False
                extrapolating_facets += 1
                #taking the last point in the hull and the dim first !
                mat = []
                mat.append(convex_aux.points[0] - convex_aux.points[-1])
                if d_ == 3: #to have a second vector not colinear with the first
                    for k in np.arange(len(convex_aux.points))+1:
                        test = convex_aux.points[k] - convex_aux.points[-1]
                        test /= np.linalg.norm(test)
                        if np.linalg.norm(np.cross(test, mat[0] / np.linalg.norm(mat[0]))) > 1.e-5:
                            mat.append(convex_aux.points[k] - convex_aux.points[-1])
                            break

                    
                for k in np.arange(len(convex_aux.points))+1: #to have the last points to have an invertible matrix
                    test = convex_aux.points[k] - convex_aux.points[-1]
                    test /= np.linalg.norm(test)
                    if np.absolute(np.linalg.det(np.array([mat[0]/np.linalg.norm(mat[0]),mat[1]/np.linalg.norm(mat[1])] + [test]))) >  1.e-5:
                            mat.append(convex_aux.points[k] - convex_aux.points[-1])
                            break
                        

                mat = np.array(mat)
                assert(mat.shape == (dim,dim))
                assert(np.absolute(np.linalg.det(mat)) > 1.e-10) #no degenerate barycentric coordiantes !
                rhs = np.array(x - convex_aux.points[-1])
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
                num = pos_voisins[nb_points] #quel rapport avec nb_points ?
                if d_ == 1:
                    aux_num.append([num]) #numéro dans vecteur ccG
                elif d_ == 2:
                    aux_num.append([num * d_, num * d_ + 1])
                elif d_ == 3:
                    aux_num.append([num * d_, num * d_ + 1, num * d_ + 2])
            
            try: #checking barycentric coordinates are ok !
                assert(np.absolute(coord_bary).max() < 1.e3)
            except AssertionError:
                print(test)
                print('Coord bary')
                print(coord_bary)
                sys.exit()
                

            #On associe le tetra à la face
            #print(test)
            #print(aux_num)
            assert(len(aux_num) > 0)
            res_num[i] = aux_num
            #print(aux_num
            res_pos[i] = aux_pos

    print('I = %i' % I)
    print(J)
    print('Nb inner facets: %i' % nb_inner_facets)
    print('Nb extrapolation on inner facets: %i' % extrapolating_facets)
    print('Percentage related: %.3f%%' % (extrapolating_facets / nb_inner_facets * 100))

    return res_num,res_pos,res_coord,0.

#def dico_position_vertex_bord(mesh_, facets_, face_num, nb_ddl_cellules, d_, dim, BC_D, BC_N): #lists of tags for the various boundary conditions on facets
#    vertex_associe_face = dict([])
#    pos_ddl_vertex = dict([])
#    num_ddl_vertex_ccG = dict([])
#    compteur = nb_ddl_cellules-1
#    bc_associe_facet = dict([])
#    vertex_N = set([]) #index of vertices with Neumann BC
#    vertex_D = set([]) #index of vertices with Dirichlet BC
#    for f,g in zip(facets(mesh_),facets_):
#            if len(face_num.get(f.index())) == 1: #Face sur le bord car n'a qu'un voisin
#                bc_associe_facet[f.index()] = g
#                vertex_associe_face[f.index()] = []
#                for v in vertices(f):
#                    if g in BC_D:
#                        vertex_D.add(v.index())
#                    elif g in BC_N:
#                        vertex_N.add(v.index())
#
#    #Taking out of vertex_N all vertices with at least one
#                        
#    for v in vertices(mesh_):
#        test_bord = False #pour voir si vertex est sur le bord
#        for f in facets(v):
#            if len(face_num.get(f.index())) == 1: #Face sur le bord car n'a qu'un voisin
#                vertex_associe_face[f.index()] = vertex_associe_face[f.index()] + [v.index()]
#                test_bord = True
#        if test_bord: #vertex est bien sur le bord.
#            if dim == 2: #pour position du vertex dans l'espace
#                pos_ddl_vertex[v.index()] = np.array([v.x(0),v.x(1)])
#            elif dim == 3:
#                pos_ddl_vertex[v.index()] = np.array([v.x(0),v.x(1),v.x(2)])
#
#    #first run for Neumann Boundary conditions
#    for f,g in zip(facets(mesh_),bc_associe_facet.values()):
#        if g in BC_N:
#            if d_ == 2: #pour num du dof du vertex dans vec ccG
#                num_ddl_vertex_ccG[v.index()] = [compteur+1, compteur+2]
#                compteur += 2
#            elif d_ == 3:
#                num_ddl_vertex_ccG[v.index()] = [compteur+1, compteur+2, compteur+3]
#                compteur += 3
#            elif d_ == 1:
#                num_ddl_vertex_ccG[v.index()] = [compteur+1]
#                compteur += 1
#                
#    #second run for dirichlet boundary conditions
#    nb_ddl_reduit = compteur+1
#    for f,g in zip(facets(mesh_),bc_associe_facet.values()):
#        if g in BC_D:
#            if d_ == 2: #pour num du dof du vertex dans vec ccG
#                num_ddl_vertex_ccG[v.index()] = [compteur+1, compteur+2]
#                compteur += 2
#            elif d_ == 3:
#                num_ddl_vertex_ccG[v.index()] = [compteur+1, compteur+2, compteur+3]
#                compteur += 3
#            elif d_ == 1:
#                num_ddl_vertex_ccG[v.index()] = [compteur+1]
#                compteur += 1
#
#    return vertex_associe_face,pos_ddl_vertex,num_ddl_vertex_ccG,nb_ddl_reduit

def schur(A_BC):
    nb_ddl_ccG = A_BC.shape[0]
    l = A_BC.nonzero()[0]
    aux = set(l) #contains number of Dirichlet dof
    nb_ddl_Dirichlet = len(aux)
    aux_bis = set(range(nb_ddl_ccG))
    aux_bis = aux_bis.difference(aux) #contains number of vertex non Dirichlet dof
    sorted(aux_bis) #sort the set

    #Get non Dirichlet values
    mat_not_D = sp.dok_matrix((nb_ddl_ccG - nb_ddl_Dirichlet, nb_ddl_ccG))
    for (i,j) in zip(range(mat_not_D.shape[0]),aux_bis):
        mat_not_D[i,j] = 1.

    #Get Dirichlet boundary conditions
    mat_D = sp.dok_matrix((nb_ddl_Dirichlet, nb_ddl_ccG))
    for (i,j) in zip(range(mat_D.shape[0]),aux):
        mat_D[i,j] = 1.
    return mat_not_D.tocsr(), mat_D.tocsr()

def mass_vertex_dofs(mesh_, nb_ddl_ccG_, pos_ddl_vertex, num_ddl_vertex_ccG, d_, dim, rho_):
    res = np.zeros(nb_ddl_ccG_)
    if d_ == dim: #vectorial problem
        U_DG = VectorFunctionSpace(mesh_, "DG", 0)
        #U_CR = VectorFunctionSpace(mesh_, "CR", 1)
    elif d_ == 1: #scalar problem
        U_DG = FunctionSpace(mesh_, "DG", 0)
        #U_CR = FunctionSpace(mesh_, "CR", 1)
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

def matrice_passage_ccG_CG(mesh_, nb_ddl_ccG_,num_vert_ccG,d_,dim):
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

def matrice_passage_ccG_DG(nb_ddl_cells,nb_ddl_ccG):
    return sp.eye(nb_ddl_cells, n = nb_ddl_ccG, format='csr')

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

def matrice_passage_ccG_DG_1(mesh_, nb_ddl_ccG_, d_, dim_,mat_grad, passage_ccG_CR):
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
    matrice_resultat_1 = sp.dok_matrix((nb_total_dof_DG_1,nb_ddl_ccG_)) #Matrice vide.
    matrice_resultat_2 = sp.dok_matrix((nb_total_dof_DG_1,nb_ddl_grad)) #Matrice vide.
    
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

def gradient_matrix(mesh_, d_):
    if d_ == 1:
        W = VectorFunctionSpace(mesh_, 'DG', 0)
        U_CR = FunctionSpace(mesh_, 'CR', 1)
    elif d_ >= 2:
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
