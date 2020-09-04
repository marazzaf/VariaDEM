# coding: utf-8
import sys
sys.path.append('../../')
import matplotlib.pyplot as plt
from facets import *
from scipy.sparse.linalg import spsolve,cg

# Form compiler options
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

# Scaled variables
E = Constant(70e3)
nu = Constant(0.3)
lmbda = E*nu/(1+nu)/(1-2*nu)
mu = E/2./(1+nu)
sig0 = Constant(250.)  # yield strength
Et = Constant(E/5.)  # tangent modulus
H = E*Et/(E-Et)  # hardening modulus
penalty = mu

# Create mesh and define function space
Lx,Ly = 0.4, 0.4
Lz = 1. #longeur de la poutre
S = Lx * Ly #surface de la poutre
#mesh = BoxMesh(Point(0., 0., 0.), Point(Lx, Ly, Lz), 3, 3, 10) #5, 5, 60)
mesh = Mesh("../maillages/poutre_h_0_2.xml")
facets = MeshFunction("size_t", mesh, 2)
#facets = MeshFunction("size_t", mesh, "../maillages/poutre_h_0_05_facet_region.xml")
ds = Measure('ds')(subdomain_data=facets)
h_max = mesh.hmax() #Taille du maillage.


# Sub domain for clamp at left end
def left(x, on_boundary):
    return near(x[2], 0.) and on_boundary

# Sub domain for rotation at right end
def right(x, on_boundary):
    return near(x[2], Lz) and on_boundary

# Mesh-related functions
h = CellVolume(mesh) #Pour volume des particules voisines
hF = FacetArea(mesh)
h_avg = (h('+') + h('-'))/ (2. * hF('+'))
nF = FacetNormal(mesh)

#Function spaces
U_DG = VectorFunctionSpace(mesh, 'DG', 0) #Pour délacement dans cellules
U_CR = VectorFunctionSpace(mesh, 'CR', 1) #Pour interpollation dans les faces
U_CG = VectorFunctionSpace(mesh, 'CG', 1) #Pour bc
W0 = FunctionSpace(mesh, 'DG', 0) 
W = TensorFunctionSpace(mesh, 'DG', 0)
U_DG_1 = VectorFunctionSpace(mesh, 'DG', 1) #Pour reconstruction ccG

#Functions for computation
solution_u_CR = Function(U_CR, name="Disp CR")
solution_u_DG = Function(U_DG_1, name="Disp DG")
sig = Function(W, name="Stress")
sig_old = Function(W)
sig_aux = Function(W)
n_elas = Function(W)
beta = Function(W0)
p = Function(W0, name="Cumulative plastic strain")
dp = Function(W0)
Du_CR = Function(U_CR, name="Current increment")
dim = solution_u_CR.geometric_dimension()  # space dimension
d = dim #pb vectoriel

#Dirichlet boundary condition on the boundary
delta_lim = Constant(2. * sig0 / E * Lz) #2 fois limite élastique
u_D = Expression('delta*t', delta=delta_lim, t=0, degree=2)
u_D_prev = Expression('delta*t', delta=delta_lim, t=0, degree=2)

#reference solution
x = SpatialCoordinate(mesh)
u_ref = Expression(('-nu*u_D/L*x[0]', '-nu*u_D/L*x[1]', 'u_D*x[2]/L'), u_D=u_D, L=Lz, nu=nu, degree=2)
u_ref_prev = Expression(('-nu*u_D/L*x[0]', '-nu*u_D/L*x[1]', 'u_D*x[2]/L'), u_D=u_D_prev, L=Lz, nu=nu, degree=2)
#Du_ref = Expression((('-nu*u_D/L', '0', '0'), ('0', '-nu*u_D/L', '0'), ('0', '0', 'u_D/L')), u_D=u_D, L=Lz, nu=nu, degree=1)
delta_e = Constant(sig0 / E * Lz)
sig_ref = Expression((('0', '0', '0'), ('0', '0', '0'), ('0', '0', 'u_D > delta_e ? sig0 + H*(u_D-delta_e)/L : E * u_D / L')), u_D=u_D, L=Lz, E=E, delta_e=delta_e, sig0=sig0, H=Et, degree=1) #Constant(H), degree=2)

def eps(vv):
    return sym(grad(vv))

def sigma(eps_el):
    return lmbda*tr(eps_el)*Identity(3) + 2*mu*eps_el

def sigma_tang(e,betaa,n_elass): #,sig_vm):
    return sigma(e) - 3*mu*(3*mu/(3*mu+H)-betaa)*inner(n_elass, e)*n_elass-2*mu*betaa*dev(e)

ppos = lambda x: (x+abs(x))/2. #fonction partie positive

def proj_sig(deps, old_sig, old_p): #c'est le retour radial
    sig_elas = old_sig + sigma(deps)
    s = dev(sig_elas)
    sig_eq = sqrt(3/2.*inner(s, s))
    f_elas = sig_eq - sig0 - H*old_p
    dp = ppos(f_elas)/(3*mu+H)
    n_elas = s/sig_eq*ppos(f_elas)/f_elas
    beta = 3*mu*dp/sig_eq
    new_sig = sig_elas-beta*s
    return new_sig, n_elas, beta, dp

#Cell-centre Galerkin reconstruction
dofm = U_DG.dofmap()
nb_ddl_cells = len(dofm.dofs())
elt = U_DG.element()
dofmap_CG = U_CG.dofmap()
nb_ddl_CG = len(dofmap_CG.dofs())
face_num,face_pos = facet_neighborhood(mesh,elt)
dofmap_ = U_CR.dofmap()
nb_ddl_CR = len(dofmap_.dofs())
elt_bis = U_CR.element()
dico_pos_bary_faces = dico_position_bary_face(mesh,dofmap_,elt_bis)
pos_bary_cells = position_ddl_cells(mesh,elt)
vertex_associe_face,pos_ddl_vertex,num_ddl_vertex_ccG = dico_position_vertex_bord(mesh, face_num, nb_ddl_cells, d, dim)
nb_ddl_ccG = nb_ddl_cells + d * len(pos_ddl_vertex)
print('nb ddl ccG : %i' % nb_ddl_ccG)
convexe_num,convexe_coord = smallest_convexe_bary_coord_bis(face_num,pos_bary_cells,pos_ddl_vertex,dico_pos_bary_faces,dim,d)
print('Convexe ok !')
passage_ccG_to_CR = matrice_passage_ccG_CR(mesh, nb_ddl_ccG, convexe_num, convexe_coord, vertex_associe_face, num_ddl_vertex_ccG, d, dim)
passage_ccG_to_CG = matrice_passage_ccG_CG(mesh, nb_ddl_ccG,num_ddl_vertex_ccG,d,dim)
passage_ccG_to_DG = matrice_passage_ccG_DG(nb_ddl_cells,nb_ddl_ccG)
print('matrices passage ok !')

# Define variational problem
u_CR = TrialFunction(U_CR)
v_CR = TestFunction(U_CR)
u_DG = TrialFunction(U_DG)
v_DG = TestFunction(U_DG)
u_CG = TrialFunction(U_CG)
v_CG = TestFunction(U_CG)
u_DG_1 = TrialFunction(U_DG_1)
v_DG_1 = TestFunction(U_DG_1)

#bilinear forms
#gradient matrix
Dv_DG = TestFunction(W)
a6 = inner(grad(u_CR), Dv_DG) / h * dx
A6 = assemble(a6)
row,col,val = as_backend_type(A6).mat().getValuesCSR()
mat_grad = sp.csr_matrix((val, col, row))
#DG P1 reconstruction
passage_ccG_to_DG_1 = matrice_passage_ccG_DG_1(mesh, nb_ddl_ccG, d, dim, mat_grad, passage_ccG_to_CR)

#FV penalty term
A_pen = penalty_FV(penalty, nb_ddl_ccG, mesh, face_num, d, dim, mat_grad, dico_pos_bary_faces, passage_ccG_to_CR)

#cell-boundary penalty
A_pen_bis = penalty_boundary(penalty, nb_ddl_ccG, mesh, face_num, d, dim, num_ddl_vertex_ccG, mat_grad, dico_pos_bary_faces, passage_ccG_to_CR, pos_bary_cells)

#Variational problem
#a1 = inner(eps(u_CR), sigma(eps(v_CR))) * dx
#A1 = assemble(a1)
#row,col,val = as_backend_type(A1).mat().getValuesCSR()
#A1 = sp.csr_matrix((val, col, row))
#AA1 = passage_ccG_to_CR.T * A1 * passage_ccG_to_CR
#A = AA1 + A_pen + A_pen_bis

def bilinear_form(betaa,n_elass): #pour itérations de point fixe
    #Variational problem
    a1 = inner(eps(u_CR), sigma_tang(eps(v_CR),betaa,n_elass)) * dx
    A1 = assemble(a1)
    row,col,val = as_backend_type(A1).mat().getValuesCSR()
    A1 = sp.csr_matrix((val, col, row))
    AA1 = passage_ccG_to_CR.T * A1 * passage_ccG_to_CR

    return AA1 + A_pen + A_pen_bis

def linear_form(sigg, X_ccG): #pour itérations point fixe
    #residual of Newton-Raphson procedure...
    r1 = inner(eps(v_CR), sigg) * dx
    R1 = assemble(-r1)
    RR1 = passage_ccG_to_CR.T * R1.get_local()

    #FV penalty bilinear form
    RR2 = -A_pen * X_ccG

    #Boundary penalty bilinear form
    RR3 = -A_pen_bis * X_ccG
    
    return RR1 + RR2 + RR3

#setting Dirichlet BC
facets.set_all(0)
still_boundary = AutoSubDomain(left)
still_boundary.mark(facets, 29)
moving_boundary = AutoSubDomain(right)
moving_boundary.mark(facets, 30)

#Imposition des conditions de Dirichlet non-homogène
a4 = v_CG('+')[2] / hF * (ds(29) + ds(30))
A4 = assemble(a4)
A_BC = passage_ccG_to_CG.T * A4.get_local()
nz = A_BC.nonzero()[0]
A_BC[nz[0]-1] = 1.
A_BC[nz[0]-2] = 1.

#Imposing strongly Dirichlet BC
mat_not_D,mat_D = schur(A_BC)
print('matrices passage Schur ok')

file_results = File("h_0_2/test_.pvd")
res = open('h_0_2/res.txt', 'w')
stresses = open('h_0_2/stresses.txt', 'w')

#Parameters iterations
Nitermax, tol = 200, 1.e-8 # parameters of the Newton-Raphson procedure
Nincr = 20 #50
load_steps = np.linspace(0, 1., Nincr+1)[1:]

u = np.zeros(nb_ddl_ccG) #solution
for (i, t) in enumerate(load_steps):
    u_D.t = t
    print('\nIncrement: %i' % (i+1))

    #increments
    Du = np.zeros(nb_ddl_ccG)
    Du_CR.interpolate(Constant((0, 0, 0)))

    #assembing bilinear form
    A = bilinear_form(beta,n_elas)
    #assembling linear form
    Res = linear_form(sig, Du)

    # imposing strongly the BC...
    A_D = mat_D * A * mat_D.T
    A_not_D = mat_not_D * A * mat_not_D.T
    B = mat_not_D * A * mat_D.T
    L_not_D = mat_not_D * Res
    
    #interpolation of increment of Dirichlet BC
    F = local_project(u_ref - u_ref_prev, U_CG).vector().get_local()
    F = mat_D * passage_ccG_to_CG.T * F
    L_not_D = L_not_D - B * F #Residual with nonhomogeneous BC

    #initial residual
    nRes0 = np.linalg.norm(L_not_D)
    nRes = nRes0
    print("    Initial residual: %.3e" % nRes0)
    niter = 0
    incr_norm = tol+1.

    while ((i == 0 and nRes/nRes0 > tol) or (i>0 and incr_norm > tol)) and niter < Nitermax:
        #solving problem
        du_red,info = cg(A_not_D, L_not_D, tol=tol)
        assert(info == 0)
        #du_red = spsolve(A_not_D, L_not_D)
        if niter == 0:
            du = mat_not_D.T * du_red + mat_D.T * F
        else:
            du = mat_not_D.T * du_red
        #updating solution
        Du += du
        Du_CR.vector().set_local(passage_ccG_to_CR * Du)

        #updating problem
        deps = eps(Du_CR)
        sig_, n_elas_, beta_, dp_ = proj_sig(deps, sig_old, p)
        local_project(sig, W, sig_aux)
        local_project(sig_, W, sig)
        local_project(n_elas_, W, n_elas)
        local_project(beta_, W0, beta)
        local_project(dp_, W0, dp)
        
        #recomputing tangent stiffness matrix
        A = bilinear_form(beta,n_elas) #Changes with plastic deformation
        A_not_D = mat_not_D * A * mat_not_D.T
        B = mat_not_D * A * mat_D.T

        #recomputing rhs
        Res = linear_form(sig, Du)
        L_not_D = mat_not_D * Res #residual without homogeneous BC after first iteration...

        #Computing convergence test
        nRes = np.linalg.norm(L_not_D)
        print("    Residual: %.3e" % nRes)
        if i>0:
            incr_norm = local_project(sig - sig_aux,W).vector().norm('l2') / (sig_old.vector().norm('l2'))
        print('    incr_norm : %.5e' % incr_norm)
        niter += 1

    # Post-processing
    u += Du
    p.assign(p+dp)

    #visualisation output
    solution_u_DG.vector().set_local(passage_ccG_to_DG_1 * u)
    file_results.write(solution_u_DG, t)
    solution_u_CR.vector().set_local(passage_ccG_to_CR * u)
    file_results.write(sig, t)
    file_results.write(p, t)

    #error computation...
    disp_ref = interpolate(u_ref, U_DG_1)
    err_u_L2 = errornorm(solution_u_DG, disp_ref, 'L2')
    u_ref_ccG = passage_ccG_to_DG.T * interpolate(u_ref, U_DG).vector().get_local() + passage_ccG_to_CG.T * interpolate(u_ref, U_CG).vector().get_local()
    #diff = u - u_ref_ccG
    #err_energy = np.dot(diff, A * diff)
    #err_energy = np.sqrt(0.5 * err_energy)
    stresses.write('%.5e %.5e %.5e\n' % (t, sig(0.5*Lx,0.5*Ly,0.5*Lz)[-1], sig_ref(0.5*Lx,0.5*Ly,0.5*Lz)[-1]))
    stress_ref = interpolate(sig_ref, W)
    err_energy = errornorm(sig, stress_ref, 'L2')

    #output
    res.write("%.5e %.5e %.5e\n" % (t, err_u_L2, err_energy))

    #updating for next increment
    u_D_prev.t = t
    sig_old.assign(sig)

#end of computation
res.close()
stresses.close()

