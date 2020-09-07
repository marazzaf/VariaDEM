# coding: utf-8

#will contain all functions for bilinear forms etc... Even penalty ?

from dolfin import *
from scipy.sparse import csr_matrix
from DEM.errors import *

def elastic_bilinear_form(mesh_, d_, DEM_to_CR_matrix, sigma=grad, eps=grad):
    if d_ == 1:
        U_CR = FunctionSpace(mesh_, 'CR', 1)
    elif d_ == 2 or d_ == 3:
        U_CR = VectorFunctionSpace(mesh_, 'CR', 1)
    else:
        raise DimensionError('Problem with dimension of problem')

    u_CR = TrialFunction(U_CR)
    v_CR = TestFunction(U_CR)

    #Mettre eps et sigma en arguments de la fonction ?
    if d_ == 1:
        a1 = eps(u_CR) * sigma(v_CR) * dx
    elif d_ == 2 or d_ == 3:
        a1 = inner(eps(u_CR), sigma(v_CR)) * dx
    else:
        raise DimensionError('Problem with dimension of problem')
    
    A1 = assemble(a1)
    row,col,val = as_backend_type(A1).mat().getValuesCSR()
    A1 = csr_matrix((val, col, row))
    return DEM_to_CR_matrix.T * A1 * DEM_to_CR_matrix
