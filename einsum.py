from numpy import *
from numpy import array       as ar
from numpy import einsum      as es
from numpy import squeeze     as sz
from numpy import expand_dims as ed

''' Define autodifferentiation purely on einsum and activations '''

def split(op):
    ''' Split an einstein summation into inputs and outputs '''
    inp, out = op.split('->')
    elements = (*inp.split(','), out)
    return elements

def make(op):
    ''' Recreate an einstein summation from inputs and outputs '''
    return ','.join(op[:-1]) + '->' + str(op[-1])

def reorder(op):
    ''' Reorder a einstein summation to calculate the gradient '''
    *first, out = split(op)
    return make((out, *reversed(first)))

def jacobian_vector_product(l, y, op, *a):
    ''' Differentiating an einstein sum is deceptively simple: just re-order the arguments in many cases!
        Still needs to account for some edge cases, such as reducing summations like "ii->"
        Also needs to handle triple-tensor operations (if desired) '''
    grad_op = reorder(op)
    y       = es(op, *a)
    grad    = es(grad_op, l, a[-1])
    return grad
