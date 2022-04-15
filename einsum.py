from numpy import *
from numpy import array       as ar
from numpy import einsum      as es
from numpy import squeeze     as sz
from numpy import expand_dims as ed

''' We define autodifferentiation purely on einsum and activations '''

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
    a, b, out = split(op)
    return make((out, b, a))

class ES:
    def __init__(self, op):
        self.op      = op
        self.grad_op = reorder(op)

    def __call__(self, l, w, x):
        y    = es(self.op, w, x)
        grad = es(self.grad_op, l, x)
        return y, grad
