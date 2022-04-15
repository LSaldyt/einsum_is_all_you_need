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

class Node:
    def __call__(self, *args):
        raise NotImplementedError
    def grad(self, *args):
        raise NotImplementedError

class ES:
    def __init__(self, op):
        self.op      = op
        self.grad_op = reorder(op)

    def __call__(self, w, x, loss):
        y    = es(self.op, w, x)
        l    = loss(y)
        grad = es(self.grad_op, l, x)
        return y, l, grad

# def sigmoid(x):
#     return 1 / (1 + exp(-x))
#
# def mynetwork(x):
#     layers = ['ij,jk->ik', 'ik->', 'sig']
#     loss   = 'bce'

def main():
    x = ar([[2., 2.], [2., 2.]])
    w = ar([[1., 0.], [0., 1.]])

    mult = ES('ij,jk->ik')
    loss = lambda y : ones_like(y)
    y, l, gy = mult(w, x, loss)

    print(w)
    print(y)
    print(l)
    print(gy)
    # α = 3e-4
    # w -= α * gy
    # print(w)

if __name__ == '__main__':
    main()
