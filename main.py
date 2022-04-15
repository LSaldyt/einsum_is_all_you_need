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

def value_and_grad(op, w, x, yt=None):
    ''' Run a forward and a backward pass for op '''
    y    = es(op, w, x)
    yt = ones_like(y) if yt is None else yt
    grad = es(reorder(op), yt, x)
    return y, grad

def sigmoid(x):
    return 1 / (1 + exp(-x))

'''
f/g
->
0 * g - 1 * g'
--------------
g^2

g' = -exp(-x)
g^2 = 1/(1+exp(-x))**2

so d_sigmoid
=
-exp(-x)
--------------
1/(1+exp(-x))**2

Much easier to do via autodiff :)
'''

def main():
    x = ar([[2., 2.], [2., 2.]])
    w = ar([[1., 0.], [0., 1.]])
    y, gy = value_and_grad('ij,jk->ik', w, x)
    print(w)
    print(y)
    print(gy)
    print(sigmoid(y))
    # α = 3e-4
    # w -= α * gy
    # print(w)

if __name__ == '__main__':
    main()
