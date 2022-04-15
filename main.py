from numpy import *
from numpy import array       as ar
from numpy import einsum      as es
from numpy import squeeze     as sz
from numpy import expand_dims as ed

from einsum import *

# def sigmoid(x):
#     return 1 / (1 + exp(-x))
#
# def mynetwork(x):
#     layers = ['ij,jk->ik', 'ik->', 'sig']
#     loss   = 'bce'

def main():
    x = ar([[2., 2.],
            [2., 2.]])
    w = ar([[1., 0.],
            [0., 1.]])
    mult = ES('ij,jk->ik')
    loss = lambda y : ones_like(y)
    l = loss(x)
    y, gy = mult(w, x, l)

    print(w)
    print(y)
    print(l)
    print(gy)
    # α = 3e-4
    # w -= α * gy
    # print(w)

if __name__ == '__main__':
    main()
