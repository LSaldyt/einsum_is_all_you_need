from numpy import *
from numpy import array       as ar
from numpy import einsum      as es
from numpy import squeeze     as sz
from numpy import expand_dims as ed

from einsum     import *
from primitives import *

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
    l = ones_like(x)

    add, add_jvps = autodiff_rules['add']
    y, gy = autodiff(add, add_jvps, l, x, w)
    print(y)
    print(gy)
    # def autodiff(f, jvps, l, *args):

    # mult = ES('ij,jk->ik')
    # y, gy = mult(w, x, l)

    # print(w)
    # print(y)
    # print(l)
    # print(gy)
    # α = 3e-4
    # w -= α * gy
    # print(w)

if __name__ == '__main__':
    main()
