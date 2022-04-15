from numpy import *
from numpy import array       as ar
from numpy import squeeze     as sz
from numpy import expand_dims as ed

from einsum     import *
from primitives import *
from autodiff   import *

# def sigmoid(x):
#     return 1 / (1 + exp(-x))
#
# def mynetwork(x):
#     layers = ['ij,jk->ik', 'ik->', 'sig']
#     loss   = 'bce'

def main():
    # x = ar([[2., 2.],
    #         [2., 2.]])
    # w = ar([[1., 0.],
    #         [0., 1.]])
    # x = ar([2.])
    x = linspace(-10, 10)
    l = ones_like(x)

    multiply, multiply_jvps = autodiff_rules['multiply']
    y, gy0 = autodiff(multiply, multiply_jvps, l, x, x)
    print(y)
    print(gy0)
    # y, gy1 = autodiff(multiply, multiply_jvps, gy0, y, w)
    # print(y)
    # print(gy1)

    # mult = ES('ij,jk->ik')
    # y, gy = mult(l, w, x)

    # print(w)
    # print(y)
    # print(l)
    # print(gy)
    # α = 3e-4
    # w -= α * gy
    # print(w)

if __name__ == '__main__':
    main()
