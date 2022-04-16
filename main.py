from pprint import pprint

import numpy as np
np.seterr(all='raise')
from numpy import array       as ar
from numpy import squeeze     as sz
from numpy import expand_dims as ed

# from einsum     import *
from primitives import *
from autodiff   import autodiff, backprop
from tracer     import *


def sigmoid(x):
    return 1 / (1 + exp(-x))

def test(w, x):
    h = es('ij,jk->ik', w, x)
    return 1 / (1 + exp(-h))

def main():
    sig_trace = trace(sigmoid)
    pprint(sig_trace.topology)
    y, gy = backprop(sig_trace, ar([1.]), (ar([1.]),))

    test_trace = trace(test)
    pprint(test_trace.topology)
    x = ar([[1., 0.], [0., 0.]])
    w = ar([[1., 0.], [0., 0.]])
    y, gy = backprop(test_trace, np.ones_like(x), (w, x))
    print(y, gy)

    # x = ar([[2., 2.],
    #         [2., 2.]])
    # w = ar([[0.1, 0.],
    #         [0.,  0.1]])

    # yt = ar([[2., 2.],
    #          [2., 2.]])

    # l = ones_like(x)

    # mult = ES('ij,jk->ik')
    # y, gy = mult(l, w, x)
    # y, gy = mean_squared_error(gy, y)
    # print(y)

    # # print(gy)

    # # print(w)
    # # print(y)
    # # print(l)
    # # print(gy)
    # α = 3e-4
    # α = 1.
    # w -= α * gy[1]

if __name__ == '__main__':
    main()
