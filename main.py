from pprint import pprint

from numpy import array       as ar
from numpy import squeeze     as sz
from numpy import expand_dims as ed

# from einsum     import *
from primitives import *
from autodiff   import autodiff, apply, backprop
from tracer     import *

def sigmoid(x):
    return 1 / (1 + exp(-x))

def test(x):
    return x**2 + x**2

def main():
    sig_trace = trace(sigmoid)
    pprint(sig_trace.topology)
    test_trace = trace(test)
    pprint(test_trace.topology)
    backprop(sig_trace, ar([1.]), ar([1.]))
    1/0

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
    # y, gy = apply('subtract', gy[0], y, yt)
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
