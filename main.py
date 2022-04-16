from numpy import *
from numpy import array       as ar
from numpy import squeeze     as sz
from numpy import expand_dims as ed

from einsum     import *
from primitives import *
from autodiff   import *

def apply(name, *args):
    func, func_jvps = autodiff_rules[name]
    return autodiff(func, func_jvps, *args)

def sigmoid(l, x):
    ''' Typically:
        def sigmoid(x):
            return 1 / (1 + exp(-x)) '''
    mid, gm = apply('negative', l,  x)
    mid, gm = apply('exp',      gm, mid)
    mid, gm = apply('add',      gm, 1., mid)
    mid, gm = apply('divide',   gm, 1., mid)
    return mid, gm

def mean_squared_error(l, x):
    ''' Typically:
        def mse(y, yh):
            return (y-yh)**2 / sum((y-yh)**2) '''

    sq, gsq = apply('multiply', l, x, x)
    sm = np.sum(sq)
    mid, gm = apply('divide', l, sq, sm)
    return mid, gm

def main():
    x = ar([[2., 2.],
            [2., 2.]])
    w = ar([[0.1, 0.],
            [0.,  0.1]])

    yt = ar([[2., 2.],
             [2., 2.]])

    l = ones_like(x)

    mult = ES('ij,jk->ik')
    y, gy = mult(l, w, x)
    y, gy = mean_squared_error(gy, y)
    y, gy = apply('subtract', gy[0], y, yt)
    print(y)
    l = y
    y, gy = mult(l, w, x)
    y, gy = mean_squared_error(gy, y)
    y, gy = apply('subtract', gy[0], y, yt)
    print(y)

    # print(gy)

    # print(w)
    # print(y)
    # print(l)
    # print(gy)
    α = 3e-4
    α = 1.
    w -= α * gy[1]

if __name__ == '__main__':
    main()
