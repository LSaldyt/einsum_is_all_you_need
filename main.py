from pprint import pprint

import numpy as np
np.seterr(all='raise')
from numpy import array       as ar
from numpy import squeeze     as sz
from numpy import expand_dims as ed

from autodiff   import autodiff, backprop
from primitives import *
from tracer     import *

def sigmoid(x):
    return 1 / (1 + exp(-x))

def logistic(w, x):
    h = es('ij,jk->ik', w, x)
    return sigmoid(h)

def main():
    test_trace = trace(logistic)
    pprint(test_trace.topology)
    x = ar([[1., 0.],
            [0., 0.]])
    w = ar([[1., 0.],
            [0., 0.]])
    for i in range(10000):
        y, gy = backprop(test_trace, np.ones_like(x), (w, x))
        print(y, gy)

        α = 3e-4
        w -= α * gy
        print(w)

if __name__ == '__main__':
    main()
