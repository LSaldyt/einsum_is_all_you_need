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

def softmax(x):
    top  = exp(x)
    norm = es('ik->i', top)
    return top / norm

def sqrt(x):
    return x ** 0.5

def attention(keys, values, queries):
    # Reference: http://einops.rocks/pytorch-examples.html
    c = 2 # Hardcoded for now
    attention = es('ct,cl->tl', keys, values)
    attention = softmax(attention / sqrt(c)) # Scaled dot product attn
    lookup    = es('ct,tl->cl', values, attention)
    return lookup

def main():
    # test_trace = trace(logistic)
    # test_trace = trace(softmax)
    test_trace = trace(attention)
    pprint(test_trace.topology)
    1/0
    x = ar([[1., 0.],
            [0., 0.]])
    w = ar([[1., 0.],
            [0., 0.]])
    for i in range(100):
        y, gy = backprop(test_trace, np.ones_like(x), (w, x))
        print(y, gy)

        α = 3e-4
        w -= α * gy
        print(w)

if __name__ == '__main__':
    main()
