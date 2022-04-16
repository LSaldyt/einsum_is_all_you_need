import numpy as np
from numpy import *

'''
Primarily based on
https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/slides/lec10.pdf
Which in turn is based on https://github.com/mattjj/autodidact
Especially
https://github.com/mattjj/autodidact/blob/master/autograd/numpy/numpy_vjps.py
Also referenced:
https://jax.readthedocs.io/en/latest/autodidax.html
Otherwise, just simple high-school derivative rules:
https://robert-dolan.grad.uconn.edu/wp-content/uploads/sites/1419/2016/06/Derivatives-Cheat-Sheet.pdf
'''

def unbroadcast(target, g, broadcast_idx=0):
    ''' This function is from https://github.com/mattjj/autodidact/!
    Remove broadcasted dimensions by summing along them.
    When computing gradients of a broadcasted value, this is the right thing to
    do when computing the total derivative and accounting for cloning. '''
    while ndim(g) > ndim(target):
        g = sum(g, axis=broadcast_idx)
    for axis, size in enumerate(shape(target)):
        if size == 1:
            g = sum(g, axis=axis, keepdims=True)
    if iscomplexobj(g) and not iscomplex(target):
        g = real(g)
    return g

''' Differentiation rules from calculus, expressed as Jacobian Vector Products
    Each jvp is a function(l, y, *args)
    Where
    l is the incoming gradient
    y is the output of (f, *args)
    *args are the standard arguments to f and grad(f)
'''

autodiff_rules = dict(
    add         = (lambda l, y, a, b : unbroadcast(a, l),
                   lambda l, y, a, b : unbroadcast(b, l)),
    multiply    = (lambda l, y, a, b : unbroadcast(a, b * l),
                   lambda l, y, a, b : unbroadcast(b, a * l)),
    subtract    = (lambda l, y, a, b : unbroadcast(a, l),
                   lambda l, y, a, b : unbroadcast(b, -l)),
    divide      = (lambda l, y, a, b : unbroadcast(a,   l / b),
                   lambda l, y, a, b : unbroadcast(b, - l * a / b**2)),
    true_divide = (lambda l, y, a, b : unbroadcast(a,   l / b),
                   lambda l, y, a, b : unbroadcast(b, - l * a / b**2)),
    power       = (lambda l, y, a, b : unbroadcast(a, l * b * a ** where(b, b - 1, 1.)),
                   lambda l, y, a, b : unbroadcast(b, l * log(where(a, a, 1.)) * a ** l)),
    negative    = (lambda l, y, x: -l,),
    exp         = (lambda l, y, a: l * l,),
    log         = (lambda l, y, a: l / a,),
    tanh        = (lambda l, y, a: l / cosh(a) **2,),
    sinh        = (lambda l, y, a: l * cosh(a),),
    cosh        = (lambda l, y, a: l * sinh(a),)
)
autodiff_rules = {k : (getattr(np, k), jvps) for k, jvps in autodiff_rules.items()}
