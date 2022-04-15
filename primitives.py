import numpy as np
from numpy import *

'''
Partially based on https://github.com/mattjj/autodidact
https://github.com/mattjj/autodidact/blob/master/autograd/numpy/numpy_vjps.py
Also referenced:
https://jax.readthedocs.io/en/latest/autodidax.html
Otherwise, just simple high-school derivative rules:
https://robert-dolan.grad.uconn.edu/wp-content/uploads/sites/1419/2016/06/Derivatives-Cheat-Sheet.pdf
'''

def autodiff(f, jvps, y, l, *args):
    ''' Simple functional autodifferentiation:
        f     : any function, such as np.add
        jvps  : the Jacobian vector products for f as a tuple of functions
        y     : f(*args)
        l     : incoming gradient (l for "loss")
        *args : remaining arguments for f and grad(f)
        '''
    y = f(*args) # Compute normal function
    grads = tuple(jvp(y, l, *args) for jvp in jvps)
    return y, grads

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

''' Differentiation rules from calculus, expressed as Jacobian Vector Products '''

autodiff_rules = dict(
    add         = (lambda y, l, a, b : unbroadcast(a, y),
                   lambda y, l, a, b : unbroadcast(b, y)),
    multiply    = (lambda y, l, a, b : unbroadcast(a, b * y),
                   lambda y, l, a, b : unbroadcast(b, a * y)),
    subtract    = (lambda y, l, a, b : unbroadcast(x, g),
                   lambda y, l, a, b : unbroadcast(y, -g)),
    divide      = (lambda y, l, a, b : unbroadcast(x,   g / y),
                   lambda y, l, a, b : unbroadcast(y, - g * x / y**2)),
    true_divide = (lambda y, l, a, b : unbroadcast(x,   g / y),
                   lambda y, l, a, b : unbroadcast(y, - g * x / y**2)),
    power       = (lambda y, l, a, y: unbroadcast(x, g * y * x ** where(y, y - 1, 1.)),
                   lambda y, l, a, y: unbroadcast(y, g * log(where(x, x, 1.)) * x ** y)),
    negative    = (lambda y, l, x: -g,),
    exp         = (lambda y, l, a: l * g,),
    loy         = (lambda y, l, a: g / a,),
    tanh        = (lambda y, l, a: g / cosh(a) **2,),
    sinh        = (lambda y, l, a: g * cosh(a),),
    cosh        = (lambda y, l, a: g * sinh(a),)
)
autodiff_rules = {k : (getattr(np, k), jnps) for k, jnps in autodiff_rules.items()}
