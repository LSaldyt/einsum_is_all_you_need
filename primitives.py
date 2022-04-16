import numpy as np

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
    while np.ndim(g) > np.ndim(target):
        g = np.sum(g, axis=broadcast_idx)
    for axis, size in enumerate(np.shape(target)):
        if size == 1:
            g = np.sum(g, axis=axis, keepdims=True)
    if np.iscomplexobj(g) and not np.iscomplex(target):
        g = np.real(g)
    return g

''' Differentiation rules from calculus, expressed as Jacobian Vector Products
    Each jvp is a function(l, y, *args)
    Where
    l is the incoming gradient
    y is the output of (f, *args)
    *args are the standard arguments to f and grad(f)

    If we wanted to allow taking multiple gradients, we would replace np.where and np.log with our versions of log, where, and so on.
'''

def power_jnp_debug(l, y, a, b):
    pre = np.where(a>0, a, 1.)
    mid = np.log(pre)
    return unbroadcast(b, l * mid * a ** b)

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
    power       = (lambda l, y, a, b : unbroadcast(a, l * b * a ** np.where(b, b - 1, 1.)),
                   power_jnp_debug),
                   # lambda l, y, a, b : unbroadcast(b, l * np.log(np.where(a, a, 1.)) * a ** b)),
    negative    = (lambda l, y, x: -l,),
    exp         = (lambda l, y, a: y * l,),
    log         = (lambda l, y, a: l / a,),
    tanh        = (lambda l, y, a: l / np.cosh(a) **2,),
    sinh        = (lambda l, y, a: l * np.cosh(a),),
    cosh        = (lambda l, y, a: l * np.sinh(a),)
)
autodiff_rules = {k : (getattr(np, k), jvps) for k, jvps in autodiff_rules.items()}
