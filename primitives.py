import numpy  as np
import einsum as einsum

'''
Differentiation rules from calculus, expressed as Jacobian Vector Products
    Each jvp is a function(l, y, *args)
    Where:
        l is the incoming gradient
        y is the output of (f, *args)
        *args are the standard arguments to f and grad(f)
If we wanted to allow taking multiple gradients, we would replace np.where and np.log with our versions of log, where, and so on.

Primarily based on
https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/slides/lec10.pdf
Which in turn is based on https://github.com/mattjj/autodidact
Especially
https://github.com/mattjj/autodidact/blob/master/autograd/numpy/numpy_vjps.py
Note that I have made several simplifications/fixes relative to the autodidact jvps & tracing
Also referenced:
https://jax.readthedocs.io/en/latest/autodidax.html
Otherwise, just simple high-school derivative rules:
https://robert-dolan.grad.uconn.edu/wp-content/uploads/sites/1419/2016/06/Derivatives-Cheat-Sheet.pdf
'''

autodiff_rules = dict(
    add         = (lambda l, y, a, b : l,
                   lambda l, y, a, b : l),
    multiply    = (lambda l, y, a, b : b * l,
                   lambda l, y, a, b : a * l),
    subtract    = (lambda l, y, a, b : l,
                   lambda l, y, a, b : -l),
    divide      = (lambda l, y, a, b :   l / b,
                   lambda l, y, a, b : - l * a / b**2),
    true_divide = (lambda l, y, a, b :   l / b,
                   lambda l, y, a, b : - l * a / b**2),
    power       = (lambda l, y, a, b : l * b * a ** np.where(b, b - 1, 1.),
                   lambda l, y, a, b : l * np.log(np.where(a > 0, a, 1.)) * a ** b),
    negative    = (lambda l, y, x: -l,),
    exp         = (lambda l, y, a: y * l,),
    log         = (lambda l, y, a: l / a,),
    tanh        = (lambda l, y, a: l / np.cosh(a) **2,),
    sinh        = (lambda l, y, a: l * np.cosh(a),),
    cosh        = (lambda l, y, a: l * np.sinh(a),),
    einsum      = (einsum.jacobian_vector_product,) * 2
)
autodiff_rules = {k : (getattr(np, k), jvps) for k, jvps in autodiff_rules.items()}
