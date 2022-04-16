import numpy as np
from primitives import autodiff_rules
from tracer     import Tracer

def autodiff(f, jvps, l, args):
    ''' Simple functional autodifferentiation:
        f     : any function, such as np.add
        jvps  : the Jacobian vector products for f as a tuple of functions
        l     : incoming gradient (l for "loss")
        args : remaining arguments for f and grad(f)
        '''
    print(f, jvps, l, args)
    y = f(*args) # Compute normal function
    grads = tuple(jvp(l, y, *args) for jvp in jvps)
    return y, grads

def apply(name, grads, args):
    func, func_jvps = autodiff_rules[name]
    return autodiff(func, func_jvps, grads, args)

def backprop(topology, in_grad, args):
    print('Backprop:', topology, in_grad, args)
    if isinstance(topology, Tracer):
        topology = topology.topology
    if not isinstance(topology, dict): # Currently implies a constant
        return topology, 1.
    items = topology.items()
    assert len(items) == 1, 'For now :)'
    name, arg_tracers = list(items)[0]
    if name == 'input':
        return args, in_grad
    else:
        inputs, grads = zip(*(backprop(arg_tracer, in_grad, args)
                              for arg_tracer in arg_tracers))
        grad = np.sum(np.array(grads), axis=0)
        y, grads = apply(name, grad, inputs)
        assert len(grads) == len(arg_tracers), f'Must have the same number of tracers and gradients! Got {len(grads)} and {len(arg_tracers)}'
        return y, grad


