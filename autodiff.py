import numpy as np
from primitives import autodiff_rules
from tracer     import Tracer

def autodiff(name, i, l, args):
    f, jvps = autodiff_rules[name]
    ''' Simple functional autodifferentiation:
        f     : any function, such as np.add
        jvps  : the Jacobian vector products for f as a tuple of functions
        i     : index of jvp
        l     : incoming gradient (l for "loss")
        args : remaining arguments for f and grad(f)
        '''
    y = f(*args)
    grad = jvps[i](l, y, *args)
    return y, grad

def backprop(topology, in_grad, args):
    print('Backprop:', topology)
    if isinstance(topology, Tracer):
        topology = topology.topology
    if not isinstance(topology, dict): # Currently implies a constant
        return topology, np.array([0.]) # Constant rule
    items = topology.items()
    assert len(items) == 1, 'For now :)'
    name, arg_tracers = list(items)[0]
    if name == 'input': # Recurse to base case, pass input and grad from inputs
        return args[arg_tracers], in_grad
    elif name == 'einsum':
        spec, *arg_tracers = arg_tracers
    # Recursively apply backprop to incoming items
    # Ex: {multiply : (a, b)} will call backprop on a and b
    inputs, grads = zip(*(backprop(arg_tracer, in_grad, args)
                          for arg_tracer in arg_tracers))
    if name == 'einsum':
        current_args = (spec,) + inputs # Small hack to pass around einsum str
    else:
        current_args = inputs
    # Sum the gradients and return them (requires multiple evals)
    all_grads = []
    for i, (inp, prev_grad) in enumerate(zip(inputs, grads)):
        y, grad = autodiff(name, i, prev_grad, current_args)
        all_grads.append(grad)
    final_grad = np.sum(all_grads, axis=0)
    return y, final_grad


