def autodiff(f, jvps, l, *args):
    ''' Simple functional autodifferentiation:
        f     : any function, such as np.add
        jvps  : the Jacobian vector products for f as a tuple of functions
        l     : incoming gradient (l for "loss")
        *args : remaining arguments for f and grad(f)
        '''
    y = f(*args) # Compute normal function
    grads = tuple(jvp(y, l, *args) for jvp in jvps)
    return y, grads

def backpropogate(structure, l):
    ''' TODO '''
    raise NotImplementedError
