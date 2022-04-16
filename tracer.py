from primitives import autodiff_rules # Table of JVPs
from inspect    import signature

def trace(f):
    ''' Use the Tracer object (below) to build up the differentiable graph '''
    inx = Tracer()
    sig = signature(f)
    out = f(*(inx,) * len(sig.parameters))
    return out.topology

class Tracer:
    ''' Class that acts like an ndarray but instead builds up the differentiable
        graph of operations '''
    def __init__(self, topology=None):
        if topology is None:
            self.topology = dict()
        else:
            self.topology = topology

    def lbinary(self, name, other):
        if isinstance(other, Tracer):
            return Tracer({name : (self.topology, other.topology)})
        else:
            return Tracer({name : (self.topology, other)})

    def rbinary(self, name, other):
        if isinstance(other, Tracer):
            return Tracer({name : (other.topology, self.topology)})
        else:
            return Tracer({name : (other, self.topology)})

    def unary(self, name):
        return Tracer({name : (self.topology,)})

    def __abs__(self):             return self.unary('abs')
    def __neg__(self):             return self.unary('negative')

    def __add__(self, other):      return self.lbinary('add', other)
    def __sub__(self, other):      return self.lbinary('subtract', other)
    def __mul__(self, other):      return self.lbinary('multiply', other)
    def __pow__(self, other):      return self.lbinary('power', other)
    def __div__(self, other):      return self.lbinary('divide', other)
    def __truediv__(self, other):  return self.lbinary('true_divide', other)

    def __radd__(self, other):     return self.rbinary('add', other)
    def __rsub__(self, other):     return self.rbinary('subtract', other)
    def __rmul__(self, other):     return self.rbinary('multiply', other)
    def __rpow__(self, other):     return self.rbinary('power', other)
    def __rdiv__(self, other):     return self.rbinary('divide', other)
    def __rtruediv__(self, other): return self.rbinary('true_divide', other)

def _unary(name):
    def inner(t):
        t.topology = {name : t.topology}
        return t
    return inner

exp  = _unary('exp')
log  = _unary('log')
tanh = _unary('tanh')
sinh = _unary('sinh')
cosh = _unary('cosh')
