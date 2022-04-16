import plotly.graph_objects as go
import plotly.express as px
import plotly
import pandas as pd

from utils import *

from pprint import pprint

import numpy as np
np.seterr(all='raise')
from numpy import array       as ar
from numpy import squeeze     as sz
from numpy import expand_dims as ed

# from einsum     import *
from primitives import *
from autodiff   import autodiff, backprop
from tracer     import *

def sigmoid(x):
    return 1 / (1 + exp(-x))

def test(x):
    return 1 / (1 + exp(-x))

def test(x):
    # return x**2 + x**2
    # return exp(x)
    return x**2
    # return sinh(x)
    # return x + 2

def main():
    test_trace = trace(test)
    pprint(test_trace.topology)

    n = 10
    x = np.linspace(-10, 10, n)
    l = np.ones_like(x)
    y, gy = backprop(test_trace, l, x)

    print(gy)

    fig = px.line(x=x, y=y)
    fig.show()
    fig = px.line(x=x, y=gy)
    fig.show()

if __name__ == '__main__':
    main()
