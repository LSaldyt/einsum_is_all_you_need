import plotly.graph_objects as go
import plotly.express as px
import plotly
import pandas as pd

from utils import *

from numpy import *
from numpy import array       as ar
from numpy import squeeze     as sz
from numpy import expand_dims as ed

from einsum     import *
from primitives import *
from autodiff   import *

def main():
    n = 1000
    x  = linspace(-10, 10, n)
    l = ones_like(x)
    func = 'sinh'
    func, func_jvps = autodiff_rules[func]
    y, gy = autodiff(func, func_jvps, l, x)
    fig = px.line(x=x, y=y)
    fig.show()
    fig = px.line(x=x, y=gy[0])
    fig.show()

if __name__ == '__main__':
    main()
