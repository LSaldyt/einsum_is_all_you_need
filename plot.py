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

def visualize(l, legend_title='', title='', filename='diff'):
    fig = go.Figure()
    width = 2; mode = 'lines'
    for yi, (name, x, y) in enumerate(l):
        fig.add_trace(go.Scatter(x=x, y=y, mode=mode,
            name=name, line=dict(color=QUAL_COLORS[yi], width=width)))
    fig.update_layout(font_size=32, xaxis_title='X',
        yaxis_title='Y', title_text=title)
    fig.update_layout(legend=dict(yanchor='top', y=1.1,
                                  xanchor='center', x=0.5,
                                  orientation='h',
                                  font_size=32),
                      legend_title_text=legend_title)
    save(fig, filename, w=1400, h=700)
    return fig

def sigmoid(x):
    return 1 / (1 + exp(-x))

def test(x):
    return 1 / (1 + exp(-x))

def main():
    test_trace = trace(test)
    pprint(test_trace.topology)

    n = 1000
    x = np.linspace(-10, 10, n)
    l = np.ones_like(x)
    y, gy = backprop(test_trace, l, (x,))


    visualize([('Sigmoid', x, y),
               ('Sigmoid derivative', x, gy)])

    # print(y)
    # print(gy)

    # fig = px.line(x=x, y=y)
    # fig.show()
    # fig = px.line(x=x, y=gy)
    # fig.show()

if __name__ == '__main__':
    main()
