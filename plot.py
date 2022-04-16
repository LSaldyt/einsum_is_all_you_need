import plotly.graph_objects as go
import plotly.express as px
import plotly
import pandas as pd
from network_graph import nx, plot_g

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

def get_unique_key(k, seen):
    i = 0
    sub_k = f'{k}_{i}'
    while sub_k in seen:
        sub_k = f'{k}_{i}'
        i += 1
    seen.add(sub_k)
    return sub_k

def create_graph(topology, G=None, seen=None):
    if not isinstance(topology, dict):
        return topology
    if G is None:
        G = nx.MultiGraph()
    if seen is None:
        seen = set()
    for k, v in topology.items():
        k = get_unique_key(k, seen)
        if isinstance(v, tuple):
            for sv in v:
                H = create_graph(sv, seen=seen)
                if isinstance(H, nx.MultiGraph):
                    G = nx.compose(G, H)
                if isinstance(sv, dict):
                    for sk in sv:
                        sk = get_unique_key(sk, seen)
                        G.add_edge(k, sk)
    return G

def plot_graph(topology, name='topology'):
    G = create_graph(topology)
    pos = nx.kamada_kawai_layout(G)
    pprint(pos)
    for node in G:
        G.nodes[node]['pos'] = pos[node]
        G.nodes[node]['text'] = node
    fig = plot_g(G)
    save(fig, name, w=2400, h=920)
    return fig

def sigmoid(x):
    return 1 / (1 + exp(-x))

def test(x):
    return 1 / (1 + exp(-x))

def logistic(w, x):
    h = es('ij,jk->ik', w, x)
    return sigmoid(h)

def softmax(x):
    top  = exp(x)
    norm = es('ik->i', top)
    return top / norm

def sqrt(x):
    return x ** 0.5

def attention(keys, values, queries):
    # Reference: http://einops.rocks/pytorch-examples.html
    c = 2 # Hardcoded for now
    attention = es('ct,cl->tl', keys, values)
    attention = softmax(attention / sqrt(c)) # Scaled dot product attn
    lookup    = es('ct,tl->cl', values, attention)
    return lookup

def main():
    test_trace = trace(test)
    # test_trace = trace(attention)
    pprint(test_trace.topology)
    plot_graph(test_trace.topology, 'attention_topology')
    1/0

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
