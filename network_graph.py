import plotly.graph_objects as go
import networkx as nx
from utils import *
import white_theme
from pprint import pprint


def plot_g(G):
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(1-y0)
        edge_y.append(1-y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1., color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(1-y)

    node_text = []
    node_depths = []
    first = None
    for node in G:
        if first is None:
            first = node
        node_text.append(G.nodes[node]['text'])
        # Inefficient but these are small graphs
        try:
            node_depths.append(nx.shortest_path_length(G, source=first, target=node))
        except:
            node_depths.append(1)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        text=node_text,
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',
            reversescale=True,
            color=node_depths,
            size=200,
            colorbar=dict(
                thickness=15,
                title='Depth',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    annotations = []
    # annotations=[
    #     dict(
    #     text='',
    #     showarrow=False,
    #     xref="paper", yref="paper",
    #     x=0.005, y=-0.002 )]

    for x, y, t in zip(node_x, node_y, node_text):
        color = 'black'
        annotations.append(dict(
            x=x, y=y, text=t,
            xanchor='center',
            yanchor='middle',
            align='center',
            xref='x',
            yref='y',
            font=dict(color=color, size=40),
            showarrow=False, arrowhead=1
            ))

    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title='',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    annotations=annotations,
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig

# G = nx.random_geometric_graph(200, 0.125)
# names = ['a'] * 100 + ['b'] * 100
# nx.set_node_attributes(G, {k : {'text' : n} for k, n in enumerate(names)})
# plot_g(G)
