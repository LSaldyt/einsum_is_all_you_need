import plotly.graph_objects as go
import plotly
from subprocess import call

QUAL_COLORS = plotly.colors.qualitative.G10 + plotly.colors.qualitative.Dark24 + plotly.colors.qualitative.Light24
DASH = ['dot', 'dash']

def save(fig, name, w=1920, h=1080, parentdir='data'):
    fig.update_layout(
        font=dict(family='Times', size=30, color='black'),
        legend=dict(
            orientation="h",
            yanchor='top',
            y=1.1,
            xanchor='center',
            x=0.5,
            bgcolor='white',
            bordercolor='black',
            borderwidth=1),)
    fig.show()
    fig.write_image(f'{parentdir}/{name}.svg', width=w, height=h)
    fig.write_image(f'{parentdir}/{name}.png', width=w, height=h)
    call(f'rsvg-convert -f pdf -o {parentdir}/{name}.pdf {parentdir}/{name}.svg',
         shell=True)

