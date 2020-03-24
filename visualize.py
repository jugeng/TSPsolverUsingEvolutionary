import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import plotly
import plotly.graph_objects as go
import plotly.io as pio

import numpy as np
from datetime import datetime
import sys

import matplotlib.pyplot as plt

generation_fitness =[]
f_name = "./logs/visualize_data.txt"

if(len(sys.argv) == 2):
    f_name = "./logs/"+ sys.argv[1] +".txt"
    print(f_name)

with open(f_name, "r") as f:
    fitness = f.readline()
    fitness_curve = list(float(item) for item in fitness.split())
    for item in f:
        fit = item.split()
        tx =  list(float(item) for item in fit)
        generation_fitness.append(tx)

t = len(fitness_curve)
print("Data successfully added")


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(
    html.Div([
        html.H3('Genetic Algorithm Visualization'),
        html.H5('Generations Evolved {}'.format(t)),
        html.H5('Minimum Distance {:.2f}'.format(min(fitness_curve))),
        dcc.Graph(id='fitness-graph') ,
        dcc.Slider(
            id='gen-slider',
            min=0,
            max= t,
            step=1,
            value = 1
        ),
        dcc.Graph(id='generation-graph')
        
    ])
)

r = float(max(generation_fitness[0])) + 0.005

@app.callback([Output('generation-graph', 'figure'), Output('fitness-graph', 'figure')], [Input('gen-slider', 'value')])
def update_gen_graph(v):

    m = max(generation_fitness[v])
    colors = ['lightslategray'] * t

    for i, val in enumerate(generation_fitness[v]):
        if val == m:
            colors[i] =  'crimson'

    data = go.Bar(y=generation_fitness[v],x = np.arange(t), width=0.2, marker_color = colors)
    layout = go.Layout(
        title= 'Generation:{}'.format(v), 
        yaxis = dict(
            range = [0,r ], 
            title = 'Fitness'))

    

    data_1 = [go.Scatter(y = fitness_curve, x = np.arange(t), marker = dict(line = dict(color = 'crimson')) , showlegend = False), 
        go.Scatter(x = [v], y = [fitness_curve[v]], mode='markers' )]
   
    layout_1 = go.Layout(title = "Fitness Curve",width=800, height=500, autosize=False)
    
    graph_1 = go.Figure(data=data, layout=layout)
    graph_2 = go.Figure(data=data_1,  layout = layout_1)

    return (graph_1,graph_2)



if __name__ == '__main__':
    app.run_server(debug=True)
