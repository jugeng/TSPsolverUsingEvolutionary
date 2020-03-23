import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np


generation_fitness =[]

with open("./logs/visualize_data.txt", "r") as f:
    fitness = f.readline()
    fitness_curve = fitness.split()
    for item in f:
        fit = item.split()
        generation_fitness.append(fit)

t = len(fitness_curve)

#fit_fig = go.Figure(data=[go.Scatter(y = fitness_curve, x = np.arange(t)), go.Scatter(x = v, y = fitness_curve[x])],  layout =  {'title': 'Fitness Evolution'})

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(
    html.Div([
        html.H3('Genetic Algorithm Visualization'),
        html.H4('Generations Evolved {}'.format(t)),
        html.H4('Minimum Distance {:.2}'.format(min(fitness_curve))),
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
            range = [0,0.1], 
            title = 'Fitness'))

    

    data_1 = [go.Scatter(y = fitness_curve, x = np.arange(t)), 
        go.Scatter(x = [v], y = [fitness_curve[v]], mode='markers' )]
   
    layout_1 = go.Layout(title = "Fitness Curve")
    
    graph_1 = go.Figure(data=data, layout=layout)
    graph_2 = go.Figure(data=data_1,  layout = layout_1)

    
    return (graph_1,graph_2)



if __name__ == '__main__':
    app.run_server(debug=True)

#z = generation_fitness[v]