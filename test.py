from tradingbot_utils import get_stock_data
import coloredlogs
from tradingbot_utils import switch_k_backend_device
from tradingbot_agent import Agent
from tradingbot_methods import test_model
import dash
from dash.dependencies import Output, Input
import dash_core_components as dcc
import dash_html_components as html
import plotly
import random
import plotly.graph_objs as go
from collections import deque
import time

X = deque(maxlen=20)
X.append(0)
Y = deque(maxlen=20)
Y.append(0)
i = 1

app = dash.Dash(__name__)
app.layout = html.Div(
    [
        html.Div(id='output',style={'color': 'blue'}),
        dcc.Graph(id='live-graph', animate=True),
        dcc.Interval(
            id='graph-update',
            interval=1*1000
        ),
        
    ])


@app.callback(Output('live-graph', 'figure'),
              [Input('graph-update', 'n_intervals')])
def update_graph_scatter(input_data):
    global i
    if i>len(history):
        raise dash.exceptions.PreventUpdate
    elif i == len(history):
        data = plotly.graph_objs.Scatter(x=[i for i in range(len(history))],y=[x[0] for x in history],name='Scatter',mode= 'lines+markers')
        # print(data)
        i = i+1
        return {'data': [data],'layout' : go.Layout(xaxis=dict(range=[0,len(history)]),
                                                    yaxis=dict(range=[0,max([x[0] for x in history])]),)}
        
    else:
        X.append(i)
        Y.append(history[i-1][0])
        # print(history[i][0])
        data = plotly.graph_objs.Scatter(x=list(X),y=list(Y),name='Scatter',mode= 'lines+markers')
        i = i+1
        return {'data': [data],'layout' : go.Layout(xaxis=dict(range=[min(X),max(X)]),
                                                    yaxis=dict(range=[min(Y),max(Y)]),)}


@app.callback(Output(component_id='output', component_property='children'),
              [Input('graph-update', 'n_intervals')])
def update_value(input_data):
    global i
    if i >= len(history):
        return "Total profit is: ",history[len(history)-1][2]
    else:
        return 'Bot is going to {} and total profit right now is {}'.format(history[i-1][1], history[i-1][2])
    
@app.callback(Output("output","style"),
              [Input('graph-update', 'n_intervals')])
def update_style(input_data):
    global i
    if i >=len(history):
        if history[len(history)-1][2]>0:
            return {'color': 'green'}
        else:
            return {'color': 'red'}
    else:
        if history[i-1][2]>0:
            return {'color': 'green'}
        else:
            return {'color': 'red'}

def main(test_stock, window_size, strategy, model_path, debug=False):
    model_name = model_path.split("/")
    model_name = model_name[len(model_name)-1]
    agent = Agent(window_size, strategy=strategy, model_name=model_name, test=True, model_path=model_path)
    test_data = get_stock_data(test_stock)
    total_profit, history = test_model(agent, test_data,window_size=window_size)
    return total_profit, history


if __name__ == '__main__':
    test_stock = "test_GOOG_2019.csv"
    strategy = "dqn"
    window_size = 10
    model_path = "models/dqn_40"
    debug = False
    
    coloredlogs.install(level="DEBUG")
    switch_k_backend_device()


    total_profit, history = main(test_stock, window_size, strategy=strategy, model_path=model_path, debug=debug)
    app.run_server(debug=True)

