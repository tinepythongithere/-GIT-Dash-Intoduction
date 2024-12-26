from dash import dash, dcc, html, Input, Output, State, dash_table
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
from FDA import accuracy, conf_matrix, predict_species_with_proba

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

data = px.data.iris()

list_var = [col for col in data.columns if data[col].dtypes!='object']

#---------------------- DCC Components----------------------------

select_var = dcc.Dropdown(
    options=list_var,
    value='sepal_length',
    multi=False,
    id='var_id'
)

select_xvar = dcc.Dropdown(
    options=list_var,
    multi=False,
    value='sepal_length',
    placeholder='Select x axis',
    id='var_x'
)

select_yvar = dcc.Dropdown(
    options=list_var,
    multi=False,
    value='sepal_width',
    placeholder='Select y axis',
    id='var_y'
)

slider = dcc.Slider(
    min=0,
    max=20,
    marks={i: str(i) for i in range(21)},
    step=1,
    id='slider'
)
#-------------------- figure-------------------------
box_plot = px.box(data, x='petal_length')
hist = px.histogram(data, x='petal_length')

#--------------------Style css-------------------------
cardheaderstyle = {'background-color':'aqua', 'color':'black', 'text-align':'center'}
cardbodystyle = {'background-color':'Azure', 'color':'black', 'text-align':'center'}

#--------------------NavBar-----------------------------
nav_bar = html.Div([
    html.H4('Dashboard', style={'marginTop': 7, 'color': 'blue'}),
    dbc.Nav([
        dbc.NavLink(children='Stat desc', href='/', active='exact'),
        dbc.NavLink(children='FDA', href='/fatorial_analysis', active='exact'),
    ], pills=True),
    ], style={'display': 'flex', 'justify-content': 'space-between',
              'background-color': 'aqua', 'marginBottom': 10})

#------------------------------Layout--------------------------------------------
layout_homepage = html.Div([
    dbc.Row([
        dbc.Col(children=dbc.Card([
            dbc.CardHeader(children=select_var, style=cardheaderstyle),
            dbc.CardBody(children=dcc.Graph(figure=box_plot), style=cardbodystyle, id='box-plot')
        ])),

        dbc.Col(children=dbc.Card([
            dbc.CardHeader(children=slider, style=cardheaderstyle),
            dbc.CardBody(children=dcc.Graph(figure=hist, id='diag-bar'), style=cardbodystyle)
        ])),

    ]),

    dbc.Card([
        dbc.CardHeader([
            dbc.Row([
                dbc.Col(children=select_xvar),
                dbc.Col(children=select_yvar)
            ])
        ], style=cardheaderstyle),
        dbc.CardBody(children=dcc.Graph(id='nuage_de_points'), style=cardbodystyle)
    ],  style={'margin-top': '15px'}),

])

layout_fda = html.Div([
    dbc.Row([
       dbc.Col(dbc.Card([
           dbc.CardHeader(html.H5("Taux de bon classement"), style=cardheaderstyle),
           dbc.CardBody(html.H1(f"{round(accuracy * 100,2)}%"), style=cardbodystyle)
       ])),
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H5("Matrice de confusion"), style=cardheaderstyle),
            dbc.CardBody(dash_table.DataTable(conf_matrix.reset_index().to_dict(orient='records'),
                         ), style=cardbodystyle | {'display':'flex', 'justify-content':'center'})
        ]))
    ]),

    dbc.Card([
        dbc.CardHeader('Predictive modele', style=cardheaderstyle),
        dbc.CardBody([
            dbc.Row([
                dbc.Col('sepal length (cm)'),
                dbc.Col('sepal width (cm)'),
                dbc.Col('petal length (cm)'),
                dbc.Col('petal width (cm)')
            ]),
            dbc.Row([
                dbc.Col(dcc.Input(id='sepal length', type='number', value=6.5)),
                dbc.Col(dcc.Input(id='sepal width', type='number', value=3.3)),
                dbc.Col(dcc.Input(id='petal length', type='number', value=6.3)),
                dbc.Col(dcc.Input(id='petal width', type='number', value=7))
            ]),
            dbc.Button('Predict', style={'marginTop':5}, id='predict_btn'),

            html.Div(id='fda_predict', style={'marginTop':5})
        ]),

    ], style={'marginTop':5})
])


app.layout = html.Div(children=[
    nav_bar,
    dcc.Location(id='url'),
    html.Div(id='page-content', children=layout_homepage)
])

@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def render_page(path):
    if path=='/fatorial_analysis':
        return layout_fda
    elif path=='/':
        return layout_homepage
    else:
        return html.H1('Error 404: Page not found')

@app.callback(
    Output(component_id='box-plot', component_property='children'),
    Input(component_id='var_id', component_property='value')
)
def update_boxplot(value):
    output = px.box(data, x=value)

    return dcc.Graph(figure=output)

@app.callback(
    Output(component_id='nuage_de_points', component_property='figure'),
    Input(component_id='var_x', component_property='value'),
    Input(component_id='var_y', component_property='value')
)
def gen_scatter(varx, vary):
    nuage = px.scatter(data, x=varx, y=vary,color='species')

    return nuage

@app.callback(
    Output(component_id='diag-bar', component_property='figure'),
    [Input(component_id='slider', component_property='value'),
    Input(component_id='var_id', component_property='value')]
)
def update_hist(bins, xvar):
    output = px.histogram(data, x=xvar, nbins=bins)

    return output

@app.callback(
    Output(component_id='var_x', component_property='options'),
    Input(component_id='var_y', component_property='value')
)
def choice_filter(vary):

    output = list_var.copy()
    output.remove(vary)
    return output

@app.callback(
    Output(component_id='var_y', component_property='options'),
    Input(component_id='var_x', component_property='value')
)
def choice_filter(varx):
    output = list_var.copy()
    output.remove(varx)
    return output

@app.callback(
    Output('fda_predict', 'children'),
    Input('predict_btn', 'n_clicks'),
    State('sepal length', 'value'),
    State('sepal width', 'value'),
    State('petal length', 'value'),
    State('petal width', 'value')
)
def prediction(n_click, sl, sw, pl, pw):
    predicted_species, probabilities = predict_species_with_proba([sl, sw, pl, pw])
    data_bis = data.copy()

    data_bis.loc[len(data)]=[sl, sw, pl, pw, 'Predicted', '151']

    df_predict = pd.DataFrame({
        'Espèces': ['setosa', 'versicolor', 'virginica'],
        'Proba': [probabilities[0].round(3),probabilities[1].round(3),
                  probabilities[2].round(3)]})

    scatter_3D = px.scatter_3d(data_bis, x='sepal_width', y='sepal_length', z='petal_length', color='species')
    if n_click>0:
        return dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader(html.H4('Espèce prédite'), style=cardheaderstyle),
                dbc.CardBody(html.H4(predicted_species), style=cardbodystyle)
            ])),
            dbc.Col(dbc.Card([
                dbc.CardHeader(html.H4('Probabilité de prédiction'), style=cardheaderstyle),
                dbc.CardBody(html.H4(dash_table.DataTable(df_predict.to_dict(orient='records'))), style=cardbodystyle)
            ])),
            dbc.Col(dcc.Graph(figure=scatter_3D), width=6)
        ])

app.run_server(debug=False)