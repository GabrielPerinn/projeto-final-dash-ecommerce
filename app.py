import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html
from sklearn.linear_model import LinearRegression
import numpy as np

# ============================================================
# 1. CARREGAR DADOS
# ============================================================
df = pd.read_csv('/Users/mimobilrierablanca/Desktop/ecommerce_estatistica.csv')

# ============================================================
# 2. CRIAR GRÁFICOS COM PLOTLY
# ============================================================

# Gráfico 1: Histograma
fig_histograma = px.histogram(df, x='Nota', nbins=20,
                              title='Distribuição de Notas dos Produtos',
                              labels={'Nota': 'Nota (0-5)', 'count': 'Frequência'})

# Gráfico 2: Dispersão
fig_dispersao = px.scatter(df, x='Preço', y='Nota',
                           title='Correlação entre Preço e Nota',
                           labels={'Preço': 'Preço (R$)', 'Nota': 'Nota (0-5)'})

# Gráfico 3: Mapa de Calor
cols = ['Nota', 'N_Avaliações', 'Desconto', 'Preço']
corr_matrix = df[cols].corr()
fig_heatmap = go.Figure(data=go.Heatmap(z=corr_matrix.values,
                                        x=corr_matrix.columns,
                                        y=corr_matrix.columns,
                                        colorscale='RdBu',
                                        zmid=0))
fig_heatmap.update_layout(title='Matriz de Correlação')

# Gráfico 4: Barras (Top 10 Marcas)
top_marcas = df['Marca'].value_counts().head(10).reset_index()
top_marcas.columns = ['Marca', 'Quantidade']
fig_barras = px.bar(top_marcas, x='Marca', y='Quantidade',
                    title='Top 10 Marcas por Quantidade',
                    labels={'Marca': 'Marca', 'Quantidade': 'Quantidade'})
fig_barras.update_xaxes(tickangle=45)

# Gráfico 5: Pizza
genero_dist = df['Gênero'].value_counts().reset_index()
genero_dist.columns = ['Gênero', 'Quantidade']
fig_pizza = px.pie(genero_dist, names='Gênero', values='Quantidade',
                   title='Distribuição de Produtos por Gênero')

# Gráfico 6: Densidade
fig_densidade = px.histogram(df, x='Preço', nbins=50, marginal='rug',
                             title='Densidade de Preços',
                             labels={'Preço': 'Preço (R$)', 'count': 'Frequência'})

# Gráfico 7: Regressão
data_reg = df[['N_Avaliações', 'Nota']].dropna()
X = data_reg['N_Avaliações'].values.reshape(-1, 1)
y = data_reg['Nota'].values

modelo = LinearRegression()
modelo.fit(X, y)
y_pred = modelo.predict(X)

fig_regressao = go.Figure()
fig_regressao.add_trace(go.Scatter(x=X.flatten(), y=y, mode='markers',
                                   name='Dados Reais',
                                   marker=dict(color='blue', size=8, opacity=0.5)))
fig_regressao.add_trace(go.Scatter(x=X.flatten(), y=y_pred, mode='lines',
                                   name='Linha de Regressão',
                                   line=dict(color='red', width=2)))
fig_regressao.update_layout(title='Regressão Linear: Avaliações vs Nota',
                            xaxis_title='N° Avaliações',
                            yaxis_title='Nota')

# ============================================================
# 3. CRIAR APLICAÇÃO DASH
# ============================================================

app = Dash(__name__)

app.layout = html.Div(
    children=[
        # Header
        html.Div(
            children=[
                html.H1("Dashboard E-commerce - Análise de Dados",
                        style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),
                html.P("Visualização interativa dos dados de e-commerce",
                       style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': 16})
            ],
            style={'padding': '30px', 'backgroundColor': '#ecf0f1', 'marginBottom': '30px', 'borderRadius': '10px'}
        ),

        # Grid de gráficos
        html.Div(
            children=[
                # Linha 1
                html.Div([
                    html.Div([dcc.Graph(figure=fig_histograma)], style={'flex': '1', 'padding': '10px'}),
                    html.Div([dcc.Graph(figure=fig_dispersao)], style={'flex': '1', 'padding': '10px'}),
                    html.Div([dcc.Graph(figure=fig_heatmap)], style={'flex': '1', 'padding': '10px'}),
                ], style={'display': 'flex', 'flexWrap': 'wrap'}),

                # Linha 2
                html.Div([
                    html.Div([dcc.Graph(figure=fig_barras)], style={'flex': '1', 'padding': '10px'}),
                    html.Div([dcc.Graph(figure=fig_pizza)], style={'flex': '1', 'padding': '10px'}),
                    html.Div([dcc.Graph(figure=fig_densidade)], style={'flex': '1', 'padding': '10px'}),
                ], style={'display': 'flex', 'flexWrap': 'wrap'}),

                # Linha 3
                html.Div([
                    html.Div([dcc.Graph(figure=fig_regressao)], style={'flex': '1', 'padding': '10px'}),
                ], style={'display': 'flex', 'flexWrap': 'wrap'}),
            ],
            style={'padding': '20px'}
        ),

        # Footer
        html.Div(
            children=[
                html.P("Dashboard criado com Dash e Plotly",
                       style={'textAlign': 'center', 'color': '#95a5a6', 'marginTop': '30px'})
            ]
        )
    ],
    style={'fontFamily': 'Arial, sans-serif', 'margin': '0', 'padding': '20px', 'backgroundColor': '#f5f5f5'}
)

# ============================================================
# 4. RODAR A APLICAÇÃO
# ============================================================

if __name__ == '__main__':
    app.run(debug=False)
