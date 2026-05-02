import dash_bootstrap_components as dbc
from dash import html

def criar_kpi_card(titulo, valor):
    return dbc.Card([
        dbc.CardBody([
            html.H6(titulo, className="text-muted text-center"),
            html.H3(valor, className="text-center font-weight-bold")
        ])
    # Usando a sua classe CSS junto com a sombra do Bootstrap
    ], className="meu-card-kpi shadow-sm mb-4")