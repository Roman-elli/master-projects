import dash_bootstrap_components as dbc
from dash import html, dcc

def criar_kpi_card(titulo, valor):
    """
    Gera um cartão padronizado para os KPIs principais no topo do dashboard.
    """
    return dbc.Card([
        dbc.CardBody([
            html.H6(titulo, className="text-muted text-center"),
            html.H3(valor, className="text-center font-weight-bold")
        ])
    ], className="meu-card-kpi shadow-sm mb-4")

def criar_seccao_resumo(texto_markdown):
    """
    Gera o cartão de (Storytelling) no fundo da página.
    """
    return dbc.Card([
        dbc.CardBody([
            html.H5("Summary", className="font-weight-bold mb-3", style={"color": "#202a35"}),
            html.Hr(style={"borderColor": "rgba(0,0,0,0.1)"}),
            
            # O dcc.Markdown permite renderizar negritos e formatação dinâmica gerada no App.py
            dcc.Markdown(
                texto_markdown, 
                className="text-muted mt-3", 
                style={"fontSize": "1.05rem", "lineHeight": "1.8", "textAlign": "justify"}
            )
        ], className="p-4")
    ], className="mb-4 shadow-sm border-0", style={"borderLeft": "5px solid #06B6D4", "backgroundColor": "#ffffff"})