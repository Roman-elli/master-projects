import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# ---------------------------------------------------------
# 1. ESTILOS (CSS Inline)
# ---------------------------------------------------------
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "18rem", # Largura da barra escura
    "padding": "2rem 1rem",
    "background-color": "#202a35", # Cor escura parecida com a da imagem
    "color": "white"
}

CONTENT_STYLE = {
    "margin-left": "19rem", # Compensa a largura da sidebar
    "padding": "2rem",
    "background-color": "#f0f2f5", # Fundo cinza claro da página
    "min-height": "100vh"
}

# ---------------------------------------------------------
# 2. COMPONENTES REUTILIZÁVEIS
# ---------------------------------------------------------
# Função para criar os cartões de KPI do topo de forma rápida
def criar_kpi_card(titulo, valor):
    return dbc.Card([
        dbc.CardBody([
            html.H6(titulo, className="text-muted text-center"),
            html.H3(valor, className="text-center font-weight-bold")
        ])
    ], className="shadow-sm mb-4") # shadow-sm dá a sombrinha igual na foto

# ---------------------------------------------------------
# 3. LAYOUT PRINCIPAL
# ---------------------------------------------------------

# --- A BARRA LATERAL (SIDEBAR) ---
sidebar = html.Div([
    html.H3("Analytics", className="text-center mb-4 font-weight-bold"),
    
    # Substitua por dcc.Dropdown reais no seu app
    dbc.Button("Selecione o Estado ⬇", color="dark", className="w-100 mb-3", outline=True, style={"background": "#2c3e50"}),
    dbc.Button("Nível de Ensino ⬇", color="dark", className="w-100 mb-3", outline=True, style={"background": "#2c3e50"}),
    dbc.Button("Tipo de Instituição ⬇", color="dark", className="w-100 mb-3", outline=True, style={"background": "#2c3e50"}),
    dbc.Button("Área de Estudo ⬇", color="dark", className="w-100 mb-3", outline=True, style={"background": "#2c3e50"}),
], style=SIDEBAR_STYLE)


# --- O CONTEÚDO PRINCIPAL (GRÁFICOS) ---
content = html.Div([
    
    # LINHA 1: Os 3 KPIs (Total de Alunos, Dívida, Salário)
    dbc.Row([
        dbc.Col(criar_kpi_card("TOTAL DE ALUNOS", "2,450,123"), width=4),
        dbc.Col(criar_kpi_card("DÍVIDA MEDIANA", "$ 18,500"), width=4),
        dbc.Col(criar_kpi_card("SALÁRIO MEDIANO (4 ANOS)", "$ 45,200"), width=4),
    ]),

    # LINHA 2: Mapa e Gráfico de Rosca
    dbc.Row([
        # O Mapa ocupa mais espaço (8 de 12 colunas)
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Distribuição Geográfica"),
            # Aqui entraria o seu dcc.Graph(figure=fig_mapa)
            html.Div("Espaço do Mapa", style={"height": "350px", "background": "#e9ecef"}) 
        ]), className="shadow-sm"), width=8),

        # O Gráfico de Rosca ocupa menos espaço (4 de 12 colunas)
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Distribuição de Alunos"),
            # Aqui entraria o dcc.Graph(figure=fig_rosca)
            html.Div("Espaço do Donut", style={"height": "350px", "background": "#e9ecef"})
        ]), className="shadow-sm"), width=4),
    ], className="mb-4"), # Margem inferior para afastar da próxima linha

    # LINHA 3: Gráfico de Dispersão (Radar) e Boxplot
    dbc.Row([
        # O Scatter plot parece um pouco maior, vamos dar width 7
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Radar de ROI: Dívida vs. Salário"),
            html.Div("Espaço do Scatter", style={"height": "300px", "background": "#e9ecef"})
        ]), className="shadow-sm"), width=7),

        # O Boxplot fica com o resto, width 5
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Risco de Endividamento por Setor"),
            html.Div("Espaço do Boxplot", style={"height": "300px", "background": "#e9ecef"})
        ]), className="shadow-sm"), width=5),
    ]),

], style=CONTENT_STYLE)

# Juntando tudo no layout do app
app.layout = html.Div([sidebar, content])

if __name__ == "__main__":
    app.run(debug=True)