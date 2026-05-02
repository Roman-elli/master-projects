from dash import html
import components # Seu arquivo de componentes

sidebar = html.Div([
    html.H3("Analytics", className="text-center mb-4 font-weight-bold"),
    # ... seus botões
], className="sidebar-personalizada") # <--- Substitui o argumento style=

content = html.Div([
    # ... seus gráficos e cards
], className="conteudo-principal") # <--- Substitui o argumento style=

# ...