from dash import html, dcc
import dash_bootstrap_components as dbc

# =========================================================
# 1. ESTILOS BASE 
# =========================================================
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "18rem",
    "padding": "2rem 1rem",
    "background-color": "#202a35",
    "color": "white",
    "z-index": 1000
}

CONTENT_STYLE = {
    "margin-left": "19rem", 
    "padding": "2rem",
    "background-color": "#f0f2f5",
    "min-height": "100vh",
    "overflow-y": "auto" 
}

# =========================================================
# 2. DICIONÁRIOS DE MAPEAMENTO
# =========================================================
MAPA_ESTADOS = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
    'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
    'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
    'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
    'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri',
    'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
    'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
    'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont',
    'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming',
    'DC': 'District of Columbia', 'PR': 'Puerto Rico', 'VI': 'Virgin Islands'
}

CIP_NAMES = {
    '1': 'Agriculture', '3': 'Natural Resources', '4': 'Architecture', '5': 'Area/Gender Studies', 
    '9': 'Communication', '10': 'Communications Tech', '11': 'Computer Science', '12': 'Personal Services', 
    '13': 'Education', '14': 'Engineering', '15': 'Engineering Tech', '16': 'Foreign Languages', 
    '19': 'Family/Consumer Sci', '22': 'Legal Professions', '23': 'English', '24': 'Liberal Arts', 
    '25': 'Library Science', '26': 'Biology', '27': 'Mathematics', '29': 'Military Tech', 
    '30': 'Multi/Interdisciplinary', '31': 'Parks & Recreation', '38': 'Philosophy/Religion', 
    '39': 'Theology', '40': 'Physical Sciences', '41': 'Science Tech', '42': 'Psychology', 
    '43': 'Homeland Security', '44': 'Public Admin', '45': 'Social Sciences', '46': 'Construction', 
    '47': 'Mechanic/Repair', '48': 'Precision Production', '49': 'Transportation', '50': 'Visual/Performing Arts', 
    '51': 'Health Professions', '52': 'Business/Marketing', '54': 'History'
}

# =========================================================
# 3. FUNÇÕES AUXILIARES DE INTERFACE
# =========================================================
# Função para cortar nomes longos e evitar que a barra lateral fique desconfigurada
def truncar_nome(texto, limite=28):
    texto_str = str(texto)
    if len(texto_str) > limite:
        return texto_str[:limite] + '...'
    return texto_str

# =========================================================
# 4. FUNÇÃO PRINCIPAL DE LAYOUT
# =========================================================
def serve_layout(df_final):
    # --- SIDEBAR (Barra Lateral Esquerda) ---
    sidebar = html.Div([
        
        # Título e Subtítulo
        html.H2("EduVision USA", className="text-center mb-1 font-weight-bold", style={"letterSpacing": "1px"}),
        html.Hr(style={"borderColor": "rgba(255,255,255,0.1)"}),
        html.H6("Strategic analysis of student ROI and financial outcomes.", 
                className="text-center mb-4 font-italic", style={"opacity": "0.8", "fontSize": "0.85rem"}),
        
        # Botão de Reset
        dbc.Button("Reset Filters ↺", id="reset-btn", color="danger", outline=True, 
                   className="w-100 mb-4 border-0", style={"fontSize": "0.9rem"}),

        # FILTRO: STATE
        html.Label("1 - Select State", className="mb-2", style={"fontSize": "0.9rem", "textTransform": "uppercase"}),
        dcc.Dropdown(
            id='select-estado',
            options=[{'label': 'All States', 'value': 'TODOS'}] + 
                    [{'label': MAPA_ESTADOS.get(str(s).strip().upper(), s), 'value': s} 
                     for s in sorted(df_final['STABBR'].dropna().unique().tolist())],
            value='TODOS',
            clearable=False,
            className="mb-4",
            style={"color": "#000"} 
        ),
        
        # FILTRO: EDUCATION LEVEL
        html.Label("2 - Select Education Level", className="mb-2", style={"fontSize": "0.9rem", "textTransform": "uppercase"}),
        dcc.Dropdown(
            id='select-nivel',
            options=[
                {'label': 'All Levels', 'value': 'TODOS'},
                {'label': 'Undergraduate Certificate/Assoc.', 'value': '2'},
                {'label': 'Bachelor’s Degree', 'value': '3'},
                {'label': 'Post-baccalaureate Cert.', 'value': 4}
            ],
            value='TODOS',
            clearable=False,
            className="mb-4",
            style={"color": "#000"} 
        ),

        # FILTRO: INSTITUTION TYPE
        html.Label("3 - Institution Type", className="mb-2", style={"fontSize": "0.9rem", "textTransform": "uppercase"}),
        dcc.Dropdown(
            id='select-tipo',
            options=[
                {'label': 'All Types', 'value': 'TODOS'},
                {'label': 'Public', 'value': 'Public'},
                {'label': 'Private (Non-Profit)', 'value': 'Private (Non-Profit)'},
                {'label': 'Private (For-Profit)', 'value': 'Private (For-Profit)'}
            ],
            value='TODOS',
            clearable=False,
            className="mb-4",
            style={"color": "#000"} 
        ),

        # FILTRO: FIELD OF STUDY
        html.Label("4 - Field of Study", className="mb-2", style={"fontSize": "0.9rem", "textTransform": "uppercase"}),
        dcc.Dropdown(
            id='select-area',
            options=[{'label': 'All Fields', 'value': 'TODOS', 'title': 'All Fields'}] + 
                    [
                        {
                            'label': truncar_nome(area), 
                            'value': area,              
                            'title': area               
                        } 
                        for area in sorted(df_final['CIPDESC'].dropna().unique().tolist())
                    ],
            value='TODOS',
            clearable=False,
            className="mb-4",
            style={
                "color": "#000",
                "width": "100%",
            },
        ),            
    ], style=SIDEBAR_STYLE) 

    # --- ÁREA DE CONTEÚDO (Gráficos e Resumo) ---
    content = html.Div([
        
        # 1. KPIs (Topo)
        dbc.Row([
            dbc.Col(html.Div(id='kpi-alunos-container'), width=4),
            dbc.Col(html.Div(id='kpi-divida-container'), width=4),
            dbc.Col(html.Div(id='kpi-salario-container'), width=4),
        ], className="mb-4"),

        # 2. Linha Superior: Mapa e Donut
        dbc.Row([
            # Card do Mapa 
            dbc.Col(dbc.Card([
                dbc.CardHeader([
                    html.H5("Geographic Distribution of Median Earnings", className="font-weight-bold mb-1"),
                    html.P("Visualization of annual earnings 4 years after graduation by campus location", 
                           className="text-muted mb-0", style={"fontSize": "0.85rem", "fontWeight": "normal"})
                ], className="bg-white border-0 pt-3 pb-2"),

                dbc.CardBody([
                    dcc.Graph(
                        id='graph-mapa',
                        config={
                            'displayModeBar': True,
                            'displaylogo': False,
                            'modeBarButtonsToAdd': ['lasso2d', 'select2d'],
                            'modeBarButtonsToRemove': ['pan2d', 'zoom2d']
                        }
                    ) 
                ])
            ], className="shadow-sm border-0 h-100"), width=8),
            
            # Card do Donnut 
            dbc.Col(dbc.Card([
                dbc.CardHeader([
                    html.H5("Distribution of Students", className="font-weight-bold mb-1"),
                    html.P("Visualization of Students in public or private schools", 
                           className="text-muted mb-0", style={"fontSize": "0.85rem", "fontWeight": "normal"})
                ], className="bg-white border-0 pt-3 pb-2"),
                
                dbc.CardBody([
                    dcc.Graph(id='graph-donut', config={'displayModeBar': False})
                ])
            ], className="shadow-sm border-0 h-100"), width=4),
        ], className="mb-4"),

        # 3. Linha do Meio: Scatter e Boxplot
        dbc.Row([
            # Card do Scatter 
            dbc.Col(dbc.Card([
                dbc.CardHeader([
                    html.H5("Return on Investment Radar", className="font-weight-bold mb-1"),
                    html.P("Correlation between median student debt and earnings after 4 years", 
                        className="text-muted mb-0", style={"fontSize": "0.85rem", "fontWeight": "normal"})
                ], className="bg-white border-0 pt-3 pb-2"),
                dbc.CardBody([dcc.Graph(id='graph-scatter', config={'displayModeBar': False})])
            ], className="shadow-sm border-0 h-100"), width=7),

            # Card do Boxplot (Debt Risk)
            dbc.Col(dbc.Card([
                dbc.CardHeader([
                    html.H5("Indebtedness Risk", className="font-weight-bold mb-1"),
                    html.P("Debt distribution and outliers by institution sector", 
                        className="text-muted mb-0", style={"fontSize": "0.85rem", "fontWeight": "normal"})
                ], className="bg-white border-0 pt-3 pb-2"),
                dbc.CardBody([dcc.Graph(id='graph-box', config={'displayModeBar': False})])
            ], className="shadow-sm border-0 h-100"), width=5),
        ]),
        
        # 4. Linha Inferior: Evolução e Género
        dbc.Row([
            # Card: Salary Evolution & Pell Grant Impact
            dbc.Col(dbc.Card([
                dbc.CardHeader([
                    html.H5("Salary Evolution & Pell Grant Impact", className="font-weight-bold mb-1"),
                    html.P("Comparative trajectory of earnings for low-income (Pell) vs. high-income students", 
                        className="text-muted mb-0", style={"fontSize": "0.85rem", "fontWeight": "normal"})
                ], className="bg-white border-0 pt-3 pb-2"),

                dbc.CardBody([
                    dcc.Graph(id='graph-evolution', config={'displayModeBar': False})
                ])
            ], className="shadow-sm border-0 h-100"), width=6),

            # Card: Gender Enrollment Distribution
            dbc.Col(dbc.Card([
                dbc.CardHeader([
                    html.H5("Gender Enrollment Distribution", className="font-weight-bold mb-1"),
                    html.P("Breakdown of male and female student proportions within the selected scope", 
                        className="text-muted mb-0", style={"fontSize": "0.85rem", "fontWeight": "normal"})
                ], className="bg-white border-0 pt-3 pb-2"),

                dbc.CardBody([
                    dcc.Graph(id='graph-gender', config={'displayModeBar': False})
                ])
            ], className="shadow-sm border-0 h-100"), width=6),
        ], className="mb-4 mt-4"),
        
        # =========================================================
        # 5. RESUMO DINÂMICO 
        # =========================================================
        html.Div(id='resumo-dinamico-container', className="mt-5 mb-4"),
        
    ], style=CONTENT_STYLE) 

    # Retorna o layout final da App combinando a Sidebar e o Content
    return html.Div([sidebar, content])