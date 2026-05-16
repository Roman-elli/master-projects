import os
import dash
from dash import html, dcc, Input, Output, ctx
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px

from src.core.engine import calcular_kpis, carregar_e_limpar_dados 
from src.core.components import criar_kpi_card, criar_seccao_resumo 
from src.core.layout import serve_layout, MAPA_ESTADOS

# Dicionários de formatação e cores
TRADUCAO_SETORES = {'Pública': 'Public', 'Privada (s/ fins)': 'Private (Non-Profit)', 'Privada (c/ fins)': 'Private (For-Profit)'}
cores_setor = {'Public': '#06B6D4', 'Private (Non-Profit)': '#10B981', 'Private (For-Profit)': '#F43F5E'}

# Inicialização da App apenas com o tema Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# =========================================================
# 1. CARREGAMENTO E TRATAMENTO DE DADOS
# =========================================================
df_final = carregar_e_limpar_dados()

app.layout = serve_layout(df_final)

# =========================================================
# 2. CALLBACK PRINCIPAL 
# =========================================================
@app.callback(
    [Output('resumo-dinamico-container', 'children'),
     Output('kpi-alunos-container', 'children'),
     Output('kpi-divida-container', 'children'),
     Output('kpi-salario-container', 'children'),
     Output('graph-mapa', 'figure'),
     Output('graph-donut', 'figure'),
     Output('graph-scatter', 'figure'),
     Output('graph-box', 'figure'),
     Output('graph-evolution', 'figure'), 
     Output('graph-gender', 'figure')],    
    [Input('select-estado', 'value'),      
     Input('select-nivel', 'value'),       
     Input('select-tipo', 'value'),        
     Input('select-area', 'value'),        
     Input('graph-mapa', 'clickData'),
     Input('graph-mapa', 'selectedData'), 
     Input('graph-donut', 'clickData'),
     Input('graph-scatter', 'clickData'),
     Input('graph-box', 'clickData'),
     Input('reset-btn', 'n_clicks')]
)
def atualizar_dashboard(estado, nivel, tipo, area, click_mapa, selected_mapa, click_donut, click_scatter, click_box, n_reset):
    # Identificar qual foi a ação do utilizador que disparou o callback
    trigger_id = ctx.triggered_id
    trigger_prop = ctx.triggered[0]['prop_id'] if ctx.triggered else ''
    
    dff_base = df_final.copy()
    
    # --- A. APLICAÇÃO DOS FILTROS DA SIDEBAR ---
    if estado != 'TODOS':
        dff_base = dff_base[dff_base['STABBR'] == estado]
        
    if nivel is not None and nivel != 'TODOS':
        nivel_int = int(nivel)
        dff_base = dff_base[dff_base['CREDLEV'] == nivel_int]
        
    if area is not None and area != 'TODOS':
        dff_base = dff_base[dff_base['CIPDESC'] == area]
        
    if tipo != 'TODOS':
        dff_base = dff_base[dff_base['Tipo_Instituicao'] == tipo]

    # Calcular o centro do mapa com base no estado filtrado
    if not dff_base.empty and estado != 'TODOS':
        lat_foco, lon_foco, zoom_foco = dff_base['LATITUDE'].mean(), dff_base['LONGITUDE'].mean(), 5
    else:
        lat_foco, lon_foco, zoom_foco = 38.0, -96.0, 2.5 # Centro dos EUA por defeito
    
    # Se a seleção não tiver dados, devolver gráficos vazios
    if dff_base.empty:
        fig_vazia = px.scatter(title="No data found for this selection")
        return [dash.no_update]*4 + [fig_vazia]*6
    
    # --- B. LÓGICA DE CROSS-FILTERING (Seleção ao Clicar) ---
    dff_detalhe = dff_base.copy()
    dff_detalhe['is_selected'] = True  
    
    selecionados = []
    # Se o botão de Reset for clicado, limpa as seleções
    if trigger_id != 'reset-btn':
        if trigger_prop == 'graph-mapa.selectedData' and selected_mapa:
            selecionados = [pt['hovertext'] for pt in selected_mapa['points']]
        elif trigger_prop == 'graph-mapa.clickData' and click_mapa:
            selecionados = [click_mapa['points'][0]['hovertext']]
            lat_foco, lon_foco, zoom_foco = click_mapa['points'][0]['lat'], click_mapa['points'][0]['lon'], 10
        elif trigger_id == 'graph-donut' and click_donut:
            setor_clicado = click_donut['points'][0]['label']
            selecionados = dff_base[dff_base['Tipo_Instituicao'] == setor_clicado]['INSTNM'].tolist()
        elif trigger_id == 'graph-scatter' and click_scatter:
            selecionados = [click_scatter['points'][0]['hovertext']]
        elif trigger_id == 'graph-box' and click_box:
            setor_clicado = click_box['points'][0]['x']
            selecionados = dff_base[dff_base['Tipo_Instituicao'] == setor_clicado]['INSTNM'].tolist()

        # Atualiza a coluna de destaque
        if selecionados:
            dff_detalhe['is_selected'] = dff_detalhe['INSTNM'].isin(selecionados)

    # --- C. DADOS ATIVOS PARA KPIS E GRÁFICOS ESTATÍSTICOS ---
    dff_kpis = dff_detalhe[dff_detalhe['is_selected'] == True]
    k_al, k_div, k_sal = calcular_kpis(dff_kpis)

    # --- D. GERAÇÃO DO RESUMO ---
    if dff_kpis.empty:
        card_resumo = criar_seccao_resumo("No data found for the current selection.")
    else:
        total_inst = len(dff_kpis)
        
        # Parágrafo 1: Mapa / Foco Institucional
        # Conta quantas instituições ÚNICAS existem na seleção
        num_unicas = dff_kpis['INSTNM'].nunique()
        
        # 1. Obter as siglas dos estados na seleção atual e traduzir para os nomes completos
        estados_siglas = dff_kpis['STABBR'].dropna().unique()
        estados_nomes = sorted([MAPA_ESTADOS.get(sigla, sigla) for sigla in estados_siglas])
        
        # 2. Formatar o texto com a lista de estados (ex: "California, Texas and Florida")
        if len(estados_nomes) > 1:
            # Proteção: se forem muitos estados (ex: > 10), evitamos uma lista gigante
            if len(estados_nomes) > 10:
                estados_texto = "across the country"
            else:
                estados_texto = ", ".join(estados_nomes[:-1]) + " and " + estados_nomes[-1]
        elif len(estados_nomes) == 1:
            estados_texto = estados_nomes[0]
        else:
            estados_texto = "unknown locations"

        # 3. Construção do Parágrafo 1
        if num_unicas == 1:
            nome_instituicao = dff_kpis['INSTNM'].iloc[0]
            if area != 'TODOS':
                p1_mapa = f"🌍 **Institutional Focus:** Analyzing **{nome_instituicao}** (**{estados_texto}**) in detail, focusing on the study area of **{area}**."
            else:
                p1_mapa = f"🌍 **Institutional Focus:** Analyzing the overall data for **{nome_instituicao}** (**{estados_texto}**)."
                
        elif area != 'TODOS' and estado == 'TODOS':
            if len(estados_nomes) > 10:
                p1_mapa = f"🌍 **Geographic Distribution:** Analyzing the selected sample in **{area}**, distributed {estados_texto} ({num_unicas} institutions)."
            else:
                p1_mapa = f"🌍 **Geographic Distribution:** Analyzing the selected sample in **{area}**, distributed across {len(estados_nomes)} states (**{estados_texto}**)."
                
        elif area != 'TODOS' and estado != 'TODOS':
            nome_estado_filtro = MAPA_ESTADOS.get(estado, estado)
            p1_mapa = f"🌍 **Geographic Distribution:** Focusing on the state of **{nome_estado_filtro}** for **{area}** ({num_unicas} active institutions in the selection)."
            
        elif estado != 'TODOS':
            nome_estado_filtro = MAPA_ESTADOS.get(estado, estado)
            p1_mapa = f"🌍 **Geographic Distribution:** Focusing on the state of **{nome_estado_filtro}**, encompassing the {num_unicas} currently selected institutions."
            
        else:
            if len(estados_nomes) > 10:
                p1_mapa = f"🌍 **Geographic Distribution:** Analyzing the active selected overview {estados_texto}, encompassing {num_unicas} institutions."
            else:
                p1_mapa = f"🌍 **Geographic Distribution:** Analyzing the active selected overview across {len(estados_nomes)} states (**{estados_texto}**), encompassing {num_unicas} institutions."
                
        mapa_setores = {
            1.0: "Public", 1: "Public", "1": "Public",
            2.0: "Private (Non-Profit)", 2: "Private (Non-Profit)", "2": "Private (Non-Profit)",
            3.0: "Private (For-Profit)", 3: "Private (For-Profit)", "3": "Private (For-Profit)"
        }

        tipos_disponiveis = dff_kpis['CONTROL'].dropna().unique()
        texto_tipos_alunos = []
        texto_roi = []
        texto_risco = []

        COLUNA_ALUNOS = 'IPEDSCOUNT1' if 'IPEDSCOUNT1' in dff_kpis.columns else None
        total_alunos = dff_kpis[COLUNA_ALUNOS].sum() if COLUNA_ALUNOS else 1
        NOME_COLUNA_DIVIDA = 'DEBT_ALL_STGP_ANY_MDN' 

        for tipo_cod in sorted(tipos_disponiveis):
            dff_tipo = dff_kpis[dff_kpis['CONTROL'] == tipo_cod]
            if not dff_tipo.empty:
                nome_tipo = mapa_setores.get(tipo_cod, str(tipo_cod))
                
                # Texto Alunos (Donut)
                if COLUNA_ALUNOS and total_alunos > 0:
                    perc_alunos = (dff_tipo[COLUNA_ALUNOS].sum() / total_alunos) * 100
                    texto_tipos_alunos.append(f"**{perc_alunos:.1f}%** in **{nome_tipo}** institutions")
                else:
                    perc_inst = (len(dff_tipo) / total_inst) * 100
                    texto_tipos_alunos.append(f"**{perc_inst:.1f}%** of schools are **{nome_tipo}**")

                # Texto ROI e Risco
                salario = dff_tipo['EARN_MDN_4YR'].mean() if 'EARN_MDN_4YR' in dff_tipo.columns else 0
                divida = dff_tipo[NOME_COLUNA_DIVIDA].mean() if NOME_COLUNA_DIVIDA in dff_tipo.columns else 0
                
                if pd.notna(salario) and salario > 0:
                    texto_roi.append(f"in **{nome_tipo}** institutions the median salary is **${salario:,.0f}**")
                if pd.notna(divida) and divida > 0:
                    texto_risco.append(f"in **{nome_tipo}** institutions the median debt is **${divida:,.0f}**")

        # Parágrafos 2, 3 e 4
        p2_donut = "🎓 **Student Distribution:** The academic choice of students in the current selection is distributed as follows: " + ", ".join(texto_tipos_alunos) + "."
        p3_scatter = "💰 **ROI Radar (Earnings):** The return on investment highlights discrepancies: " + " and ".join(texto_roi) + "."
        p4_box = "⚠️ **Debt Risk:** The financial effort required from students reveals that " + " and ".join(texto_risco) + "."

        # Colunas de Evolução Salarial
        COLUNA_PELL_ANO3 = 'EARN_PELL_NE_MDN_3YR'       
        COLUNA_PELL_ANO4 = 'EARN_PELL_WNE_MDN_4YR'       
        COLUNA_PELL_ANO5 = 'EARN_PELL_WNE_MDN_5YR'
        COLUNA_NOPELL_ANO3 = 'EARN_NOPELL_NE_MDN_3YR'  
        COLUNA_NOPELL_ANO4 = 'EARN_NOPELL_WNE_MDN_4YR'   
        COLUNA_NOPELL_ANO5 = 'EARN_NOPELL_WNE_MDN_5YR' 

        # Parágrafo 5: Evolução (Pell)
        if COLUNA_PELL_ANO5 in dff_kpis.columns and COLUNA_NOPELL_ANO5 in dff_kpis.columns:
            pell_y3 = dff_kpis[COLUNA_PELL_ANO3].mean() if COLUNA_PELL_ANO3 in dff_kpis.columns else 0
            pell_y4 = dff_kpis[COLUNA_PELL_ANO4].mean()
            pell_y5 = dff_kpis[COLUNA_PELL_ANO5].mean()
            
            nopell_y3 = dff_kpis[COLUNA_NOPELL_ANO3].mean() if COLUNA_NOPELL_ANO3 in dff_kpis.columns else 0
            nopell_y4 = dff_kpis[COLUNA_NOPELL_ANO4].mean() 
            nopell_y5 = dff_kpis[COLUNA_NOPELL_ANO5].mean()
            
            fosso_salarial = nopell_y5 - pell_y5
            
            p5_evolucao = (
                f"📈 **Salary Evolution:** The salary trajectory highlights the impact of socioeconomic context. "
                f"After 5 years, selected students supported by *Pell Grants* reach a median salary of **${pell_y5:,.0f}**, "
                f"while students without grants reach **${nopell_y5:,.0f}**, creating a wage gap of approximately **${fosso_salarial:,.0f}**."
            )
        else:
            p5_evolucao = "📈 **Salary Evolution:** Detailed numerical data on Pell Grants is not available for this selection."

        # Parágrafo 6: Género
        if 'UGDS_WOMEN' in dff_kpis.columns and 'UGDS_MEN' in dff_kpis.columns:
            mulheres = dff_kpis['UGDS_WOMEN'].mean() * 100
            homens = dff_kpis['UGDS_MEN'].mean() * 100
            p6_genero = f"⚖️ **Gender Demographics:** The population balance in the current selection reveals a representation of **{mulheres:.1f}% women** and **{homens:.1f}% men**."
        else:
            p6_genero = "⚖️ **Gender Demographics:** Detailed demographic distribution is not available."

        texto_completo_markdown = f"{p1_mapa}\n\n{p2_donut}\n\n{p3_scatter}\n\n{p4_box}\n\n{p5_evolucao}\n\n{p6_genero}"
        card_resumo = criar_seccao_resumo(texto_completo_markdown)
    

    # --- E. GERAÇÃO DOS GRÁFICOS VISUAIS ---
    
    # Preparação para Opacidade (Contexto)
    status_opacity = dff_detalhe['is_selected'].map({True: 1.0, False: 0.15})
    dff_detalhe['IPEDSCOUNT1'] = dff_detalhe['IPEDSCOUNT1'].fillna(1)
    dff_detalhe.loc[dff_detalhe['IPEDSCOUNT1'] < 0, 'IPEDSCOUNT1'] = 0

    # 1. Gráfico Mapa
    fig_mapa = px.scatter_map(dff_detalhe, lat="LATITUDE", lon="LONGITUDE", color="EARN_MDN_4YR", size="IPEDSCOUNT1", hover_name="INSTNM", 
        hover_data={'LATITUDE': False, 'LONGITUDE': False, 'Tipo_Instituicao': True, 'DEBT_ALL_STGP_ANY_MDN': ':$,.0f', 'EARN_MDN_4YR': ':$,.0f', 'IPEDSCOUNT1': True, 'is_selected': False},
        labels={'Tipo_Instituicao': 'Sector', 'DEBT_ALL_STGP_ANY_MDN': 'Median Debt ($)', 'EARN_MDN_4YR': 'Median Earnings ($)', 'IPEDSCOUNT1': 'Total Students'},
        zoom=zoom_foco, center=dict(lat=lat_foco, lon=lon_foco), color_continuous_scale=px.colors.sequential.Plasma,size_max=15, opacity=status_opacity)
    
    fig_mapa.update_traces(selected={'marker': {'opacity': 1.0}}, unselected={'marker': {'opacity': 0.15}})
    fig_mapa.update_layout(mapbox_style="carto-positron", uirevision=estado, margin={"r":0, "t":0, "l":0, "b":50},
        coloraxis_colorbar=dict(orientation="h", y=-0.2, x=0.5, xanchor="center", len=0.5, thickness=15, title={'text': "Median Earnings ($)", 'side': 'top'})
    )
   
    # 2. Gráfico Donut
    fig_donut = px.pie(dff_kpis, values='IPEDSCOUNT1', names='Tipo_Instituicao', hole=0.45, color='Tipo_Instituicao', color_discrete_map=cores_setor,
        labels={'Tipo_Instituicao': 'Sector', 'IPEDSCOUNT1': 'Total Students'}
    )
    fig_donut.update_traces(textinfo='percent', hovertemplate="<b>%{label}</b><br>Students: %{value:,.0f}<br>Percentage: %{percent}")
    fig_donut.update_layout(margin={"r":20, "t":30, "l":20, "b":80}, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5))

    # 3. Gráfico Scatter
    opacidade_base = 0.8 if not selecionados else 0.3 
    fig_scatter = px.scatter(dff_detalhe, x='DEBT_ALL_STGP_ANY_MDN', y='EARN_MDN_4YR', size='IPEDSCOUNT1', color='Tipo_Instituicao', hover_name='INSTNM',
        hover_data={'Tipo_Instituicao': True,'DEBT_ALL_STGP_ANY_MDN': ':$,.0f', 'EARN_MDN_4YR': ':$,.0f', 'IPEDSCOUNT1': ':,', 'is_selected': False},
        color_discrete_map=cores_setor, opacity=opacidade_base,
        labels={'DEBT_ALL_STGP_ANY_MDN': 'Median Debt','EARN_MDN_4YR': 'Median Earnings','Tipo_Instituicao': 'Sector','IPEDSCOUNT1': 'Total Students'}
    )

    if selecionados:
        dff_vip = dff_detalhe[dff_detalhe['is_selected'] == True]
        fig_vip = px.scatter(dff_vip, x='DEBT_ALL_STGP_ANY_MDN', y='EARN_MDN_4YR',size='IPEDSCOUNT1', color='Tipo_Instituicao',hover_name='INSTNM',color_discrete_map=cores_setor)
        fig_vip.update_traces(marker=dict(opacity=1.0, line=dict(width=1.5, color='black')), showlegend=False)
        for trace in fig_vip.data:
            fig_scatter.add_trace(trace)

    fig_scatter.update_layout(margin={"r":10,"t":10,"l":10,"b":10}, plot_bgcolor='#FFFFFF',showlegend=True)

    # 4. Gráfico Boxplot
    fig_box = px.box(dff_kpis, x="Tipo_Instituicao", y="DEBT_ALL_STGP_ANY_MDN", color="Tipo_Instituicao", color_discrete_map=cores_setor,
        labels={'DEBT_ALL_STGP_ANY_MDN': 'Debt ($)', 'Tipo_Instituicao': 'Sector'}
    )
    fig_box.update_traces(hovertemplate="<b>Sector:</b> %{x}<br><b>Value:</b> $%{y:,.0f}")
    fig_box.update_layout(margin={"r":0,"t":10,"l":0,"b":0}, plot_bgcolor='#FFFFFF', showlegend=True, xaxis={'visible': False})
    
    # 5. Gráfico de Linhas (Evolução)
    df_evolucao = pd.DataFrame({
        'Year': ['Year 3', 'Year 4', 'Year 5', 'Year 3', 'Year 4', 'Year 5'],
        'Type': ['Pell Grant', 'Pell Grant', 'Pell Grant', 'No Pell', 'No Pell', 'No Pell'],
        'Earnings': [
            dff_kpis['EARN_PELL_NE_MDN_3YR'].mean() if 'EARN_PELL_NE_MDN_3YR' in dff_kpis else 0, 
            dff_kpis['EARN_PELL_WNE_MDN_4YR'].mean() if 'EARN_PELL_WNE_MDN_4YR' in dff_kpis else 0,
            dff_kpis['EARN_PELL_WNE_MDN_5YR'].mean() if 'EARN_PELL_WNE_MDN_5YR' in dff_kpis else 0,
            dff_kpis['EARN_NOPELL_NE_MDN_3YR'].mean() if 'EARN_NOPELL_NE_MDN_3YR' in dff_kpis else 0,
            dff_kpis['EARN_NOPELL_WNE_MDN_4YR'].mean() if 'EARN_NOPELL_WNE_MDN_4YR' in dff_kpis else 0,
            dff_kpis['EARN_NOPELL_WNE_MDN_5YR'].mean() if 'EARN_NOPELL_WNE_MDN_5YR' in dff_kpis else 0
        ]
    })

    fig_evolution = px.line(df_evolucao, x='Year', y='Earnings', color='Type',
        markers=True, color_discrete_map={'Pell Grant': '#F43F5E', 'No Pell': '#06B6D4'},
        labels={'Earnings': 'Median Earnings ($)'}
    )
    fig_evolution.update_layout(plot_bgcolor='white', margin={"r":10,"t":10,"l":10,"b":10})

    # 6. Gráfico de Barras (Género)
    total_homens = dff_kpis['UGDS_MEN'].mean() if not dff_kpis['UGDS_MEN'].isnull().all() else 0
    total_mulheres = dff_kpis['UGDS_WOMEN'].mean() if not dff_kpis['UGDS_WOMEN'].isnull().all() else 0
    
    fig_gender = px.bar(x=['Men', 'Women'], y=[total_homens, total_mulheres], color=['Men', 'Women'],
        color_discrete_map={'Men': '#06B6D4', 'Women': '#F43F5E'}, labels={'x': 'Gender', 'y': 'Proportion'}
    )
    fig_gender.update_traces(hovertemplate="<b>%{x}</b>: %{y:.1%}")
    fig_gender.update_layout(plot_bgcolor='white', showlegend=False, margin={"r":10,"t":10,"l":10,"b":10})

    return (card_resumo, 
            criar_kpi_card("TOTAL STUDENTS", k_al), 
            criar_kpi_card("MEDIAN DEBT", k_div), 
            criar_kpi_card("MEDIAN EARNINGS", k_sal), 
            fig_mapa, fig_donut, fig_scatter, fig_box, fig_evolution, fig_gender
    )
    
if __name__ == "__main__":
    app.run(debug=True)