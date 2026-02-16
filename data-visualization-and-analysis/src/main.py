import pandas as pd
import numpy as np

def processar_dados_universidades(caminho_arquivo):
    print("--- 1. INÍCIO DO PROCESSAMENTO ---")
    
    # 1. Carregar o CSV
    # low_memory=False evita avisos sobre tipos de dados mistos
    try:
        df = pd.read_csv(caminho_arquivo, low_memory=False)
        print(f"Total de linhas carregadas: {len(df)}")
    except FileNotFoundError:
        print("Erro: Arquivo não encontrado.")
        return None

    # 2. FILTRO DE NÍVEL (Essencial para comparação justa)
    # CREDLEV 3 = Bachelor's Degree (Licenciatura/Bacharelato)
    df = df[df['CREDLEV'].isin([3,5])].copy()
    print(f"Linhas após filtrar apenas Bacharelados: {len(df)}")

    # --- 3. DEFINIÇÃO DAS COLUNAS ---
    # Colunas Numéricas (Financeiras e Contagens)
    colunas_numericas = [
        'DEBT_ALL_STGP_ANY_MDN',      # Dívida Mediana (Geral)
        'EARN_MDN_4YR',               # Ganhos Medianos após 4 anos (Geral)
        'EARN_MALE_WNE_MDN_4YR',      # Ganhos Medianos (Homens)
        'EARN_NOMALE_WNE_MDN_4YR',    # Ganhos Medianos (Mulheres)
        'EARN_COUNT_MALE_WNE_4YR',    # Qtd Homens (para proporção)
        'EARN_COUNT_NOMALE_WNE_4YR'   # Qtd Mulheres (para proporção)
    ]
    
    # Colunas de Texto (Identificação)
    colunas_texto = ['UNITID', 'INSTNM', 'CONTROL', 'CIPCODE', 'CIPDESC']

    # --- 4. LIMPEZA DE DADOS ("PS" e "NULL") ---
    print("A limpar dados protegidos (PS)...")
    for col in colunas_numericas:
        # errors='coerce' transforma 'PS', 'NULL' e textos em NaN (Nulo)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- 5. ENGENHARIA DE DADOS (Cálculos) ---
    
    # A. Criar "Família do Curso" (Agrupamento Macro)
    # Pega nos 2 primeiros dígitos do CIPCODE (ex: 14.0101 -> 14)
    df['CIP_FAMILY'] = df['CIPCODE'].astype(str).str.split('.').str[0].str.zfill(2)
    
    # B. Mapear Códigos para Nomes Legíveis (Principais Áreas)
    mapa_areas = {
        '11': 'Computação e TI', 
        '14': 'Engenharia', 
        '26': 'Biologia e Biomédicas',
        '52': 'Negócios e Gestão', 
        '51': 'Saúde e Medicina', 
        '50': 'Artes Visuais e Cénicas',
        '45': 'Ciências Sociais', 
        '23': 'Letras e Literatura', 
        '09': 'Comunicação',
        '42': 'Psicologia', 
        '13': 'Educação', 
        '40': 'Ciências Físicas (Física/Química)',
        '04': 'Arquitetura',
        '01': 'Agricultura'
    }
    df['AREA_NOME'] = df['CIP_FAMILY'].map(mapa_areas).fillna('Outras Áreas')

    # C. Calcular Proporção de Mulheres (%)
    # Soma total da amostra de género
    df['TOTAL_GENDER_SAMPLE'] = df['EARN_COUNT_MALE_WNE_4YR'].fillna(0) + df['EARN_COUNT_NOMALE_WNE_4YR'].fillna(0)
    
    # Calcula % (evitando divisão por zero)
    df['PCT_FEMALE'] = df.apply(
        lambda x: (x['EARN_COUNT_NOMALE_WNE_4YR'] / x['TOTAL_GENDER_SAMPLE']) 
        if x['TOTAL_GENDER_SAMPLE'] > 0 else np.nan, axis=1
    )

    # D. Calcular Gap Salarial (Diferença Absoluta)
    # Positivo = Mulher ganha mais (raro), Negativo = Homem ganha mais
    df['GENDER_EARN_GAP'] = df['EARN_NOMALE_WNE_MDN_4YR'] - df['EARN_MALE_WNE_MDN_4YR']

    return df

# --- FUNÇÕES PARA GERAR OS CSVs FINAIS ---

def gerar_datasets_finais(df_completo):
    
    # 1. Dataset para ROI (Scatter Plot)
    # Removemos linhas onde não há Dívida OU não há Salário Geral
    df_roi = df_completo.dropna(subset=['DEBT_ALL_STGP_ANY_MDN', 'EARN_MDN_4YR']).copy()
    df_roi = df_roi[['INSTNM', 'AREA_NOME', 'CIPDESC', 'CONTROL', 'DEBT_ALL_STGP_ANY_MDN', 'EARN_MDN_4YR']]
    
    # 2. Dataset para Género (Dumbbell Plot / Disparidade)
    # Removemos linhas onde falta salário de Homem OU de Mulher
    df_genero = df_completo.dropna(subset=['EARN_MALE_WNE_MDN_4YR', 'EARN_NOMALE_WNE_MDN_4YR']).copy()
    df_genero = df_genero[[
        'INSTNM', 'AREA_NOME', 'CIPDESC', 
        'EARN_MALE_WNE_MDN_4YR', 'EARN_NOMALE_WNE_MDN_4YR', 
        'GENDER_EARN_GAP', 'PCT_FEMALE'
    ]]

    # 3. Dataset de Médias Nacionais (Benchmarks)
    # Agrupa por Área e calcula a média das métricas
    df_nacional = df_completo.groupby('AREA_NOME')[[
        'DEBT_ALL_STGP_ANY_MDN', 
        'EARN_MDN_4YR',
        'GENDER_EARN_GAP',
        'PCT_FEMALE'
    ]].mean().reset_index()

    return df_roi, df_genero, df_nacional

# --- EXECUÇÃO DO SCRIPT ---

# 1. Processar o ficheiro bruto
df_processado = processar_dados_universidades('assets/raw_data.csv')

if df_processado is not None:
    # 2. Separar em tabelas específicas
    roi_data, genero_data, nacional_data = gerar_datasets_finais(df_processado)

    # 3. Exibir Estatísticas
    print("\n--- RELATÓRIO FINAL ---")
    print(f"1. Dataset ROI (Dívida x Ganhos): {len(roi_data)} cursos disponíveis.")
    print(f"2. Dataset Género (Comparação H/M): {len(genero_data)} cursos disponíveis.")
    print(f"3. Médias Nacionais geradas para {len(nacional_data)} áreas de estudo.")
    
    # 4. (Opcional) Salvar em arquivos
    # roi_data.to_csv('analise_roi.csv', index=False)
    # genero_data.to_csv('analise_genero.csv', index=False)
    # nacional_data.to_csv('analise_nacional.csv', index=False)
    
    print("\nExemplo das Médias Nacionais:")
    print(nacional_data.head())