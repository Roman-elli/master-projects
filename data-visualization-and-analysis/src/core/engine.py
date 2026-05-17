import pandas as pd
import src.config as cfg

def carregar_e_limpar_dados():
    """
    Centraliza o carregamento do CSV e o pré-processamento inicial.
    Utiliza o config.py para encontrar os caminhos e as colunas.
    """
    
    try:
        df_final = pd.read_csv(cfg.MERGED_DATA_PATH, low_memory=False)
        
        # 1. Tratamento e Conversão Numérica 
        for col in cfg.colunas_para_converter:
            # Verifica se a coluna existe no CSV para não dar erro
            if col in df_final.columns:
                df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
        
        # 2. Preencher nulos nos alunos
        if 'IPEDSCOUNT1' in df_final.columns:
            df_final['IPEDSCOUNT1'] = df_final['IPEDSCOUNT1'].fillna(1)
        
        # 3. Remover linhas sem os dados principais 
        df_final = df_final.dropna(subset=['EARN_MDN_4YR', 'DEBT_ALL_STGP_ANY_MDN'])
        
        # 4. Criar a coluna de Família
        if 'CIPCODE' in df_final.columns:
            df_final['CIP_FAMILIA'] = df_final['CIPCODE'].astype(str).str.split('.').str[0]

        # 5. Mapeamento do Tipo de Instituição
        if 'CONTROL' in df_final.columns:
            mapa_control = {1: 'Public', 2: 'Private (Non-Profit)', 3: 'Private (For-Profit)'}
            df_final['Tipo_Instituicao'] = df_final['CONTROL'].map(mapa_control)

        return df_final
    
    except Exception as e:
        print(f"Erro crítico ao carregar dados no engine.py: {e}")
        return pd.DataFrame()


def calcular_kpis(df):
    """
    Calcula as métricas principais para os cartões de topo.
    """
    if df.empty:
        return "0", "$ 0", "$ 0", "0%"
        
    total_alunos = f"{int(df['IPEDSCOUNT1'].sum()):,}".replace(",", ".") if 'IPEDSCOUNT1' in df.columns else "0"
    
    divida_val = df['DEBT_ALL_STGP_ANY_MDN'].median() if 'DEBT_ALL_STGP_ANY_MDN' in df.columns else 0
    salario_val = df['EARN_MDN_4YR'].median() if 'EARN_MDN_4YR' in df.columns else 0
    
    # Risco de Incumprimento (Default Risk)
    default_val = df['BBRR2_FED_COMP_DFLT'].mean() * 100 if 'BBRR2_FED_COMP_DFLT' in df.columns else 0
    
    divida = f"$ {divida_val:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    salario = f"$ {salario_val:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    
    # Se o valor for NaN (dados suprimidos), devolve "N/A"
    if pd.isna(default_val):
        default_rate = "N/A"
    else:
        default_rate = f"{default_val:.1f}%"
    
    return total_alunos, divida, salario, default_rate