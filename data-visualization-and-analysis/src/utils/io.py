import pandas as pd
import numpy as np
import src.config as cfg

def extract_and_clean_raw_courses_data():
    df = pd.read_csv(cfg.RAW_COURSES_PATH, sep=',', low_memory=False)
    df_projeto = df[cfg.colunas_roi].copy()

    # Tratamento de valores ocultos (Privacy)
    df_projeto.replace(['PrivacySuppressed', 'PS', 'NULL'], np.nan, inplace=True)
    df_projeto.replace(r'^\s*$', np.nan, regex=True, inplace=True) # Limpa espaços em branco invisíveis

    # Conversão forçada para numérico em TODAS as colunas financeiras
    for col in cfg.colunas_para_converter:
        df_projeto[col] = pd.to_numeric(df_projeto[col], errors='coerce')

    # Trata valores inválidos

    # A linha TEM de ter a informação da dívida geral
    df_limpo = df_projeto.dropna(subset=['DEBT_ALL_STGP_ANY_MDN'])

    # A linha TEM de ter TODOS os anos de salário globais (1, 4 e 5 anos)
    df_limpo = df_limpo.dropna(subset=['EARN_MDN_1YR', 'EARN_MDN_4YR', 'EARN_MDN_5YR'], how='any')

    # A linha TEM de ter TODOS os dados de Género ao longo do tempo (1, 3, 4 e 5 anos)
    df_limpo = df_limpo.dropna(subset=cfg.colunas_genero, how='any')

    # A linha TEM de ter TODOS os dados de Bolsa Pell ao longo do tempo (1, 3, 4 e 5 anos)
    df_limpo = df_limpo.dropna(subset=cfg.colunas_pell, how='any')

    # Guardar os dados e extrair o código da família
    df_limpo['CIP_FAMILIA'] = df_limpo['CIPCODE'].astype(str).str.replace(r'\.0', '', regex=True).str.zfill(4).str[:2]

    df_limpo.to_csv(cfg.CLEAN_DATA_PATH)
    print(f"Nº final de linhas: {len(df_limpo)}")

# Esta função faz merge do dataset dos cursos com o dataset das instituições
def merge_datasets():
    df_geo = pd.read_csv(cfg.RAW_INSTITUTION_PATH, usecols=cfg.important_institution_cols)
    df_limpo = pd.read_csv(cfg.CLEAN_DATA_PATH, sep=',', low_memory=False)

    # Usamos 'left' para garantir que não perdemos nenhum curso do teu dataset original
    df_final = pd.merge(df_limpo, df_geo, on='UNITID', how='left')

    df_final = df_final.dropna(subset=cfg.important_institution_cols, how='any')

    # Guardar o ficheiro
    df_final.to_csv('data/merged_data.csv', index=False)
    print(f"Dataset pronto! Agora tens localização para {df_final['LATITUDE'].notna().sum()} linhas.")

if __name__ == '__main__':
    print("Starting extractions and cleaning pipeline...")

    print("Starting raw courses data operation...")
    extract_and_clean_raw_courses_data()
    print("Operation completed...")

    print("Merging datasets operation...")
    merge_datasets()
    print("Operation completed...")
