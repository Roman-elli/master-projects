import pandas as pd
import config as cfg

def extract_and_clean_data(file_path, clean=True, drop_NAN=True, save=True):
    # Carregar os dados
    df = pd.read_csv(file_path, sep=',')

    if not clean:
        return df
    
    df = df.drop(columns=cfg.unused_columns, errors='ignore')

    df = df.dropna(subset=cfg.important_columns)

    # Substituir NaNs por categorias claras
    df = df.fillna(value=cfg.cancel_fill_map)

    # Converter para float  
    for col in cfg.numeric_columns:
        if col in df.columns:
            # Converte para numérico; dados incorretos viram NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Padronizar nomes de locais ou categorias para evitar duplicatas por letras maiúsculas/minúsculas
    if 'Vehicle Type' in df.columns:
        df['Vehicle Type'] = df['Vehicle Type'].str.strip().str.title()
    
    if drop_NAN:
        df = df.dropna()
 
    if save:
        df.to_csv(cfg.SAVE_DATA_PATH)
        print(f"Clean data save completed in folder {cfg.SAVE_DATA_PATH}...")
    
    print("Raw data extraction & cleaning completed...")
    return df
