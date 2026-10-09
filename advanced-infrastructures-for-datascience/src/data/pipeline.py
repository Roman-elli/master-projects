import src.config as cfg
import pandas as pd

def transform_data(df):
    """Filtra, limpa e cria novas features a partir dos dados brutos."""
    # 1. Filtrar colunas
    df = df[cfg.IMPORTANT_COLUMNS].copy()
    
    # 2. Transformar variável alvo (True = Faltou)
    df['No-show'] = df['No-show'] == 'Yes'
    
    # 3. Converter datas
    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
    
    # 4. Engenharia de Features
    df['WaitingDays'] = (df['AppointmentDay'].dt.normalize() - df['ScheduledDay'].dt.normalize()).dt.days
    df['ScheduledHour'] = df['ScheduledDay'].dt.hour
    df['ScheduledDayOfWeek'] = df['ScheduledDay'].dt.dayofweek
    df['AppointmentDayOfWeek'] = df['AppointmentDay'].dt.dayofweek
    df['AppointmentMonth'] = df['AppointmentDay'].dt.month
    
    # 5. Limpeza (Remover tempos de espera negativos)
    anomalies_mask = df['WaitingDays'] < 0
    print(f"[*] Registos removidos (Tempo de espera negativo): {anomalies_mask.sum()}")
    df = df[~anomalies_mask]
    
    return df

