import pandas as pd

def analyze_data(df):
    """Imprime métricas e distribuições para análise de sanidade dos dados."""
    print("\n--- Balanceamento da Classe Alvo (No-show) ---")
    target_counts = df['No-show'].value_counts()
    target_pct = df['No-show'].value_counts(normalize=True) * 100
    print(pd.DataFrame({'Contagem': target_counts, 'Percentagem (%)': target_pct.round(2)}))
        
    print("--- Contagem de Valores Nulos ---")
    print(df.isnull().sum())
    
    categorical_features = ['Gender', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received', 'Neighbourhood']
    print("\n--- Distribuição das Variáveis Categóricas (%) ---")
    for col in categorical_features:
        if col in df.columns:
            print(f"\n{col}:")
            print((df[col].value_counts(normalize=True) * 100).round(2))