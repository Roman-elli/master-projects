import pandas as pd

def calcular_informacao_valida(arquivo_csv):
    try:
        # 1. Ler os dados brutos
        # O parâmetro low_memory=False ajuda se o ficheiro for muito grande
        df = pd.read_csv(arquivo_csv, low_memory=False)
        
        # 2. Definir as colunas cruciais para o gráfico de ROI
        # Dívida Mediana e Ganhos Medianos (4 anos)
        cols_interesse = ['DEBT_ALL_STGP_ANY_MDN', 'EARN_MDN_4YR']
        
        # 3. Limpeza: Converter para números e transformar "PS" em NaN (Vazio)
        # 'errors="coerce"' força qualquer texto a virar um valor nulo
        for col in cols_interesse:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # 4. Cálculos Estatísticos
        total_linhas = len(df)
        
        # Filtra apenas as linhas onde AMBOS (Dívida E Ganhos) existem
        df_limpo = df.dropna(subset=cols_interesse)
        linhas_validas = len(df_limpo)
        
        porcentagem = (linhas_validas / total_linhas) * 100 if total_linhas > 0 else 0
            
        # 5. Output dos Resultados
        print(f"--- Relatório de Disponibilidade de Dados ---")
        print(f"Total de cursos no ficheiro: {total_linhas}")
        print(f"Cursos utilizáveis (Com Dívida e Salário): {linhas_validas}")
        print(f"Cursos excluídos (Dados Ocultos/PS): {total_linhas - linhas_validas}")
        print(f"Taxa de aproveitamento do dataset: {porcentagem:.1f}%")
        
        return df_limpo

    except Exception as e:
        print(f"Erro ao processar: {e}")

# Para usar, bastaria chamar a função:
df_final = calcular_informacao_valida('assets/raw_data.csv')