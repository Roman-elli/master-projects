import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import adfuller

from utils.metrics import get_tls_beta, calculate_hurst
import config as cfg

import hdbscan

def RMT_clustering_hdbscan(X_pca, returns):
    """
    Utiliza HDBSCAN para lidar com densidades variáveis.
    """
    # 1. Executar o HDBSCAN (Não requer EPS)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, gen_min_span_tree=True)
    labels = clusterer.fit_predict(X_pca)

    # 2. Agrupar os Tickers pelos Clusters
    clusters = {}
    tickers = returns.columns

    for ticker, cluster_id in zip(tickers, labels):
        if cluster_id == -1:  # Ignorar o ruído
            continue
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(ticker)

    # 3. Filtrar apenas os clusters válidos (Pares ou Baskets)
    pairs = [group for group in clusters.values() if len(group) > 1]

    # 4. Mostrar os Resultados
    num_clusters = len(pairs)
    num_ruido = list(labels).count(-1)

    print(f"Resumo do HDBSCAN:")
    print(f"-> {num_clusters} clusters estruturais encontrados.")
    print(f"-> {num_ruido} ações rejeitadas como ruído.\n")

    # Plot results
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab20', alpha=0.8, s=50)
    
    for i, ticker in enumerate(tickers):
        if labels[i] != -1:
            plt.text(X_pca[i, 0] + 0.02, X_pca[i, 1] + 0.02, ticker, fontsize=8, alpha=0.7)

    plt.title("Visualização dos Clusters (HDBSCAN)", fontsize=14)
    plt.xlabel("Componente Principal 1", fontsize=12)
    plt.ylabel("Componente Principal 2", fontsize=12)
    plt.colorbar(scatter, label='ID do Cluster (-1 é Ruído)')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()

    return pairs, clusters

def calculate_RMT(data):
    # 1. Definir as dimensões dos dados
    # 'data' tem o formato (dias, ações). Precisamos de N e T.
    T_days, N_stocks = data.shape

    # 2. Calcular a matriz de correlação (N x N)
    # A correlação é preferível aqui porque estandardiza a volatilidade entre ativos
    corr_matrix = data.corr().to_numpy()

    # 3. Calcular os Autovalores (Eigenvalues) da matriz de correlação
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1] # Ordenar do maior (sinal) para o menor (ruído)

    # 4. Calcular o limite de Marchenko-Pastur (RMT)
    q = N_stocks / T_days # Rácio entre o número de variáveis e observações
    # Como usamos a matriz de correlação, a variância base é assumida como 1
    lambda_max = (1 + np.sqrt(q))**2

    # 5. Encontrar quantos componentes estão ACIMA do ruído
    n_components = len(eigenvalues[eigenvalues > lambda_max])

    print(f"Total de ações (N): {N_stocks} | Total de dias (T): {T_days}")
    print(f"Limite de Ruído Marchenko-Pastur (lambda_max): {lambda_max:.4f}")
    print(f"-> Componentes Principais Retidos (Sinal Puro): {n_components}")

    # 6. Aplicar o teu PCA original, mas agora com o número exato do SOTA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data.T)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # 7. Visualização RMT vs Ruído
    plt.figure(figsize=(10, 5))
    plt.plot(eigenvalues, marker='o', linestyle='none', color='blue', label='Autovalores Empíricos (As tuas ações)')
    plt.axhline(y=lambda_max, color='red', linestyle='--', label='Limite Marchenko-Pastur (Ruído)')
    plt.xlabel('Índice do Componente Principal')
    plt.ylabel('Autovalor (Eigenvalue)')
    plt.title('Filtragem de Sinal vs Ruído (RMT)')
    plt.xlim(-1, n_components + 20)
    plt.legend()
    plt.show()

    return X_pca

def validate_pairs(candidate_pairs, prices_formation, prices_full):
    """Filtra pares cointegrados e elásticos usando TLS, ADF e Hurst."""
    log_prices_form = np.log(prices_formation)
    log_prices_full = np.log(prices_full)
    valid_pairs, spreads_dict = [], {}

    for p in candidate_pairs:
        # Gerar spread a partir do TLS (Total least square)
        s1, s2 = p[0], p[1]
        y_form, x_form = log_prices_form[s1].values, log_prices_form[s2].values
        
        beta = get_tls_beta(x_form, y_form)
        spread_form = y_form - beta * x_form
        
        # Teste ADF (Cointegração)
        p_value = adfuller(spread_form)[1]
        
        if p_value < cfg.P_VALUE_ADF:
            hurst_exp = calculate_hurst(spread_form)
            # Filtro Hurst (Reversão Rápida)
            if hurst_exp < cfg.HURST_VALUE:
                valid_pairs.append({
                    'ativo_y': s1, 'ativo_x': s2, 'beta_tls': beta, 
                    'adf_p_value': p_value, 'hurst': hurst_exp
                })
                # Spread total para uso futuro
                spreads_dict[f"{s1}_{s2}"] = log_prices_full[s1] - beta * log_prices_full[s2]

    valid_pairs_df = pd.DataFrame(valid_pairs).sort_values(by=['hurst', 'adf_p_value'])
    return valid_pairs_df, spreads_dict