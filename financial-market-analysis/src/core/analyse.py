import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

########
# PCA #
#######

def calculate_PCA(X_scaled):
    # PCA
    pca = PCA()
    pca.fit(X_scaled)

    # Cumulative Variance Explained
    explained_variance = np.cumsum(pca.explained_variance_ratio_)

    # Determine the number of components for 95% of the variance explained
    n_components = np.argmax(explained_variance >= 0.8) + 1
    print(f"Principal components: {n_components}")

    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% of Variance')
    plt.xlabel('Components number')
    plt.ylabel('Cumulative Variance Explained')
    plt.title('PCA')
    plt.legend()
    plt.show()

    return X_scaled

########
# RMT #
#######

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
    X_scaled = scaler.fit_transform(data.T) # Mantendo a tua lógica para ações como amostras

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

def calculate_nearest_neighbors(X_pca):
    # 1. Medimos a distância de cada ação à sua vizinha mais próxima (k=2)
    neighbors = NearestNeighbors(n_neighbors=2)
    neighbors_fit = neighbors.fit(X_pca)
    distances, _ = neighbors_fit.kneighbors(X_pca)

    # 2. Ordenamos as distâncias para formar a curva
    distances = np.sort(distances[:, 1], axis=0)

    # 3. Plot do Gráfico K-Distance
    plt.figure(figsize=(10, 5))
    plt.plot(distances, linewidth=2)
    plt.title('Gráfico K-Distance: Procura a subida acentuada (Cotovelo)')
    plt.xlabel('Ações (ordenadas por distância ao par mais próximo)')
    plt.ylabel('Distância (Potencial valor de EPS)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def RMT_clustering(best_eps, X_pca, returns):
    # 1. Executar o DBSCAN
    dbscan = DBSCAN(eps=best_eps, min_samples=2)
    labels = dbscan.fit_predict(X_pca)

    # 2. Agrupar os Tickers pelos Clusters
    clusters = {}
    tickers = returns.columns

    for ticker, cluster_id in zip(tickers, labels):
        if cluster_id == -1:  # Ignorar o ruído (ações sem par)
            continue
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(ticker)

    # 3. Filtrar apenas os clusters válidos (Pares ou Baskets)
    pairs = [group for group in clusters.values() if len(group) > 1]

    # 4. Mostrar os Resultados
    num_clusters = len(pairs)
    num_ruido = list(labels).count(-1)

    print(f"Resumo do DBSCAN (eps={best_eps}):")
    print(f"-> {num_clusters} clusters estruturais encontrados.")
    print(f"-> {num_ruido} ações rejeitadas como ruído.\n")

    print("Possíveis Pares/Baskets encontrados:")
    for i, pair in enumerate(pairs):
        print(f"Cluster {i}: {pair}")

    # Plot results
    # 1. Configurar o Scatter Plot
    plt.figure(figsize=(12, 8))

    # Usamos um colormap com cores distintas para os clusters e cinzento para o ruído
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab20', alpha=0.8, s=50)

    # 2. Anotações
    for i, ticker in enumerate(tickers):
        if labels[i] != -1:
            plt.text(X_pca[i, 0] + 0.02, X_pca[i, 1] + 0.02, ticker, fontsize=8, alpha=0.7)

    # 3. Estética do Gráfico
    plt.title(f"Visualização 2D dos Clusters (DBSCAN eps={best_eps})", fontsize=14)
    plt.xlabel("Componente Principal 1", fontsize=12)
    plt.ylabel("Componente Principal 2", fontsize=12)
    plt.colorbar(scatter, label='ID do Cluster (-1 é Ruído)')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()

    return pairs, clusters