import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN

def activity_metric(data_array, sensor='acc', save_dir="data/activity", save_plots=True):
    sensor_position = ["Left Wrist", "Right Wrist", "Chest", "Right Upper Leg", "Left Lower Leg"]
    sensor_cols = {
        'acc': [1, 2, 3],
        'gyro': [4, 5, 6],
        'mag': [7, 8, 9]
    }

    cols = sensor_cols[sensor]
    n_medidores = len(data_array[0])

    os.makedirs(save_dir, exist_ok=True)
    outlier_file = os.path.join(save_dir, f"{sensor}_outliers.txt")

    with open(outlier_file, "w") as f:
        results = []

        for med_idx in range(n_medidores):
            all_modules = []
            all_activities = []

            for person in data_array:
                activity = person[med_idx]
                x = activity[:, cols[0]]
                y = activity[:, cols[1]]
                z = activity[:, cols[2]]
                modulo = np.sqrt(x**2 + y**2 + z**2)
                col12 = activity[:, 11].astype(int)

                all_modules.extend(modulo)
                all_activities.extend(col12)

            all_modules = np.array(all_modules)
            all_activities = np.array(all_activities)
            results.append((all_modules, all_activities))

            unique_activities = np.unique(all_activities)
            data_activities = [all_modules[all_activities == a] for a in unique_activities]

            f.write(f"Outliers density for {sensor} | {sensor_position[med_idx]}:\n")
            print(f"Outliers density for {sensor} | {sensor_position[med_idx]}:")

            for a in unique_activities:
                activity_data = all_modules[all_activities == a]
                n_r = len(activity_data)
                q1 = np.percentile(activity_data, 25)
                q3 = np.percentile(activity_data, 75)
                iqr = q3 - q1
                low_limit = q1 - 1.5 * iqr
                sup_limit = q3 + 1.5 * iqr

                outliers = np.sum((activity_data < low_limit) | (activity_data > sup_limit))
                d = (outliers / n_r) * 100

                line = f"  Activity {a}: {d:.2f}% outliers ({outliers}/{n_r})\n"
                f.write(line)
                print(line.strip())

            f.write("\n---------------------------------------------\n\n")
            print("\n---------------------------------------------\n")

            if save_plots:
                plot_folder = os.path.join(save_dir, sensor, sensor_position[med_idx])
                os.makedirs(plot_folder, exist_ok=True)
                plt.figure(figsize=(10, 6))
                plt.boxplot(data_activities, labels=unique_activities)
                plt.xlabel("Activity (column 12)")
                plt.ylabel(f"Vector module ({sensor})")
                plt.title(f"{sensor.upper()} | {sensor_position[med_idx]}")
                plt.grid(True, axis='y', linestyle='--', alpha=0.7)
                plt.savefig(os.path.join(plot_folder, f"{sensor_position[med_idx]}_boxplot.png"))
                plt.close()

    return results

def zscore_outliers(results, sensor='acc', k=3, use_activities=True, save_dir="data/zscore"):
    sensor_position = ["Left Wrist", "Right Wrist", "Chest", "Right Upper Leg", "Left Lower Leg"]

    k_folder = os.path.join(save_dir, f"k={k}", "activity" if use_activities else "noActivity")
    os.makedirs(k_folder, exist_ok=True)

    outlier_file = os.path.join(k_folder, f"{sensor}_outliers.txt")
    with open(outlier_file, "w") as f:

        for med_idx, (modules, activities) in enumerate(results, start=1):
            sensor_name = sensor_position[med_idx - 1]

            f.write(f"Outliers density for {sensor.upper()} | {sensor_name} (Z-score method, k={k}):\n")
            print(f"Outliers density for {sensor.upper()} | {sensor_name} (Z-score method, k={k}):")

            plt.figure(figsize=(12, 6))

            if use_activities:
                unique_activities = np.unique(activities)
                for a in unique_activities:
                    dados_atividade = modules[activities == a]
                    mean = np.mean(dados_atividade)
                    std = np.std(dados_atividade)
                    z_scores = (dados_atividade - mean) / std

                    mask_outliers = np.abs(z_scores) > k

                    total = len(dados_atividade)
                    outliers = np.sum(mask_outliers)
                    perc = (outliers / total) * 100

                    line = f"  Activity {a}: {perc:.2f}% outliers ({outliers}/{total})\n"
                    f.write(line)
                    print(line.strip())

                    # Scatter plot
                    plt.scatter([a]*len(dados_atividade[~mask_outliers]),
                                dados_atividade[~mask_outliers],
                                color='blue', alpha=0.6)
                    plt.scatter([a]*len(dados_atividade[mask_outliers]),
                                dados_atividade[mask_outliers],
                                color='red', alpha=0.8)
            else:
                mean = np.mean(modules)
                std = np.std(modules)
                z_scores = (modules - mean) / std
                mask_outliers = np.abs(z_scores) > k

                total = len(modules)
                outliers = np.sum(mask_outliers)
                perc = (outliers / total) * 100

                line = f"  Total values: {perc:.2f}% outliers ({outliers}/{total})\n"
                f.write(line)
                print(line.strip())

                indices = np.arange(total)
                plt.scatter(indices[~mask_outliers], modules[~mask_outliers],
                            color='blue', alpha=0.6)
                plt.scatter(indices[mask_outliers], modules[mask_outliers],
                            color='red', alpha=0.8)

            f.write("\n---------------------------------------------\n\n")
            print("\n---------------------------------------------\n")

            # Salvar o plot
            plot_file = os.path.join(k_folder, f"{sensor}_{sensor_name}.png")
            plt.xlabel("Sample Index" if not use_activities else "Activity (Column 12)")
            plt.ylabel(f"{sensor.upper()} Vector Magnitude")
            plt.title(f"{sensor.upper()} | {sensor_name} | Z-score Outliers (k={k})")
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.savefig(plot_file, dpi=200)
            plt.close()

def manual_kmeans(X, n_clusters, max_iter, tol):
    np.random.seed(42)
    n_samples = X.shape[0]

    indices = np.random.choice(n_samples, n_clusters, replace=False)
    centroids = X[indices]

    for iteration in range(max_iter):
        # 1. Calcula distância euclidiana entre cada ponto e cada centróide
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

        # 2. Atribui cada ponto ao centróide mais próximo
        labels = np.argmin(distances, axis=1)

        # 3. Recalcula centróides
        new_centroids = np.array([
            X[labels == k].mean(axis=0) if np.any(labels == k) else centroids[k]
            for k in range(n_clusters)
        ])

        # 4. Checa convergência
        shift = np.linalg.norm(new_centroids - centroids)
        if shift < tol:
            print(f"Convergência alcançada em {iteration+1} iterações.")
            break

        centroids = new_centroids

    return labels, centroids

def k_mean(data_array, sensor='acc', n_clusters=3, max_iter=100, tol=1e-4, save_dir="data/kmean", save_plots=True):

    sensor_position = ["Left Wrist", "Right Wrist", "Chest", "Right Upper Leg", "Left Lower Leg"]
    sensor_cols = {
        'acc': [1, 2, 3],
        'gyro': [4, 5, 6],
        'mag': [7, 8, 9]
    }

    cols = sensor_cols[sensor]
    n_medidores = len(data_array[0])

    for med_idx in range(n_medidores):
        sensor_name = sensor_position[med_idx]

        all_xyz = []
        for person in data_array:
            data_sensor = person[med_idx]
            x = data_sensor[:, cols[0]]
            y = data_sensor[:, cols[1]]
            z = data_sensor[:, cols[2]]
            xyz = np.column_stack((x, y, z))
            all_xyz.append(xyz)

        all_xyz = np.vstack(all_xyz)

        labels, centroids = manual_kmeans(all_xyz, n_clusters, max_iter, tol)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(
            all_xyz[:, 0],
            all_xyz[:, 1],
            all_xyz[:, 2],
            c=labels,
            cmap='tab10',
            s=10,
            alpha=0.6
        )

        ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            centroids[:, 2],
            c='black',
            s=100,
            marker='X',
            label='Centroids'
        )

        ax.set_title(f"{sensor.upper()} | {sensor_name} | Manual K-Means (k={n_clusters})")
        ax.set_xlabel(f"{sensor.upper()} X")
        ax.set_ylabel(f"{sensor.upper()} Y")
        ax.set_zlabel(f"{sensor.upper()} Z")

        if save_plots:
            folder_path = os.path.join(save_dir, sensor, sensor_name, f"k={n_clusters}")
            os.makedirs(folder_path, exist_ok=True)
            file_path = os.path.join(folder_path, f"{sensor_name}_k={n_clusters}.png")
            plt.savefig(file_path, dpi=200)
            plt.close(fig)  # fecha o plot para não ocupar memória
            
def kmeans_outliers(data_array, sensor, n_clusters, save_dir="data/kmean_outliers", save_plots=True):   
    os.makedirs(save_dir, exist_ok=True)

    # --- 1️⃣ Extrair colunas do sensor (X, Y, Z) ---
    sensor_cols = {
        'acc': [1, 2, 3],
        'gyro': [4, 5, 6],
        'mag': [7, 8, 9]
    }
    cols = sensor_cols[sensor]

    all_xyz = []
    for person in data_array:
        for med in person:
            data_sensor = med
            x = data_sensor[:, cols[0]]
            y = data_sensor[:, cols[1]]
            z = data_sensor[:, cols[2]]
            xyz = np.column_stack((x, y, z))
            all_xyz.append(xyz)
    X = np.vstack(all_xyz)

    print(f"Executando K-Means manual ({sensor.upper()}) com k={n_clusters} ...")

    # --- 2️⃣ Executar K-Means manual ---
    labels, centroids = manual_kmeans(X, n_clusters=n_clusters, max_iter=100, tol=1e-4)

    # --- 3️⃣ Calcular distâncias e identificar outliers ---
    distances = np.linalg.norm(X - centroids[labels], axis=1)
    outliers = np.zeros(len(X), dtype=bool)
    k = 3  # limite Z-score padrão

    for c in np.unique(labels):
        mask = labels == c
        cluster_dist = distances[mask]
        mean = np.mean(cluster_dist)
        std = np.std(cluster_dist)
        z = (cluster_dist - mean) / std
        outliers[mask] = np.abs(z) > k

    total_outliers = np.sum(outliers)
    perc = total_outliers / len(X) * 100
    print(f"Total de outliers (K-Means): {perc:.2f}% ({total_outliers}/{len(X)})")

    # --- 4️⃣ Guardar resultados num ficheiro ---
    out_file = os.path.join(save_dir, f"{sensor}_outliers_kmeans.txt")
    with open(out_file, "w") as f:
        f.write(f"K-Means Outliers ({sensor.upper()})\n")
        f.write(f"Clusters (k): {n_clusters}\n")
        f.write(f"Total: {len(X)} amostras\n")
        f.write(f"Outliers: {total_outliers} ({perc:.2f}%)\n\n")
        for c in np.unique(labels):
            cluster_mask = labels == c
            cluster_outliers = np.sum(outliers[cluster_mask])
            total_cluster = np.sum(cluster_mask)
            f.write(f"Cluster {c}: {cluster_outliers}/{total_cluster} ({(cluster_outliers/total_cluster)*100:.2f}%)\n")

    # --- 5️⃣ Plot 3D (se pedido) ---
    if save_plots:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # pontos normais
        ax.scatter(X[~outliers, 0], X[~outliers, 1], X[~outliers, 2],
                   c=labels[~outliers], cmap='tab10', s=8, alpha=0.6)

        # outliers
        ax.scatter(X[outliers, 0], X[outliers, 1], X[outliers, 2],
                   c='red', s=25, label='Outliers', alpha=0.8)

        # centróides
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                   c='black', s=100, marker='X', label='Centroids')

        ax.set_title(f"{sensor.upper()} | K-Means Outliers (k={n_clusters})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{sensor}_kmeans_outliers_k={n_clusters}.png"), dpi=200)
        plt.close()

    return outliers

def dbscan_outliers(data_array, sensor='mag', eps=0.5, min_samples=10, save_dir="data/dbscan", save_plots=True):

    os.makedirs(save_dir, exist_ok=True)

    # --- 1️⃣ Extrair colunas X,Y,Z do sensor ---
    sensor_cols = {
        'acc': [1, 2, 3],
        'gyro': [4, 5, 6],
        'mag': [7, 8, 9]
    }
    cols = sensor_cols[sensor]

    all_xyz = []
    for person in data_array:
        for med in person:
            data_sensor = med
            x = data_sensor[:, cols[0]]
            y = data_sensor[:, cols[1]]
            z = data_sensor[:, cols[2]]
            xyz = np.column_stack((x, y, z))
            all_xyz.append(xyz)
    X = np.vstack(all_xyz)

    print(f"Executando DBSCAN ({sensor.upper()}) com eps={eps}, min_samples={min_samples} ...")
    # Antes do DBSCAN:
    if len(X) > 50000:
        print(f"Dataset muito grande ({len(X)} pontos). Amostrando 50.000 para DBSCAN...")
        idx = np.random.choice(len(X), 50000, replace=False)
        X = X[idx]

    # --- 2️⃣ Aplicar DBSCAN ---
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_

    # --- 3️⃣ Identificar outliers ---
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = np.sum(labels == -1)
    perc_out = (n_outliers / len(X)) * 100

    print(f"Clusters encontrados: {n_clusters}")
    print(f"Outliers (DBSCAN): {perc_out:.2f}% ({n_outliers}/{len(X)})")

    # --- 4️⃣ Guardar resultados ---
    out_file = os.path.join(save_dir, f"{sensor}_dbscan_results.txt")
    with open(out_file, "w") as f:
        f.write(f"DBSCAN ({sensor.upper()})\n")
        f.write(f"eps={eps}, min_samples={min_samples}\n")
        f.write(f"Clusters encontrados: {n_clusters}\n")
        f.write(f"Total de amostras: {len(X)}\n")
        f.write(f"Outliers: {n_outliers} ({perc_out:.2f}%)\n")

    # --- 5️⃣ Plot 3D ---
    if save_plots:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Cores para clusters
        unique_labels = set(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for label, color in zip(unique_labels, colors):
            mask = labels == label
            if label == -1:
                color = 'red'  # outliers
                label_name = 'Outliers'
            else:
                label_name = f'Cluster {label}'
            ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2],
                       c=color, s=8, alpha=0.6, label=label_name)

        ax.set_title(f"{sensor.upper()} | DBSCAN Clusters & Outliers")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{sensor}_dbscan_outliers.png"), dpi=200)
        plt.close()

    return labels


def inject_outliers(array, x=5.0, k=3, z=1.0, random_seed=42):
    """
    Injeta outliers em um array 1D para garantir densidade >= x%.
    
    Parâmetros:
    - array: np.array 1D (valores originais)
    - x: densidade mínima de outliers (%) desejada
    - k: limite do z-score
    - z: amplitude máxima adicional do outlier
    - random_seed: para reprodutibilidade
    """
    np.random.seed(random_seed)
    array = array.copy()
    
    # 1️⃣ calcular média e desvio
    mu = np.mean(array)
    sigma = np.std(array)
    
    # 2️⃣ identificar outliers existentes
    z_scores = (array - mu) / sigma
    existing_outliers = np.abs(z_scores) > k
    n_outliers = np.sum(existing_outliers)
    n_total = len(array)
    
    # densidade atual
    d = (n_outliers / n_total) * 100
    print(f"Densidade atual de outliers: {d:.2f}%")
    
    # 3️⃣ se d < x, sortear pontos para se tornarem outliers
    if d < x:
        # quantidade de pontos a transformar
        n_needed = int(np.ceil((x - d)/100 * n_total))
        
        # índices de pontos não outliers
        non_outlier_indices = np.where(~existing_outliers)[0]
        selected_indices = np.random.choice(non_outlier_indices, size=n_needed, replace=False)
        
        for idx in selected_indices:
            s = np.random.choice([-1, 1])
            q = np.random.uniform(0, z)
            array[idx] = mu + s * (k * sigma + q)
            
        print(f"Injetados {n_needed} outliers para atingir densidade {x}%")
    else:
        print("Densidade já >= x%, não foi necessário injetar outliers")
    
    return array

def linear_model(X_train, y_train):
    """
    Determina o modelo linear de ordem p usando mínimos quadrados.

    Parâmetros:
    - X_train: np.array de forma (n_samples, p), dados de entrada
    - y_train: np.array de forma (n_samples,), saídas correspondentes

    Retorna:
    - beta: np.array de forma (p+1,), vetor de pesos incluindo o intercepto
    """
    # 1️⃣ Adiciona coluna de 1s para o intercepto
    n_samples = X_train.shape[0]
    X_aug = np.hstack((np.ones((n_samples, 1)), X_train))  # (n_samples, p+1)

    # 2️⃣ Calcula beta pelos mínimos quadrados
    # beta = (X^T X)^-1 X^T y
    XtX = X_aug.T @ X_aug
    Xty = X_aug.T @ y_train
    beta = np.linalg.inv(XtX) @ Xty

    return beta

def create_windows(data, p):
    """
    Cria matriz X (janela) e vetor y para modelo linear de ordem p.
    data: array 1D com módulo da aceleração
    p: número de valores anteriores a usar
    """
    X = []
    y = []
    for i in range(p, len(data)):
        X.append(data[i-p:i])  # p valores anteriores
        y.append(data[i])      # valor atual
    return np.array(X), np.array(y)


def linear_model_correction(modulo, p=3, outlier_density=10, k=3, z=2.0, plot_examples=True):
    """
    Pipeline completo 3.10:
    1. Injeta outliers no módulo da aceleração
    2. Treina modelo linear de ordem p
    3. Substitui outliers pelos valores previstos
    4. Analisa erro de predição
    """
    # 1️⃣ Injeta 10% de outliers
    data_with_outliers = inject_outliers(modulo, x=outlier_density, k=k, z=z)

    # 2️⃣ Cria janelas
    X_train, y_train = create_windows(data_with_outliers, p)

    # 3️⃣ Treina modelo linear
    beta = linear_model(X_train, y_train)

    # 4️⃣ Predição
    y_pred = np.hstack((np.ones((X_train.shape[0], 1)), X_train)) @ beta

    # 5️⃣ Detecta outliers
    mu = np.mean(y_train)
    sigma = np.std(y_train)
    z_scores = (y_train - mu) / sigma
    outliers_mask = np.abs(z_scores) > k

    # 6️⃣ Substitui outliers pelos valores previstos
    y_corrected = y_train.copy()
    y_corrected[outliers_mask] = y_pred[outliers_mask]

    # 7️⃣ Analisa erro
    error = y_corrected - y_train

    if plot_examples:
        # Distribuição do erro
        plt.figure(figsize=(8,5))
        plt.hist(error, bins=50, alpha=0.7)
        plt.title(f"Distribuição do erro (p={p})")
        plt.xlabel("Erro")
        plt.ylabel("Frequência")
        plt.grid(True)
        plt.show()

        # Exemplo valores reais vs previstos
        plt.figure(figsize=(10,5))
        plt.plot(y_train[:100], label='Real')
        plt.plot(y_corrected[:100], label='Predito (outliers corrigidos)', alpha=0.7)
        plt.title(f"Valores reais vs previstos (p={p}, primeiras 100 amostras)")
        plt.xlabel("Amostra")
        plt.ylabel("Módulo aceleração")
        plt.legend()
        plt.show()

    return beta, error, y_corrected


def linear_model_centered_window(data_modules, p, outlier_density=10, k=3, z=2.0, plot_examples=True):
    """
    Modelo linear com janela centrada e múltiplos módulos (ex: acc, gyro, mag).
    data_modules: dicionário com {'acc': array, 'gyro': array, 'mag': array} (módulos)
    """
    half_p = p // 2
    y = data_modules['acc']  # queremos prever o módulo da aceleração

    # --- Construção da matriz X com janelas centradas ---
    n = len(y)
    X = []
    y_target = []

    # usa as três variáveis como features: acc, gyro, mag
    mod_acc = data_modules['acc']
    mod_gyro = data_modules['gyro']
    mod_mag = data_modules['mag']

    for i in range(half_p, n - half_p):
        window_acc = mod_acc[i - half_p:i + half_p + 1]
        window_gyro = mod_gyro[i - half_p:i + half_p + 1]
        window_mag = mod_mag[i - half_p:i + half_p + 1]

        features = np.concatenate([window_acc, window_gyro, window_mag])
        X.append(features)
        y_target.append(y[i])

    X = np.array(X)
    y_target = np.array(y_target)

    # --- Injeção de outliers e correção (reutiliza função anterior) --
    y_noisy = inject_outliers(y_target.copy(), x=outlier_density, k=k, z=z)

    # Encontra outliers via z-score
    z_scores = (y_noisy - np.mean(y_noisy)) / np.std(y_noisy)
    mask_outliers = np.abs(z_scores) > k

    # Ajuste modelo linear
    beta = linear_model(X, y_noisy)
    y_pred = X @ beta[1:] + beta[0]

    # Substitui outliers pelos valores previstos
    y_corrected = y_noisy.copy()
    y_corrected[mask_outliers] = y_pred[mask_outliers]

    # --- Análise do erro ---
    error = y_target - y_pred
    print(f"Erro médio: {np.mean(np.abs(error)):.4f} | Erro RMS: {np.sqrt(np.mean(error**2)):.4f}")

    # --- Plot ---
    if plot_examples:
        plt.figure(figsize=(12, 6))
        plt.plot(y_target[:500], label="Real")
        plt.plot(y_pred[:500], label="Previsto", linestyle="--")
        plt.title(f"Modelo Linear Centrado (p={p})")
        plt.xlabel("Amostra")
        plt.ylabel("Módulo Aceleração")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.hist(error, bins=50, color='gray', alpha=0.7)
        plt.title(f"Distribuição do Erro (p={p})")
        plt.xlabel("Erro")
        plt.ylabel("Frequência")
        plt.show()

    return beta, y_corrected, error

def compute_modulus_all(data_array, sensor='acc'):
    """
    Calcula o módulo da aceleração (√(x²+y²+z²)) para todas as pessoas e sensores.
    Retorna um único array concatenado com todos os módulos.
    """
    sensor_cols = {
        'acc': [1, 2, 3],
        'gyro': [4, 5, 6],
        'mag': [7, 8, 9]
    }
    cols = sensor_cols[sensor]
    all_mods = []

    for person_data in data_array:  # percorre todas as pessoas
        for sensor_data in person_data:  # percorre todos os sensores dessa pessoa
            vector = sensor_data[:, cols]  # colunas X, Y, Z do acelerómetro
            mod = np.linalg.norm(vector, axis=1)
            all_mods.append(mod)

    # Junta todos os módulos de todos os sensores/pessoas
    combined_mod = np.concatenate(all_mods)

    return combined_mod
