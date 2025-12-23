import matplotlib.pyplot as plt
import config as cfg
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

from sklearn.cluster import DBSCAN

def activity_metric(data_array, sensor='acc', save_dir="data/activity", save_plots=True):
    cols = cfg.SENSOR_COLS[sensor] # colunas correspondentes ao sensor selecionado
    n_sensors = len(data_array[0]) # nº de sensores por pessoa

    os.makedirs(save_dir, exist_ok=True)
    outlier_file = os.path.join(save_dir, f"{sensor}_outliers.txt")

    with open(outlier_file, "w") as f:
        results = []
        # Loop sobre cada sensor 
        for sen_idx in range(n_sensors):
            all_modules = []
            all_activities = []
            # Loop sobre cada pessoa
            for person in data_array:
                activity = person[sen_idx]
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

            unique_activities = np.unique(all_activities) # Identifica as atividades únicas
            data_activities = [all_modules[all_activities == a] for a in unique_activities] # Separa os modulos por atividade

            f.write(f"Outliers density for {sensor} | {cfg.BODY_PARTS[sen_idx]}:\n")
            print(f"Outliers density for {sensor} | {cfg.BODY_PARTS[sen_idx]}:")

            # Loop por cada atividade
            for a in unique_activities:
                activity_data = all_modules[all_activities == a] # Dados da atividade
                n_r = len(activity_data) # Nº de amostras
                # Calcula o intervalo interquartil
                q1 = np.percentile(activity_data, 25)
                q3 = np.percentile(activity_data, 75)
                iqr = q3 - q1
                # Define limites inferior e superior para outliers
                low_limit = q1 - 1.5 * iqr
                sup_limit = q3 + 1.5 * iqr

                outliers = np.sum((activity_data < low_limit) | (activity_data > sup_limit)) # Quantos outliers estao fora dos limites
                d = (outliers / n_r) * 100

                line = f"  Activity {a}: {d:.2f}% outliers ({outliers}/{n_r})\n"
                f.write(line)
                print(line.strip())

            f.write("\n---------------------------------------------\n\n")
            print("\n---------------------------------------------\n")

            if save_plots:
                plot_folder = os.path.join(save_dir, sensor, cfg.BODY_PARTS[sen_idx])
                os.makedirs(plot_folder, exist_ok=True)
                plt.figure(figsize=(10, 6))
                plt.boxplot(data_activities, labels=unique_activities)
                plt.xlabel("Activity (column 12)")
                plt.ylabel(f"Vector module ({sensor})")
                plt.title(f"{sensor.upper()} | {cfg.BODY_PARTS[sen_idx]}")
                plt.grid(True, axis='y', linestyle='--', alpha=0.7)
                plt.savefig(os.path.join(plot_folder, f"{cfg.BODY_PARTS[sen_idx]}_boxplot.png"))
                plt.close()

    return results

def zscore_outliers(results, sensor='acc', k=3, use_activities=True, save_dir="data/zscore"):

    k_folder = os.path.join(save_dir, f"k={k}", "activity" if use_activities else "noActivity")
    os.makedirs(k_folder, exist_ok=True)

    outlier_file = os.path.join(k_folder, f"{sensor}_outliers.txt")
    with open(outlier_file, "w") as f:
        # Loop por cada sensor 
        for sen_idx, (modules, activities) in enumerate(results, start=1):
            sensor_name = cfg.BODY_PARTS[sen_idx - 1]

            f.write(f"Outliers density for {sensor.upper()} | {sensor_name} (Z-score method, k={k}):\n")
            print(f"Outliers density for {sensor.upper()} | {sensor_name} (Z-score method, k={k}):")

            plt.figure(figsize=(12, 6))
            # Se for para usar atividades separadas
            if use_activities:
                unique_activities = np.unique(activities)
                # Analisa cada atividade separadamente
                for a in unique_activities:
                    dados_atividade = modules[activities == a]
                    mean = np.mean(dados_atividade)
                    std = np.std(dados_atividade)
                    z_scores = (dados_atividade - mean) / std

                    # Cria máscara de outliers |z| > k
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
            # Caso nao use atividades (dados tratados como um todo)
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

    # Escolha aleatoria 'n_clusters' pontos do conjunto como centróides iniciais
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
    cols = cfg.SENSOR_COLS[sensor]
    n_sensors = len(data_array[0])

    for sen_idx in range(n_sensors):
        sensor_name = cfg.BODY_PARTS[sen_idx]

        all_xyz = []
        # Combina os dados de todas as pessoas para esse sensor
        for person in data_array:
            data_sensor = person[sen_idx]
            x = data_sensor[:, cols[0]]
            y = data_sensor[:, cols[1]]
            z = data_sensor[:, cols[2]]
            xyz = np.column_stack((x, y, z))
            all_xyz.append(xyz)
        all_xyz = np.vstack(all_xyz) # União de todos os daods de todas as pessoas numa matriz
        
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

        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='black', s=100, marker='X', label='Centroids')

        ax.set_title(f"{sensor.upper()} | {sensor_name} | Manual K-Means (k={n_clusters})")
        ax.set_xlabel(f"{sensor.upper()} X")
        ax.set_ylabel(f"{sensor.upper()} Y")
        ax.set_zlabel(f"{sensor.upper()} Z")

        if save_plots:
            folder_path = os.path.join(save_dir, sensor, sensor_name, f"k={n_clusters}")
            os.makedirs(folder_path, exist_ok=True)
            file_path = os.path.join(folder_path, f"{sensor_name}_k={n_clusters}.png")
            plt.savefig(file_path, dpi=200)
            plt.close(fig)  # Fecha o plot para não ocupar memória
            
def kmeans_outliers(data_array, sensor='acc', n_clusters=3, save_dir="data/kmean_outliers", save_plots=True):
    cols = cfg.SENSOR_COLS[sensor]
    n_sensors = len(data_array[0])

    # Percorre cada posição do sensor
    for med_idx in range(n_sensors):
        sensor_name = cfg.BODY_PARTS[med_idx]
        all_xyz = []

        # Junta os dados de todas as pessoas para esse sensor
        for person in data_array:
            data_sensor = person[med_idx]
            x = data_sensor[:, cols[0]]
            y = data_sensor[:, cols[1]]
            z = data_sensor[:, cols[2]]
            xyz = np.column_stack((x, y, z))
            all_xyz.append(xyz)

        X = np.vstack(all_xyz)
        labels, centroids = manual_kmeans(X, n_clusters=n_clusters, max_iter=100, tol=1e-4)

        # Calcula a distância de cada ponto ao centro do cluster ao qual pertence
        distances = np.linalg.norm(X - centroids[labels], axis=1)
        outliers = np.zeros(len(X), dtype=bool)
        k = 3  # Limiar z-score

        # Para cada cluster, calcula o z-score das distâncias e marca outliers
        for c in np.unique(labels):
            mask = labels == c
            cluster_dist = distances[mask]
            mean = np.mean(cluster_dist)
            std = np.std(cluster_dist)
            z = (cluster_dist - mean) / std
            outliers[mask] = np.abs(z) > k # Pontos distantes são outliers

        total_outliers = np.sum(outliers)
        perc = total_outliers / len(X) * 100
        
        # Salva resultados
        folder_path = os.path.join(save_dir, sensor, sensor_name, f"k={n_clusters}")
        os.makedirs(folder_path, exist_ok=True)

        txt_path = os.path.join(folder_path, f"{sensor_name}_outliers_k={n_clusters}.txt")
        with open(txt_path, "w") as f:
            f.write(f"K-Means Outliers ({sensor.upper()} - {sensor_name})\n")
            f.write(f"Clusters (k): {n_clusters}\n")
            f.write(f"Total: {len(X)} amostras\n")
            f.write(f"Outliers: {total_outliers} ({perc:.2f}%)\n\n")
            for c in np.unique(labels):
                cluster_mask = labels == c
                cluster_outliers = np.sum(outliers[cluster_mask])
                total_cluster = np.sum(cluster_mask)
                f.write(f"Cluster {c}: {cluster_outliers}/{total_cluster} "
                        f"({(cluster_outliers/total_cluster)*100:.2f}%)\n")

        if save_plots:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(X[~outliers, 0], X[~outliers, 1], X[~outliers, 2],
                       c=labels[~outliers], cmap='tab10', s=8, alpha=0.6)
            ax.scatter(X[outliers, 0], X[outliers, 1], X[outliers, 2],
                       c='red', s=25, label='Outliers', alpha=0.8)
            ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                       c='black', s=100, marker='X', label='Centroids')

            ax.set_title(f"{sensor.upper()} | {sensor_name} | K-Means Outliers (k={n_clusters})")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.legend()

            img_path = os.path.join(folder_path, f"{sensor_name}_k={n_clusters}.png")
            plt.tight_layout()
            plt.savefig(img_path, dpi=200)
            plt.close(fig)

def dbscan_outliers(data_array, sensor='mag', eps=0.5, min_samples=10, save_dir="data/dbscan", save_plots=True):
    cols = cfg.SENSOR_COLS[sensor]

    n_sensors = len(data_array[0])

    for med_idx in range(n_sensors):
        sensor_name = cfg.BODY_PARTS[med_idx]
        all_xyz = []

        # junta dados de todas as pessoas para esse sensor
        for person in data_array:
            data_sensor = person[med_idx]
            x = data_sensor[:, cols[0]]
            y = data_sensor[:, cols[1]]
            z = data_sensor[:, cols[2]]
            xyz = np.column_stack((x, y, z))
            all_xyz.append(xyz)

        X = np.vstack(all_xyz)

        # Amostra se for muito grande
        if len(X) > 50000:
            idx = np.random.choice(len(X), 50000, replace=False)
            X = X[idx]

        # Aplica DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        labels = db.labels_

        # Conta o número de clusters e de outliers (rotulados como -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_outliers = np.sum(labels == -1)
        perc_out = (n_outliers / len(X)) * 100

        folder_path = os.path.join(save_dir, sensor, sensor_name)
        os.makedirs(folder_path, exist_ok=True)

        out_file = os.path.join(folder_path, f"{sensor_name}_dbscan_results.txt")
        with open(out_file, "w") as f:
            f.write(f"DBSCAN ({sensor.upper()} - {sensor_name})\n")
            f.write(f"eps={eps}, min_samples={min_samples}\n")
            f.write(f"Clusters encontrados: {n_clusters}\n")
            f.write(f"Total de amostras: {len(X)}\n")
            f.write(f"Outliers: {n_outliers} ({perc_out:.2f}%)\n")

        if save_plots:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            unique_labels = set(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

            for label, color in zip(unique_labels, colors):
                mask = labels == label
                if label == -1:
                    color = 'red'
                    label_name = 'Outliers'
                else:
                    label_name = f'Cluster {label}'
                ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2],
                           c=[color], s=8, alpha=0.6, label=label_name)

            ax.set_title(f"{sensor.upper()} | {sensor_name} | DBSCAN Clusters & Outliers")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.legend(loc='best')
            plt.tight_layout()

            plot_path = os.path.join(folder_path, f"{sensor_name}_dbscan_outliers.png")
            plt.savefig(plot_path, dpi=200)
            plt.close()

def inject_outliers(array, x=5.0, k=3, z=1.0, random_seed=42):
    np.random.seed(random_seed)
    array = array.copy()
    
    mu = np.mean(array)
    sigma = np.std(array)
    
    # Calcula o z-score de cada ponto (distância em desvios padrão da média)
    z_scores = (array - mu) / sigma
    
    # Identifica os pontos que já são outliers (|z| > k)
    existing_outliers = np.abs(z_scores) > k
    n_outliers = np.sum(existing_outliers)
    n_total = len(array)
  
    # Densidade atual de outliers
    d = (n_outliers / n_total) * 100
    #print(f"Densidade atual de outliers: {d:.2f}%")
    
    if d < x:
        # quantidade de outliers a ser inseridos
        n_needed = int(np.ceil((x - d)/100 * n_total))
        
        # índices de pontos que ainda não são outliers
        non_outlier_indices = np.where(~existing_outliers)[0]
        
        # Escolhe aleatoriamente quais pontos serão transformados em outliers
        selected_indices = np.random.choice(non_outlier_indices, size=n_needed, replace=False)
        
        # Para cada ponto selecionado, gera um valor fora da faixa normal
        for idx in selected_indices:
            s = np.random.choice([-1, 1]) # Outlier positivo ou negativo
            q = np.random.uniform(0, z) # Fator aleatório pequeno para variar a intensidade
            # Desloca o valor além do limiar definido por k * sigma
            array[idx] = mu + s * (k * sigma + q)
            
        #print(f"Injetados {n_needed} outliers para atingir densidade {x}%")
    #else:
        #print("Densidade já >= x%, não foi necessário injetar outliers")
    
    return array

def linear_model(X_train, y_train):
    n_samples = X_train.shape[0]
    # Adiciona uma coluna de 1 ao início da matriz 
    X_aug = np.hstack((np.ones((n_samples, 1)), X_train))  # dimensão: (n amostras, p+1)

    # Calcula o produto 
    XtX = X_aug.T @ X_aug
    Xty = X_aug.T @ y_train

    # Resolve analiticamente o modelo linear que melhor se ajusta aos dados
    beta = np.linalg.inv(XtX) @ Xty

    return beta

def create_windows(data, p):
    X = []
    y = []
    
    # Percorre o vetor de dados a partir da posição p
    # A cada passo, cria uma janela com os p valores anteriores e o valor atual como alvo
    for i in range(p, len(data)):
        X.append(data[i-p:i])  # p valores anteriores
        y.append(data[i])      # valor atual
    return np.array(X), np.array(y)

def linear_model_correction(modulo, p=3, outlier_density=10, k=3, z=2.0, save_dir="data/linear_model_correction", save_plots=True):
    os.makedirs(save_dir, exist_ok=True)
    # Cria uma copia dos dados e injeta outliers
    data_with_outliers = inject_outliers(modulo, x=outlier_density, k=k, z=z)
   
    # Cria as janelas com p valores anteriores
    X_train, y_train = create_windows(data_with_outliers, p)
   
    # Treina o modelo linear
    beta = linear_model(X_train, y_train)
   
    # Calcula as previsões do modelo linear
    y_pred = np.hstack((np.ones((X_train.shape[0], 1)), X_train)) @ beta
   
    # Identifica outliers no vetor y_train com base no z-score
    mu = np.mean(y_train)
    sigma = np.std(y_train)
    z_scores = (y_train - mu) / sigma
    outliers_mask = np.abs(z_scores) > k

    # Substitui os valores outliers pelas previsões do modelo
    y_corrected = y_train.copy()
    y_corrected[outliers_mask] = y_pred[outliers_mask]

    # Calcula o erro apenas nos pontos corrigidos
    error = y_corrected[outliers_mask] - y_train[outliers_mask]
    rmse = np.sqrt(np.mean(error**2))
    '''
    out_file = os.path.join(save_dir, f"summary_p{p}.txt")
    with open(out_file, "w") as f:
        f.write(f"Linear Model Correction\n")
        f.write(f"p={p}, outlier_density={outlier_density}, k={k}, z={z}\n")
        f.write(f"Número de amostras: {len(y_train)}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"Nº outliers corrigidos: {np.sum(outliers_mask)}\n")
    '''
    # Plots
    if save_plots:
        # Comparação real vs corrigido
        fig2 = plt.figure(figsize=(10, 5))
        plt.plot(y_train[:500], label='Real')
        plt.plot(y_corrected[:500], label='Corrigido', alpha=0.7)
        plt.title(f"Real vs Corrigido (p={p})")
        plt.xlabel("Amostra")
        plt.ylabel("Módulo Aceleração")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"real_vs_corrigido1_p{p}.png"), dpi=200)
        plt.close(fig2)
        
        # Comparação Real vs Previsto
        fig3 = plt.figure(figsize=(10, 5))
        plt.plot(y_train[:500], label='Real')
        plt.plot(y_pred[:500], label='Previsto', linestyle='--')
        plt.title(f"Real vs Previsto (p={p})")
        plt.xlabel("Amostra")
        plt.ylabel("Módulo Aceleração")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"real_vs_previsto1_p{p}.png"), dpi=200)
        plt.close(fig3)
        
        # Distribuição do erro de predição
        plt.figure(figsize=(10, 5))
        plt.hist(error, bins=16, color='royalblue', edgecolor='black', alpha=0.9)
        plt.title(f"Distribuição do Erro de Predição (p={p})", fontsize=13)
        plt.xlabel("Erro de Predição", fontsize=12)
        plt.ylabel("Frequência", fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.6)

        max_abs = np.max(np.abs(error))
        plt.xlim(-max_abs, max_abs)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"erro_distribuicao_p{p}.png"), dpi=200)
        plt.show()

    return beta, error, y_corrected

def linear_model_centered_window(data_modules, p, outlier_density=10, k=3, z=2.0, save_dir="data/linear_model_centered_window", save_plots=True):
    os.makedirs(save_dir, exist_ok=True)   
    half_p = p // 2
    y = data_modules['acc']  # Módulo da aceleração

    n = len(y)
    X = []
    y_target = []

    mod_acc = data_modules['acc']
    mod_gyro = data_modules['gyro']
    mod_mag = data_modules['mag']

    # Cria janelas centradas (valores antes e depois do ponto a prever)
    for i in range(half_p, n - half_p):
        window_acc = mod_acc[i - half_p:i + half_p + 1]
        window_gyro = mod_gyro[i - half_p:i + half_p + 1]
        window_mag = mod_mag[i - half_p:i + half_p + 1]

        # Junta todas as informações numa única janela
        features = np.concatenate([window_acc, window_gyro, window_mag])
        X.append(features)
        y_target.append(y[i])

    X = np.array(X)
    y_target = np.array(y_target)

    # Injeta outliers no sinal alvo
    y_noisy = inject_outliers(y_target.copy(), x=outlier_density, k=k, z=z)
    z_scores = (y_noisy - np.mean(y_noisy)) / np.std(y_noisy)
    mask_outliers = np.abs(z_scores) > k

    # Treina o modelo linear com base nas janelas centradas
    beta = linear_model(X, y_noisy)
    
    # Gera previsões e substitui os outliers pelos valores previstos
    y_pred = X @ beta[1:] + beta[0]
    y_corrected = y_noisy.copy()
    y_corrected[mask_outliers] = y_pred[mask_outliers]

    error = y_target - y_pred
    rmse = np.sqrt(np.mean(error**2))
  
    out_file = os.path.join(save_dir, f"summary_p{p}.txt")
    with open(out_file, "w") as f:
        f.write(f"Linear Model Centered Window\n")
        f.write(f"p={p}, outlier_density={outlier_density}, k={k}, z={z}\n")
        f.write(f"Número de amostras: {len(y_target)}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"Nº outliers corrigidos: {np.sum(mask_outliers)}\n")

    # Plots
    if save_plots:
        # Real vs previsto
        fig1 = plt.figure(figsize=(12, 6))
        plt.plot(y_target[:500], label="Real")
        plt.plot(y_pred[:500], label="Previsto", linestyle="--")
        plt.title(f"Modelo Linear Centrado (p={p})")
        plt.xlabel("Amostra")
        plt.ylabel("Módulo Aceleração")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"real_vs_previsto_p{p}.png"), dpi=200)
        plt.close(fig1)

        # Distribuição do erro
        fig2 = plt.figure(figsize=(8, 5))
        plt.hist(error, bins=50, color='gray', alpha=0.7)
        plt.title(f"Distribuição do Erro (p={p})")
        plt.xlabel("Erro")
        plt.ylabel("Frequência")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"erro_distribuicao_p{p}.png"), dpi=200)
        plt.close(fig2)

    return beta, y_corrected, error

def compute_modulus_all(data_array, sensor='acc'):
    cols = cfg.SENSOR_COLS[sensor]
    all_mods = []

    for person_data in data_array:  # percorre todas as pessoas
        for sensor_data in person_data:  # percorre todos os sensores dessa pessoa
            vector = sensor_data[:, cols]  # colunas X, Y, Z do acelerómetro
            mod = np.linalg.norm(vector, axis=1)
            all_mods.append(mod)

    # Junta todos os módulos de todos os sensores/pessoas
    combined_mod = np.concatenate(all_mods)

    return combined_mod

# No final de core/metrics.py

def clean_outliers_zscore(person_matrix, k=4.0):
    """
    Recebe a matriz completa da pessoa (N linhas x 33 colunas).
    Percorre os dados dos sensores (colunas 2 a 31).
    Se encontrar um valor com Z-Score > k, substitui por interpolação.
    """
    # Faz cópia para não estragar a original
    cleaned = person_matrix.copy()
    
    # Índices das colunas de dados (Ignora ID, TS e Label)
    start_col = 2
    end_col = 32
    
    count = 0
    
    for col in range(start_col, end_col):
        signal = cleaned[:, col]
        
        # 1. Cálculo Estatístico
        mean = np.mean(signal)
        std = np.std(signal)
        
        if std < 1e-6: continue # Sensor parado, salta
            
        z_scores = np.abs((signal - mean) / std)
        
        # 2. Identificar Outliers
        outliers_mask = z_scores > k
        
        if np.any(outliers_mask):
            count += np.sum(outliers_mask)
            
            # 3. Interpolação (Correção)
            # x_all: índices de 0 a N
            x_all = np.arange(len(signal))
            
            # x_good: índices onde NÃO há outliers
            x_good = x_all[~outliers_mask]
            y_good = signal[~outliers_mask]
            
            # Preenche os locais ruins interpolando pelos vizinhos bons
            cleaned[outliers_mask, col] = np.interp(x_all[outliers_mask], x_good, y_good)
            
    if count > 0:
        print(f"   [Limpeza Z-Score] Corrigidos {count} pontos nesta pessoa.")
        
    return cleaned
