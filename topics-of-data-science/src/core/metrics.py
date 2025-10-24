import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

import os
import numpy as np
import matplotlib.pyplot as plt

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