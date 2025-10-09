import matplotlib.pyplot as plt
import numpy as np

def activity_metric(data_array, sensor='acc'):
    sensor_cols = {
        'acc': [2, 3, 4],
        'gyro': [5, 6, 7],
        'mag': [8, 9, 10]
    }

    cols = sensor_cols[sensor]

    modules = []
    activities = []

    for person in data_array:
        for activity in person:
            x = activity[:, cols[0]]
            y = activity[:, cols[1]]
            z = activity[:, cols[2]]
            modulo = np.sqrt(x**2 + y**2 + z**2)
            col12 = activity[:, 11].astype(int)

            modules.extend(modulo)
            activities.extend(col12)

    modules = np.array(modules)
    activities = np.array(activities)

    unique_activities = np.unique(activities)
    data_activities = [modules[activities == a] for a in unique_activities]

    plt.figure(figsize=(10, 6))
    plt.boxplot(data_activities, labels=unique_activities)
    plt.xlabel("Activity (column 12)")
    plt.ylabel(f"Vector module ({sensor})")
    plt.title(f"Boxplot of {sensor} vector magnitude by activity")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # 3.2
    print(f"Outliers density for {sensor}:")
    for a in unique_activities:
        activity_data = modules[activities == a]
        n_r = len(activity_data)
        q1 = np.percentile(activity_data, 25)
        q3 = np.percentile(activity_data, 75)
        iqr = q3 - q1
        low_limit = q1 - 1.5 * iqr
        sup_limit = q3 + 1.5 * iqr

        outliers = np.sum((activity_data < low_limit) | (activity_data > sup_limit))
        
        d = (outliers / n_r) * 100
        print(f"Activity {a}: {d:.2f}% outliers ({outliers}/{n_r})")

    print("\n***********************************************\n")
    
    plt.show()

def zscore_outliers(data_array, sensor='acc', k=3):
    sensor_cols = {
        'acc': [1, 2, 3],
        'gyro': [4, 5, 6],
        'mag': [7, 8, 9]
    }

    cols = sensor_cols[sensor]
    modules = []
    activities = []

    for individuo in data_array:
        for atividade in individuo:
            x, y, z = atividade[:, cols].T
            modulo = np.sqrt(x**2 + y**2 + z**2)
            modules.extend(modulo)
            activities.extend(atividade[:, 11].astype(int))  # coluna 12

    modules = np.array(modules)
    activities = np.array(activities)
    unique_activities = np.unique(activities)

    plt.figure(figsize=(12, 6))

    for a in unique_activities:
        dados_atividade = modules[activities == a]
        mean = np.mean(dados_atividade)
        std = np.std(dados_atividade)
        z_scores = (dados_atividade - mean) / std

        mask_outliers = np.abs(z_scores) > k

        plt.scatter([a]*len(dados_atividade[~mask_outliers]), dados_atividade[~mask_outliers],
                    color='blue', alpha=0.6, label='Normal' if a==unique_activities[0] else "")

        plt.scatter([a]*len(dados_atividade[mask_outliers]), dados_atividade[mask_outliers],
                    color='red', alpha=0.8, label='Outlier' if a==unique_activities[0] else "")

    plt.xlabel("Activity (Column 12)")
    plt.ylabel(f"{sensor.upper()} Vector Magnitude")
    plt.title(f"{sensor.upper()} Vector Magnitude with Z-score Outliers (k={k})")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
