from utils.io import readFiles
from core.metrics import activity_metric, zscore_outliers, k_mean, manual_kmeans, kmeans_outliers, dbscan_outliers, inject_outliers, linear_model, create_windows, linear_model_correction, linear_model_centered_window
import config as cfg
import numpy as np

def main():
    # 2
    data_array = readFiles(cfg.ASSETS_FOLDERS_PATH)

    # 3.1 & 3.2
    # results_acc = activity_metric(data_array, 'acc', save_dir="data/activity", save_plots=False)
    # results_gyro = activity_metric(data_array, 'gyro', save_dir="data/activity", save_plots=False)
    # results_mag = activity_metric(data_array, 'mag', save_dir="data/activity", save_plots=False)

    # 3.3 & 3.4
    # useActivitie = False
    # for k_value in [3, 3.5, 4]:
    #     zscore_outliers(results_acc, sensor='acc', k=k_value, use_activities=useActivitie, save_dir="data/zscore")
    #     zscore_outliers(results_gyro, sensor='gyro', k=k_value, use_activities=useActivitie, save_dir="data/zscore")
    #     zscore_outliers(results_mag, sensor='mag', k=k_value, use_activities=useActivitie, save_dir="data/zscore")
    
    #3.6
    k_mean(data_array, sensor='mag', n_clusters=cfg.LABELS_COUNTER, save_dir="data/kmean", save_plots=True)
   
   #3.7
    kmeans_outliers(data_array, sensor='mag', n_clusters=cfg.LABELS_COUNTER, save_dir="data/kmean_outliers", save_plots=True)

    #3.7.1
    dbscan_outliers(data_array, sensor='mag',
        eps=0.5,          # ajusta conforme o teu dataset
        min_samples=10,   # idem
        save_dir="data/dbscan",
        save_plots=True
    )
    
    #3.8
    # Exemplo: dados da aceleração (colunas 1, 2, 3)
    sensor = 'acc'
    acc_sensor = data_array[0][0][:, 1:4]   # X, Y, Z do acelerómetro

    # Módulo da aceleração (raiz quadrada da soma dos quadrados)
    modulo_acc = np.linalg.norm(acc_sensor, axis=1)


    # injeta outliers no módulo
    data_with_outliers = inject_outliers(modulo_acc, x=10, k=3, z=2.0)

    #3.9
    X_train = np.random.rand(10, 3)
    y_train = np.random.rand(10)

    beta = linear_model(X_train, y_train)
    print("Vetor de pesos beta:", beta)
    
    #3.10
    best_p = None
    min_rmse = np.inf

    for p_test in range(1, 11):
        beta, error, y_corrected = linear_model_correction(
            modulo_acc,
            p=p_test,
            outlier_density=10,
            k=3,
            z=2.0,
            plot_examples=False  # para não gerar todos os plots
        )

        rmse = np.sqrt(np.mean(error**2))
        print(f"p={p_test}, RMSE={rmse:.4f}")
        if rmse < min_rmse:
            min_rmse = rmse
            best_p = p_test

    print(f"Melhor p: {best_p} (RMSE={min_rmse:.4f})")

    # Plot final com melhor p
    beta, error, y_corrected = linear_model_correction(
        modulo_acc,
        p=best_p,
        outlier_density=10,
        k=3,
        z=2.0,
        plot_examples=True
    )

    #3.11
        # Extrai módulos das variáveis
    acc = np.linalg.norm(data_array[0][0][:, 1:4], axis=1)
    gyro = np.linalg.norm(data_array[0][0][:, 4:7], axis=1)
    mag = np.linalg.norm(data_array[0][0][:, 7:10], axis=1)

    data_modules = {'acc': acc, 'gyro': gyro, 'mag': mag}

    # Executa exercício 3.11
    for p in [5, 9, 15]:
        print(f"\nExecutando modelo linear centrado com p={p}")
        beta, y_corr, err = linear_model_centered_window(
            data_modules,
            p=p,
            outlier_density=10,
            k=3,
            z=2.0,
            plot_examples=True
        )

if __name__ == '__main__':
    main()
