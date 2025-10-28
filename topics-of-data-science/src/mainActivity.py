from utils.io import readFiles
from core.metrics import activity_metric, zscore_outliers, k_mean, manual_kmeans, kmeans_outliers, dbscan_outliers, inject_outliers, linear_model, create_windows, linear_model_correction, linear_model_centered_window, compute_modulus_all
from core.features import statistical_significance, advanced_statistical_tests, compute_pca, pca_variance_analysis, top_features, build_feature_matrix, project_features
import config as cfg
import numpy as np
import matplotlib.pyplot as plt
import os

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
    #k_mean(data_array, sensor='mag', n_clusters=cfg.LABELS_COUNTER, save_dir="data/kmean", save_plots=True)
   
    #3.7
    #kmeans_outliers(data_array, sensor='mag', n_clusters=cfg.LABELS_COUNTER, save_dir="data/kmean_outliers", save_plots=True)

    #3.7.1
    #dbscan_outliers(data_array, sensor='mag',eps=0.5, min_samples=10, save_dir="data/dbscan", save_plots=True)
    '''
    #3.8
    modulo = compute_modulus_all(data_array, sensor="acc")
    
    # injeta outliers no módulo
    data_with_outliers = inject_outliers(modulo, x=10, k=3, z=2.0)

    #3.9
    X_train = np.random.rand(10, 3)
    y_train = np.random.rand(10)

    beta = linear_model(X_train, y_train)
    print("Vetor de pesos beta:", beta)
    
    #3.10
    best_p = None
    min_rmse = np.inf
    all_errors = {}
    rmse_values = {}
    
    save_dir = "data/linear_model_correction"
    os.makedirs(save_dir, exist_ok=True)

    for p_test in range(1, 101):
        beta, error, y_corrected = linear_model_correction(modulo, p=p_test, outlier_density=10, k=3, z=2.0, save_dir="data/linear_model_centered_window", save_plots=False)
        all_errors[p_test] = error
        
        rmse = np.sqrt(np.mean(error**2))
        rmse_values[p_test] = rmse
        #print(f"p={p_test}, RMSE={rmse:.4f}")
        if rmse < min_rmse:
            min_rmse = rmse
            best_p = p_test
    
    # --- Plot: RMSE em função de p ---
    plt.figure(figsize=(10, 5))
    plt.plot(list(rmse_values.keys()), list(rmse_values.values()), marker='o')
    plt.title("RMSE em função do tamanho da janela p")
    plt.xlabel("p (tamanho da janela)")
    plt.ylabel("RMSE")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("data/linear_model_correction/rmse_vs_p.png", dpi=200)
    plt.show()

    print(f"Melhor p: {best_p} (RMSE={min_rmse:.4f})")

    # Plot final com melhor p
    beta, error, y_corrected = linear_model_correction(modulo, p=best_p, outlier_density=10, k=3, z=2.0, save_dir="data/linear_model_centered_window", save_plots=True)

    #3.11
    # Extrai módulos de cada variável (todas as pessoas e sensores)
    acc = compute_modulus_all(data_array, sensor="acc")
    gyro = compute_modulus_all(data_array, sensor="gyro")
    mag = compute_modulus_all(data_array, sensor="mag")

    # Cria um dicionário com todos os módulos
    data_modules = {"acc": acc, "gyro": gyro, "mag": mag}

    # Testa vários valores de p (janela centrada)
    best_p = None
    min_rmse = np.inf

    for p in [5, 9, 15]:
        beta, y_corr, err = linear_model_centered_window(data_modules, p=p, outlier_density=10, k=3, z=2.0, save_dir="data/linear_model_centered_window", save_plots=True)

        rmse = np.sqrt(np.mean(err ** 2))
        print(f"p={p}, RMSE={rmse:.4f}")

        if rmse < min_rmse:
            min_rmse = rmse
            best_p = p

    print(f"\nMelhor p encontrado: {best_p} (RMSE={min_rmse:.4f})")
    '''
    #4
    """
    Pipeline completo de feature extraction e seleção.
    """
    body_parts = ["Left Wrist","Right Wrist","Chest","Right Upper Leg","Left Lower Leg"]
    sensors = ['acc','gyro','mag']

    for sensor in sensors:
        for idx, part in enumerate(body_parts):
            X, y, feat_names = build_feature_matrix(data_array, sensor=sensor,
                                                body_part_idx=idx, body_part_name=part)
            statistical_significance(data_array, sensor=sensor, body_part_idx=idx, body_part_name=part)
            advanced_statistical_tests(X, y, feat_names, sensor=sensor, body_part_name=part)
            X_pca, pca_model = compute_pca(X, sensor=sensor, body_part_name=part)
            n_feats = pca_variance_analysis(pca_model)
            X_proj = project_features(X_pca, n_feats)
            idx_top, scores_top = top_features(X_proj, y, method='fisher', top_k=10)
            print(f"[{sensor} | {part}] Top 10 features PCA+Fisher: {idx_top}")
    
    
if __name__ == '__main__':
    main()
