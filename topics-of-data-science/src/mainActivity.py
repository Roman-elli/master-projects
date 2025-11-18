from utils.io import readFiles, read_data_per_person
from core.data_split import splitFiles
from core.metrics import activity_metric, zscore_outliers, k_mean, manual_kmeans, kmeans_outliers, dbscan_outliers, inject_outliers, linear_model, create_windows, linear_model_correction, linear_model_centered_window, compute_modulus_all
from core.features import build_feature_matrix_activity, apply_feature_selection_to_sensor, statistical_significance, apply_pca_to_activity, apply_feature_selection_to_activity
from core.classifier import evaluate_and_save_metrics, baseline_classifier
import config as cfg
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def main():
    # 2
    #data_array = readFiles(cfg.ASSETS_FOLDERS_PATH)

    # # 3.1 & 3.2
    # results_acc = activity_metric(data_array, 'acc', save_dir="data/activity", save_plots=False)
    # results_gyro = activity_metric(data_array, 'gyro', save_dir="data/activity", save_plots=False)
    # results_mag = activity_metric(data_array, 'mag', save_dir="data/activity", save_plots=False)

    # # 3.3 & 3.4
    # useActivitie = False
    # for k_value in [3, 3.5, 4]:
    #     zscore_outliers(results_acc, sensor='acc', k=k_value, use_activities=useActivitie, save_dir="data/zscore")
    #     zscore_outliers(results_gyro, sensor='gyro', k=k_value, use_activities=useActivitie, save_dir="data/zscore")
    #     zscore_outliers(results_mag, sensor='mag', k=k_value, use_activities=useActivitie, save_dir="data/zscore")
    
    # #3.6
    # k_mean(data_array, sensor='mag', n_clusters=cfg.LABELS_COUNTER, save_dir="data/kmean", save_plots=True)
   
    # #3.7
    # kmeans_outliers(data_array, sensor='mag', n_clusters=cfg.LABELS_COUNTER, save_dir="data/kmean_outliers", save_plots=True)

    # #3.7.1
    # dbscan_outliers(data_array, sensor='mag',eps=0.5, min_samples=10, save_dir="data/dbscan", save_plots=True)
    
    # #3.8
    # modulo = compute_modulus_all(data_array, sensor="acc")
    
    # # injeta outliers no módulo
    # data_with_outliers = inject_outliers(modulo, x=10, k=3, z=2.0)

    # #3.9
    # X_train = np.random.rand(10, 3)
    # y_train = np.random.rand(10)

    # beta = linear_model(X_train, y_train)
    # print("Vetor de pesos beta:", beta)
    
    # #3.10
    # best_p = None
    # min_rmse = np.inf
    # all_errors = {}
    # rmse_values = {}
    
    # save_dir = "data/linear_model_correction"
    # os.makedirs(save_dir, exist_ok=True)

    # for p_test in range(1, 101):
    #     beta, error, y_corrected = linear_model_correction(modulo, p=p_test, outlier_density=10, k=3, z=2.0, save_dir="data/linear_model_centered_window", save_plots=False)
    #     all_errors[p_test] = error
        
    #     rmse = np.sqrt(np.mean(error**2))
    #     rmse_values[p_test] = rmse
    #     #print(f"p={p_test}, RMSE={rmse:.4f}")
    #     if rmse < min_rmse:
    #         min_rmse = rmse
    #         best_p = p_test
    
    # # --- Plot: RMSE em função de p ---
    # plt.figure(figsize=(10, 5))
    # plt.plot(list(rmse_values.keys()), list(rmse_values.values()), marker='o')
    # plt.title("RMSE em função do tamanho da janela p")
    # plt.xlabel("p (tamanho da janela)")
    # plt.ylabel("RMSE")
    # plt.grid(True, linestyle="--", alpha=0.6)
    # plt.tight_layout()
    # plt.savefig("data/linear_model_correction/rmse_vs_p.png", dpi=200)
    # plt.show()

    # print(f"Melhor p: {best_p} (RMSE={min_rmse:.4f})")

    # # Plot final com melhor p
    # beta, error, y_corrected = linear_model_correction(modulo, p=best_p, outlier_density=10, k=3, z=2.0, save_dir="data/linear_model_centered_window", save_plots=True)

    # #3.11
    # # Extrai módulos de cada variável (todas as pessoas e sensores)
    # acc = compute_modulus_all(data_array, sensor="acc")
    # gyro = compute_modulus_all(data_array, sensor="gyro")
    # mag = compute_modulus_all(data_array, sensor="mag")

    # # Cria um dicionário com todos os módulos
    # data_modules = {"acc": acc, "gyro": gyro, "mag": mag}

    # # Testa vários valores de p (janela centrada)
    # best_p = None
    # min_rmse = np.inf

    # for p in [5, 9, 15]:
    #     beta, y_corr, err = linear_model_centered_window(data_modules, p=p, outlier_density=10, k=3, z=2.0, save_dir="data/linear_model_centered_window", save_plots=True)

    #     rmse = np.sqrt(np.mean(err ** 2))
    #     print(f"p={p}, RMSE={rmse:.4f}")

    #     if rmse < min_rmse:
    #         min_rmse = rmse
    #         best_p = p

    # print(f"\nMelhor p encontrado: {best_p} (RMSE={min_rmse:.4f})")

    #4
    # calculate_PCA_and_top10 = True
    
    # if calculate_PCA_and_top10:
    #     for sensor in cfg.SENSORS:
    #         for part in cfg.BODY_PARTS_PATH:
    #             for act_id in cfg.ACTIVITIES:
    #                 #apply_pca_to_activity(sensor, part, act_id)
    #                 apply_feature_selection_to_activity(sensor, part, act_id)
    #         #print(f"\n=== Seleção de features -> Sensor: {sensor.upper()} ===")
    #         #fisher_all, relief_all = apply_feature_selection_to_sensor(sensor, cfg.BODY_PARTS_PATH, top_k=20)
    # else:
    #     data_array = readFiles(cfg.ASSETS_FOLDERS_PATH)

    #     for sensor in cfg.SENSORS:
    #         for idx, part in enumerate(cfg.BODY_PARTS):
    #             # Estatísticas gerais por sensor/parce (todas atividades)
    #             statistical_significance(data_array, sensor=sensor, body_part_idx=idx, body_part_name=part)

    #             for act_id in cfg.ACTIVITIES:
    #                 print(f"\n=== {sensor.upper()} | {part} | Atividade {act_id} ===")
    #                 X, feat_names = build_feature_matrix_activity(data_array, activity_id=act_id, sensor=sensor, fs=cfg.FS, window_ms=cfg.WINDOW_SIZE, overlap=cfg.OVERLAP, body_part_idx=idx, body_part_name=part)
                    
    #                 # salvar CSV
    #                 save_dir = os.path.join("data", "features", sensor, part.replace(" ", "_"), f"act_{act_id}")
    #                 os.makedirs(save_dir, exist_ok=True)
    #                 save_path = os.path.join(save_dir, "features.csv")

    #                 np.savetxt(save_path, X, delimiter=';', header=';'.join(feat_names), comments='', fmt='%.6f')
    #                 print(f"[OK] {sensor} | {part} | Atividade {act_id}: {len(X)} janelas salvas em {save_path}")

    """Parte B"""
    extract_data = False
    train_model = True

    if extract_data:
        data_array_per_person = read_data_per_person(cfg.ASSETS_FOLDERS_PATH)
        splitFiles(data_array_per_person, cfg.SPLIT_ASSETS_FOLDERS_PATH, save_tvt=True, save_kfold=False, kfold_save_item=2)
    

    if train_model:
        baseline_classifier()
        
if __name__ == '__main__':
    main()
