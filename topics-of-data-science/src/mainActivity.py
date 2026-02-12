import os
import numpy as np
import config as cfg
import matplotlib as plt
import pandas as pd
import datetime

# --- IMPORTS UTILS ---
from utils.io import readFiles, read_multisensor_data, read_data_per_person, get_next_version_dir, load_best_features_from_version
from core.data_split import create_dataset_final

# --- IMPORTS ANALYTICS & METRICS ---
from core.metrics import (
    activity_metric, zscore_outliers, k_mean, kmeans_outliers, dbscan_outliers, 
    inject_outliers, linear_model, linear_model_correction, 
    linear_model_centered_window, compute_modulus_all, clean_outliers_zscore, clean_outliers_kmeans
)

# --- IMPORTS FEATURES ---
from core.features import (
    build_feature_matrix_activity, statistical_significance, 
    apply_pca_to_activity, apply_feature_selection_to_activity, 
    extract_features_550, get_feature_names_550
)

# --- IMPORTS CLASSIFIERS ---
from core.classifier import baseline_classifier, knn_analysis, relieff_tvt, mlp_experiment
from core.neuro_net import run_custom_mlp

def main():
    run_part_A = False
    run_part_B = True
    
    # =================================================================
    # PARTE A: ANÁLISE ESTATÍSTICA E FEATURE SELECTION
    # =================================================================
    if run_part_A:
        print("\n=== EXECUTANDO PARTE A ===")
        # 2
        print("Carregando dados brutos...")
        data_array = readFiles(cfg.ASSETS_FOLDERS_PATH)

        # -------------------------------------------------------------
        # A.1 Métricas Básicas e Outliers (Exercícios 3.1 - 3.7)
        # -------------------------------------------------------------
        # 3.1 & 3.2
        run_basic_metrics = False
        if run_basic_metrics:
            print("--- [A.1] Análise de Outliers e Clustering ---")

            # Boxplots
            results_acc = activity_metric(data_array, 'acc', save_dir="data/activity", save_plots=False)
            results_gyro = activity_metric(data_array, 'gyro', save_dir="data/activity", save_plots=False)
            results_mag = activity_metric(data_array, 'mag', save_dir="data/activity", save_plots=False)

            # 3.3 & 3.4
            useActivitie = False
            for k_value in [3, 3.5, 4]:
                zscore_outliers(results_acc, sensor='acc', k=k_value, use_activities=useActivitie, save_dir="data/zscore")
                zscore_outliers(results_gyro, sensor='gyro', k=k_value, use_activities=useActivitie, save_dir="data/zscore")
                zscore_outliers(results_mag, sensor='mag', k=k_value, use_activities=useActivitie, save_dir="data/zscore")
            
            # Clustering
            # 3.6
            k_mean(data_array, sensor='mag', n_clusters=cfg.LABELS_COUNTER, save_dir="data/kmean", save_plots=True)

            # 3.7
            kmeans_outliers(data_array, sensor='mag', n_clusters=cfg.LABELS_COUNTER, save_dir="data/kmean_outliers", save_plots=True)

            # 3.7.1
            dbscan_outliers(data_array, sensor='mag',eps=0.5, min_samples=10, save_dir="data/dbscan", save_plots=True)

        # -------------------------------------------------------------
        # A.2 Modelo Linear (Exercícios 3.8 - 3.11)
        # -------------------------------------------------------------
        run_linear = False
        if run_linear:
            print("--- [A.2] Experiência Modelo Linear ---")
            #3.8
            modulo = compute_modulus_all(data_array, sensor="acc")

            # injeta outliers no módulo
            data_with_outliers = inject_outliers(modulo, x=10, k=3, z=2.0)

            # 3.9
            X_train = np.random.rand(10, 3)
            y_train = np.random.rand(10)

            beta = linear_model(X_train, y_train)
            print("Vetor de pesos beta:", beta)

            # 3.10
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

            # 3.11
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

        # -------------------------------------------------------------
        # A.3 Extração e Seleção (Exercício 4)
        # -------------------------------------------------------------
        calculate_PCA_and_top10 = False
        if calculate_PCA_and_top10:
            print("\n--- [A.3] Extração de Features ---")
            
            FS = cfg.FS
            WINDOW_MS = cfg.WINDOW_SIZE
            OVERLAP = cfg.OVERLAP
            sensor_label = 'combined_acc_gyro'

            # EXTRAÇÃO E SALVAMENTO ---
            for idx, part in enumerate(cfg.BODY_PARTS): 
                statistical_significance(data_array, body_part_idx=idx, body_part_name=part)
                for act_id in cfg.ACTIVITIES:
                    X, feat_names = build_feature_matrix_activity(
                        data_array, activity_id=act_id, fs=FS, window_ms=WINDOW_MS, 
                        overlap=OVERLAP, body_part_idx=idx, body_part_name=part, sensor=sensor_label
                    )
                    
                    save_dir = os.path.join("data", "features_analysis", sensor_label, part.replace(" ", "_"), f"act_{act_id}")
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, "features.csv")

                    if len(X) > 0:
                        np.savetxt(save_path, X, delimiter=';', header=';'.join(feat_names), comments='', fmt='%.6f')

            print("   Extração concluída. Iniciando Análise...")

            # ANÁLISE (PCA / RELIEF) ---
            for idx, part in enumerate(cfg.BODY_PARTS): 
                clean_part = part.replace(" ", "_")
                for act_id in cfg.ACTIVITIES:
                    apply_pca_to_activity(sensor_label, clean_part, act_id)
                    apply_feature_selection_to_activity(sensor_label, clean_part, act_id)
    
    # =================================================================
    # PARTE B: PIPELINE COMPLETO DE ML (EXTRAÇÃO -> DATASET -> TREINO)
    # =================================================================
    if run_part_B:
        print("\n=== EXECUTANDO PARTE B ===")
        
        # 1. Configurar Diretório de Resultados
        results_dir = get_next_version_dir()
        os.makedirs(results_dir, exist_ok=True)
        print(f" -> Resultados serão salvos em: {results_dir}")

        # --- CONFIGURAÇÕES DE EXPERIÊNCIA ---
        calculate_features = False
        split_dataset = True
        run_models = True
        
        # Método para tratamento de outliers
        CLEANING_METHOD  = cfg.CLEANING_METHOD
        
        # Z-Score setup
        Z_SCORE_VAL = cfg.Z_SCORE_VALUE
        
        # K-Means setup
        KMEANS_CLUSTERS  = cfg.LABELS_COUNTER
        KMEANS_THRESHOLD = cfg.KMEANS_THRESHOLD_VALUE
        
        # Método de tratamento aos sensores com números diferentes de amostras
        SAMPLING_METHOD  = cfg.SAMPLING_METHOD

        # -------------------------------------------------------------
        # 1. EXTRAÇÃO DE FEATURES
        # -------------------------------------------------------------
        if calculate_features:
            print(f"1. Extração (Limpeza: {CLEANING_METHOD})...")
            
            data_array = read_multisensor_data(cfg.ASSETS_FOLDERS_PATH)
            feat_names = get_feature_names_550() 
            FS = cfg.FS
            window_size = int(FS * (cfg.WINDOW_SIZE / 1000))
            step_size = int(window_size * (1 - cfg.OVERLAP))
            IDX_LABEL = -1 
            save_root = "data/features_550_final"

            for p_idx, person_matrix in enumerate(data_array, start=1):
                if person_matrix is None or len(person_matrix) == 0: continue
                
                # === [NOVO] LÓGICA DE LIMPEZA DINÂMICA ===
                if CLEANING_METHOD == 'ZSCORE':
                    person_matrix = clean_outliers_zscore(person_matrix, k=Z_SCORE_VAL)
                elif CLEANING_METHOD == 'KMEANS':
                    person_matrix = clean_outliers_kmeans(
                        person_matrix, 
                        n_clusters=KMEANS_CLUSTERS, 
                        percentile_threshold=KMEANS_THRESHOLD
                    )
                # ==========================================
                
                unique_activities = np.unique(person_matrix[:, IDX_LABEL])
                p_id = p_idx

                for act_id in unique_activities:
                    act_id = int(act_id)
                    act_mask = person_matrix[:, IDX_LABEL] == act_id
                    act_data = person_matrix[act_mask]
                    
                    
                    if len(act_data) < window_size: continue
                    X_list = []
                    for start in range(0, len(act_data) - window_size + 1, step_size):
                        window = act_data[start : start + window_size, :]
                        try:
                            features = extract_features_550(window, fs=FS)
                            X_list.append(features)
                        except: continue
                    
                    if len(X_list) > 0:
                        folder_path = os.path.join(save_root, f"Person_{p_id:02d}", f"Activity_{act_id:02d}")
                        os.makedirs(folder_path, exist_ok=True)
                        pd.DataFrame(X_list, columns=feat_names).to_csv(
                            os.path.join(folder_path, "features.csv"), 
                            index=False, sep=';', float_format='%.6f'
                        )
                        # print(f"   [Salvo] P{p_id:02d} Act {act_id} ({len(X_list)} wins)")

        # -------------------------------------------------------------
        # 2. DATASET SPLIT E BALANCEAMENTO
        # -------------------------------------------------------------              
        if split_dataset:
            if 'data_array' not in locals(): data_array = read_data_per_person(cfg.ASSETS_FOLDERS_PATH)        
            
            create_dataset_final(
                raw_data_list=data_array, 
                features_root=cfg.FEATURES_SOURCE_PATH, 
                save_dir=cfg.DATASET_OUTPUT_PATH,
                sampling_method=SAMPLING_METHOD
            )
                
        # -------------------------------------------------------------
        # 3. TREINO DE MODELOS
        # -------------------------------------------------------------
        if run_models:
            DATASET_DIR = cfg.DATASET_OUTPUT_PATH
            print(f"\n=== Iniciando Treino (Dados: {DATASET_DIR}) ===")
            
            try:
                X_train = np.loadtxt(f"{DATASET_DIR}/X_train.csv", delimiter=";")
                y_train = np.loadtxt(f"{DATASET_DIR}/y_train.csv", delimiter=";")
                X_val = np.loadtxt(f"{DATASET_DIR}/X_val.csv", delimiter=";")
                y_val = np.loadtxt(f"{DATASET_DIR}/y_val.csv", delimiter=";")
                X_test = np.loadtxt(f"{DATASET_DIR}/X_test.csv", delimiter=";")
                y_test = np.loadtxt(f"{DATASET_DIR}/y_test.csv", delimiter=";")
                
                # --- Guardar Informações no TXT ---
                info_file = os.path.join(results_dir, "experiment_info.txt")
                with open(info_file, "w", encoding='utf-8') as f:
                    f.write(f"=== EXPERIÊNCIA {os.path.basename(results_dir)} ===\n")
                    f.write(f"Data: {datetime.datetime.now()}\n\n")
                    f.write(f"Configurações:\n")
                    f.write(f"- Cleaning Method: {CLEANING_METHOD}\n")
                    if CLEANING_METHOD == 'ZSCORE':
                        f.write(f"  > Z-Score Threshold: {Z_SCORE_VAL}\n")
                    elif CLEANING_METHOD == 'KMEANS':
                        f.write(f"  > KMeans Clusters: {KMEANS_CLUSTERS}\n")
                        f.write(f"  > Threshold Percentile: {KMEANS_THRESHOLD}\n")
                    
                    f.write(f"- Sampling Method: {SAMPLING_METHOD}\n")
                    f.write(f"- Window Size: {cfg.WINDOW_SIZE}ms\n")
                    f.write(f"- Overlap: {cfg.OVERLAP}\n\n")
                    f.write(f"Dados:\n")
                    f.write(f"- Treino: {X_train.shape[0]} amostras\n")
                    f.write(f"- Validação: {X_val.shape[0]} amostras\n")
                    f.write(f"- Teste: {X_test.shape[0]} amostras\n")
                
                print(f"[Info] Relatório inicial salvo em: {info_file}")

                X_train_full = np.vstack([X_train, X_val])
                y_train_full = np.concatenate([y_train, y_val])

                #baseline_classifier(X_train_full, y_train_full, X_test, y_test, save_dir=f"{results_dir}/baseline")
                #knn_analysis(X_train, y_train, X_val, y_val, X_test, y_test, save_dir=f"{results_dir}/knn")
                
                # --- CONFIGURAÇÃO DE SELEÇÃO ---
                LOAD_VERSION = 5
                TOP_N = 90

                best_features = None

                # 1. Tentar carregar de versão anterior
                if LOAD_VERSION is not None:
                    best_features = load_best_features_from_version(LOAD_VERSION, top_n=TOP_N)

                # 2. Se não carregou (ou se LOAD_VERSION for None), calcula do zero
                if best_features is None:
                    print("\n>>> A calcular ReliefF do zero...")
                    best_features = relieff_tvt(
                        X_train, y_train, X_val, y_val, X_test, y_test, 
                        save_dir=f"{results_dir}/relieff"
                    )

                if best_features is not None:
                    mlp_experiment(
                        X_train, y_train, X_val, y_val, X_test, y_test, 
                        selected_features=best_features, 
                        save_dir=f"{results_dir}/mlp_experiments"
                    )
            
                    # --- PONTO 5: Rede Neuronal ---
                    run_custom_mlp(
                        X_train, y_train, 
                        X_val, y_val,
                        X_test, y_test, 
                        selected_features=best_features, 
                        save_dir=f"{results_dir}/custom_mlp"
                    )
                    
                    
            except Exception as e:
                print(f"Erro: {e}")
                with open(os.path.join(results_dir, "error_log.txt"), "w") as f:
                    f.write(str(e))
        
if __name__ == '__main__':
    main()
