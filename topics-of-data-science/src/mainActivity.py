import os
import numpy as np
import config as cfg
import matplotlib as plt
import pandas as pd

# --- IMPORTS UTILS ---
from utils.io import readFiles, read_multisensor_data, read_data_per_person
from core.data_split import create_dataset_final

# --- IMPORTS ANALYTICS & METRICS ---
from core.metrics import (
    activity_metric, zscore_outliers, k_mean, kmeans_outliers, dbscan_outliers, 
    inject_outliers, linear_model, linear_model_correction, 
    linear_model_centered_window, compute_modulus_all, clean_outliers_zscore
)

# --- IMPORTS FEATURES ---
from core.features import (
    build_feature_matrix_activity, statistical_significance, 
    apply_pca_to_activity, apply_feature_selection_to_activity, 
    extract_features_550, get_feature_names_550
)

# --- IMPORTS CLASSIFIERS ---
from core.classifier import baseline_classifier, knn_analysis, relieff_tvt, mlp_experiment

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

        calculate_features = False
        split_dataset = False
        run_models = True
        
        # -------------------------------------------------------------
        # 1. EXTRAÇÃO DE FEATURES POR ATIVIDADE
        # -------------------------------------------------------------
        if calculate_features:            
            # 1. Carregar os dados BRUTOS
            print("1. A ler ficheiros e a juntar sensores horizontalmente...")
            data_array = read_multisensor_data(cfg.ASSETS_FOLDERS_PATH)

            # 2. Configurações de Janelamento
            feat_names = get_feature_names_550() 
            
            WINDOW_MS = cfg.WINDOW_SIZE 
            OVERLAP = cfg.OVERLAP     
            FS = cfg.FS
            
            # Converter ms para número de linhas (amostras)
            window_size = int(FS * (WINDOW_MS / 1000))
            step_size = int(window_size * (1 - OVERLAP))
            
            save_root = "data/features_550_final"
            
            # A Label é a última coluna da matriz larga
            IDX_LABEL = -1 
            
            print(f"   -> Pessoas encontradas: {len(data_array)}")
            print(f"   -> Janela: {window_size} linhas | Passo: {step_size} linhas")

            for p_idx, person_matrix in enumerate(data_array, start=1):
                
                p_id = p_idx 
                print(f"\n-> Processando Pessoa {p_id:02d}...")
                
                if person_matrix is None or len(person_matrix) == 0: continue

                # === LIMPEZA DE DADOS ===
                # Aplicamos a correção Z-Score + Interpolação na matriz inteira da pessoa
                #person_matrix = clean_outliers_zscore(person_matrix, k=4.0)
                # ========================================

                unique_activities = np.unique(person_matrix[:, IDX_LABEL])

                for act_id in unique_activities:
                    act_id = int(act_id)
                    
                    # Filtrar apenas as linhas desta atividade
                    act_mask = person_matrix[:, IDX_LABEL] == act_id
                    act_data = person_matrix[act_mask]
                    
                    act_data = clean_outliers_zscore(act_data, k=4.0)
                    
                    # Se a gravação for menor que 1 janela, ignoramos
                    if len(act_data) < window_size: continue

                    X_list = []
                    
                    # --- JANELAMENTO DESLIZANTE ---
                    for start in range(0, len(act_data) - window_size + 1, step_size):
                        # Recorta a janela (50 linhas x 33 colunas)
                        window = act_data[start : start + window_size, :]
                        
                        try:
                            # Extrai 550 features desta janela (Dados já estão limpos!)
                            features = extract_features_550(window, fs=FS)
                            X_list.append(features)
                        except Exception:
                            continue
                    
                    # --- SALVAR FEATURES EM CSV ---
                    if len(X_list) > 0:
                        folder_path = os.path.join(save_root, f"Person_{p_id:02d}", f"Activity_{act_id:02d}")
                        os.makedirs(folder_path, exist_ok=True)
                        
                        df = pd.DataFrame(X_list, columns=feat_names)
                        save_file = os.path.join(folder_path, "features.csv")
                        
                        df.to_csv(save_file, index=False, sep=';', float_format='%.6f')
                        print(f"   [Salvo] Atividade {act_id}: {len(df)} janelas.")
          
        # ---------------------------------------------------------
        # 2. CRIAÇÃO DO DATASET FINAL (Split Train/Val/Test)
        # ---------------------------------------------------------              
        if split_dataset:
            # 1. Carregar dados brutos para cálculo de proporção
            if 'data_array' not in locals():
                print("A ler dados brutos...")
                data_array = read_data_per_person(cfg.ASSETS_FOLDERS_PATH)        
            
            # 2. Executar Pipeline
            if os.path.exists(cfg.FEATURES_SOURCE_PATH):
                create_dataset_final(
                    raw_data_list=data_array, 
                    features_root=cfg.FEATURES_SOURCE_PATH, 
                    save_dir=cfg.DATASET_OUTPUT_PATH
                )
            else:
                print("Erro: Extraia as features primeiro.")
                
        # ---------------------------------------------------------
        # 3. TREINO E AVALIAÇÃO DE MODELOS (Baseline, KNN, MLP)
        # ---------------------------------------------------------
        if run_models:
            DATASET_DIR = cfg.DATASET_OUTPUT_PATH
            print(f"\n=== Iniciando Treino de Modelos (Dados: {DATASET_DIR}) ===")
            
            try:
                # Carregar Datasets gerados anteriormente
                X_train = np.loadtxt(f"{DATASET_DIR}/X_train.csv", delimiter=";")
                y_train = np.loadtxt(f"{DATASET_DIR}/y_train.csv", delimiter=";")
                X_val = np.loadtxt(f"{DATASET_DIR}/X_val.csv", delimiter=";")
                y_val = np.loadtxt(f"{DATASET_DIR}/y_val.csv", delimiter=";")
                X_test = np.loadtxt(f"{DATASET_DIR}/X_test.csv", delimiter=";")
                y_test = np.loadtxt(f"{DATASET_DIR}/y_test.csv", delimiter=";")
                
                print(f"   Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

                # Combinar Train+Val para algoritmos que não usam validação (Baseline/KNN)
                X_train_full = np.vstack([X_train, X_val])
                y_train_full = np.concatenate([y_train, y_val])

                # --- 3.1 BASELINE (ZeroR / Random) ---
                print("\n--- [3.1] Classificador Baseline ---")
                baseline_classifier(X_train_full, y_train_full, X_test, y_test)

                # --- 3.2 KNN (K-Nearest Neighbors) ---
                print("\n--- [3.2] Análise KNN ---")
                knn_analysis(X_train, y_train, X_val, y_val, X_test, y_test)

                # --- 3.3 Seleção de Features (ReliefF - Supervisionado) ---
                print("\n--- [3.3] Feature Selection (ReliefF) ---")
                best_features = relieff_tvt(X_train, y_train, X_val, y_val, X_test, y_test)

                # --- 4. MLP (Rede Neural) ---
                if best_features is not None:
                    print("\n--- [4] MLP Experiments ---")
                    # Treino com Taxa de Aprendizagem Fixa
                    mlp_experiment(X_train, y_train, X_val, y_val, X_test, y_test, best_features, lr_mode='constant')
                    # Treino com Taxa de Aprendizagem Adaptativa
                    mlp_experiment(X_train, y_train, X_val, y_val, X_test, y_test, best_features, lr_mode='adaptive')
                else:
                    print("Aviso: ReliefF não retornou features, pulando MLP.")

            except Exception as e:
                print(f"Erro crítico ao rodar modelos: {e}")
                print(f"Dica: Verifique se os ficheiros CSV existem em {DATASET_DIR}")
        
        # extract_data = False
        # run_models = True

        # train_path = os.path.join(cfg.SPLIT_ASSETS_FOLDERS_PATH, "tvt", "train")
        # valid_path = os.path.join(cfg.SPLIT_ASSETS_FOLDERS_PATH, "tvt", "valid") 
        # test_path = os.path.join(cfg.SPLIT_ASSETS_FOLDERS_PATH, "tvt", "test")
        
        # if extract_data:
        #     data_per_person = read_data_per_person(cfg.ASSETS_FOLDERS_PATH)
        #     splitFiles(data_per_person, cfg.SPLIT_ASSETS_FOLDERS_PATH, save_tvt=True, save_kfold=False)

        # if run_models:
        #     SENSOR = 'acc' 
        #     print("   -> Treino:")
        #     X_train, y_train = extract_features_from_folder(train_path, sensor=SENSOR)
        #     print("   -> Validação:")
        #     X_val, y_val     = extract_features_from_folder(valid_path, sensor=SENSOR)
        #     print("   -> Teste:")
        #     X_test, y_test   = extract_features_from_folder(test_path, sensor=SENSOR)
            
        #     if X_train is None or X_val is None or X_test is None:
        #         print("ERRO: Falha na extração. Verifique os caminhos CSV.")
        #         return

        #     print(f"   Dataset pronto: X_train={X_train.shape}, X_test={X_test.shape}")

            # --- Exercicio 3.1 ---
            #X_train_full = np.vstack([X_train, X_val])
            #y_train_full = np.concatenate([y_train, y_val])
            #baseline_classifier(X_train_full, y_train_full, X_test, y_test) 

            # --- EXECUÇÃO 3.2 (kNN) ---
            #knn_analysis(X_train, y_train, X_val, y_val, X_test, y_test)
        
            # --- EXERCICIO 3.3 (ReliefF + Otimização) ---
            # best_features = relieff_tvt(X_train, y_train, X_val, y_val, X_test, y_test)

            # if best_features is not None:
            #     # --- EXERCICIO 4.1 (Taxa Fixa) ---
            #     mlp_experiment(X_train, y_train, X_val, y_val, X_test, y_test, best_features, lr_mode='constant')

            #     # --- EXERCICIO 4.2 (Taxa Variável) ---
            #     mlp_experiment(X_train, y_train, X_val, y_val, X_test, y_test, best_features, lr_mode='adaptive')
            # else:
            #     print("Saltando MLP (sem features selecionadas).")
            
if __name__ == '__main__':
    main()
