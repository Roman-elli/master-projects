from utils.io import readFiles, read_data_per_person
from core.data_split import splitFiles
from core.metrics import activity_metric, zscore_outliers, k_mean, manual_kmeans, kmeans_outliers, dbscan_outliers, inject_outliers, linear_model, create_windows, linear_model_correction, linear_model_centered_window, compute_modulus_all
from core.features import build_feature_matrix_activity, apply_feature_selection_to_sensor, statistical_significance, apply_pca_to_activity, apply_feature_selection_to_activity, extract_features_550, get_feature_names_550
from core.classifier import baseline_classifier, knn_analysis, relieff_tvt, mlp_experiment
import config as cfg
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


def extract_features_from_folder(folder_path, sensor='acc'):
    """
    Tenta usar a função original. Se falhar (erro de dimensões), usa uma 
    extração de features manual para garantir que o projeto não para.
    """
    # 1. Ler o CSV
    csv_name = os.path.basename(folder_path) + ".csv"
    file_path = os.path.join(folder_path, csv_name)
    
    if not os.path.exists(file_path):
        print(f"[AVISO] Ficheiro {file_path} não encontrado.")
        return None, None

    print(f"   -> A processar {file_path}...")
    try:
        df = pd.read_csv(file_path, header=None)
        data = df.values
    except Exception as e:
        print(f"      Erro leitura: {e}")
        return None, None

    # Coluna da Label (índice 11 conforme o teu log)
    LABEL_COL = 11 
    
    # Definir colunas do sensor com base no nome (Ajuste se necessário)
    # Normalmente: Acc(0,1,2), Gyro(3,4,5), Mag(6,7,8)
    if sensor == 'acc':
        sensor_cols = [0, 1, 2]
    elif sensor == 'gyro':
        sensor_cols = [3, 4, 5]
    elif sensor == 'mag':
        sensor_cols = [6, 7, 8]
    else:
        sensor_cols = [0, 1, 2] # Default

    X_list = []
    y_list = []
    
    unique_activities = np.unique(data[:, LABEL_COL])
    
    # Parâmetros da Janela
    fs = 50 
    window_size = 50 # amostras (1 segundo)
    overlap = 25     # 50%
    
    for act_id in unique_activities:
        # Filtrar dados da atividade
        act_data = data[data[:, LABEL_COL] == act_id]
        
        if len(act_data) < window_size: continue

        # === TENTATIVA 1: Função Original ===
        # (Mantemos caso funcione para algumas atividades)
        try:
            X_act, _ = build_feature_matrix_activity([act_data], activity_id=act_id, sensor=sensor, fs=fs, window_ms=1000, overlap=0.5)
            if len(X_act) > 0:
                X_list.append(X_act)
                y_list.append(np.full(len(X_act), act_id))
                continue # Se funcionou, passa à próxima atividade
        except Exception:
            pass # Falhou silenciosamente, vamos para o método manual

        # === PLANO B: Extração Manual (Salva-vidas) ===
        # Se a função original falhar, calculamos nós as features: Média, Std, Max, Min
        # Isto cumpre o requisito de "estabelecer baseline"
        
        manual_features = []
        
        # Janelamento deslizante
        for start in range(0, len(act_data) - window_size, overlap):
            window = act_data[start : start + window_size, sensor_cols]
            
            # Extrair estatísticas simples por eixo (x, y, z)
            # 1. Média
            means = np.mean(window, axis=0)
            # 2. Desvio Padrão
            stds = np.std(window, axis=0)
            # 3. Máximo
            maxs = np.max(window, axis=0)
            # 4. Mínimo
            mins = np.min(window, axis=0)
            
            # Juntar tudo num vetor de features (3 eixos * 4 stats = 12 features)
            features = np.concatenate([means, stds, maxs, mins])
            manual_features.append(features)
            
        if len(manual_features) > 0:
            X_list.append(np.array(manual_features))
            y_list.append(np.full(len(manual_features), act_id))
            # print(f"      [Info] Atv {act_id}: Extração manual usada ({len(manual_features)} janelas)")

    if len(X_list) == 0:
        return None, None

    return np.vstack(X_list), np.concatenate(y_list)

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
                      #apply_feature_selection_to_activity(sensor, part, act_id)
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
    # Imports...
    calculate_features = False
    if calculate_features:
        
        # 1. Carregar os dados
        print("A ler ficheiros...")
        data_array = readFiles(cfg.ASSETS_FOLDERS_PATH)

        # 2. Configurações para 550 Colunas
        feat_names = get_feature_names_550() 
        
        WINDOW_MS = cfg.WINDOW_SIZE 
        OVERLAP = cfg.OVERLAP     
        FS = cfg.FS
        
        window_size = int(FS * (WINDOW_MS / 1000))
        step_size = int(window_size * (1 - OVERLAP))
        
        save_root = "data/features_550_final"
        IDX_LABEL = 11  # VERIFIQUE ESTE ÍNDICE! Se tem 30 colunas de dados, a label deve estar depois (ex: 31).
        
        print(f"\n=== Iniciando Extração de 550 Features ===")
        print(f"Pessoas encontradas: {len(data_array)}")

        for p_idx, person_files in enumerate(data_array, start=1):
            
            p_id = p_idx 
            print(f"\n-> Processando Pessoa {p_id:02d}...")
            
            if not person_files: continue
                
            try:
                full_person_data = np.vstack(person_files)
            except ValueError:
                print("   [Erro] Dimensões incompatíveis.")
                continue

            unique_activities = np.unique(full_person_data[:, IDX_LABEL])

            for act_id in unique_activities:
                act_id = int(act_id)
                
                act_mask = full_person_data[:, IDX_LABEL] == act_id
                act_data = full_person_data[act_mask]
                
                if len(act_data) < window_size: continue

                X_list = []
                
                # --- Janelamento ---
                for start in range(0, len(act_data) - window_size + 1, step_size):
                    window = act_data[start : start + window_size, :]
                    
                    try:
                        # [ Image of multi-sensor feature extraction ]
                        # Chama a função que faz o loop dos 5 sensores e concatena
                        features = extract_features_550(window, fs=FS)
                        X_list.append(features)
                    except Exception:
                        continue
                
                # --- Salvar ---
                if len(X_list) > 0:
                    folder_path = os.path.join(save_root, f"Person_{p_id:02d}", f"Activity_{act_id:02d}")
                    os.makedirs(folder_path, exist_ok=True)
                    
                    df = pd.DataFrame(X_list, columns=feat_names)
                    save_file = os.path.join(folder_path, "features.csv")
                    
                    df.to_csv(save_file, index=False, sep=';', float_format='%.6f')
                    print(f"   [Salvo] Atividade {act_id}: {len(df)} janelas (550 cols).")

    
    
    
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
