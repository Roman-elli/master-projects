import os
import numpy as np
import pandas as pd

def readFiles(foldername):
    data = []

    for folder in os.listdir(foldername):
        folder_path = os.path.join(foldername, folder)
        
        data_individual = []

        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            array = np.loadtxt(file_path, delimiter=',')
            data_individual.append(array)

        if data_individual:
            data.append(data_individual)

    if not data:
        print("File not found")
    return data

def read_data_per_person(foldername):
    data_per_person = []

    # --- AQUI ESTÁ O SEGREDO ---
    # sorted() força a ordem alfabética: Person_01, Person_02, ...
    folders = sorted(os.listdir(foldername))

    for idx, person_folder in enumerate(folders, start=1):
        person_path = os.path.join(foldername, person_folder)
        if not os.path.isdir(person_path):
            continue

        person_data_list = []
        
        # Também ordenamos os ficheiros dentro da pasta por segurança
        files = sorted(os.listdir(person_path))

        for file in files:
            file_path = os.path.join(person_path, file)
            try:
                array = np.loadtxt(file_path, delimiter=',')
                person_data_list.append(array)
            except:
                continue
        
        if person_data_list:
            person_data = np.vstack(person_data_list)
            # Adiciona ID da pessoa (coluna extra)
            id_column = np.full((person_data.shape[0], 1), idx)
            person_data = np.hstack((person_data, id_column))
            
            data_per_person.append(person_data)

    return data_per_person

def read_multisensor_data(foldername):
    """
    Lê ficheiros de múltiplos sensores, corta o excesso de linhas para
    ficarem iguais, e junta horizontalmente.
    """
    data_per_person = []
    
    # Ordenar pastas (Person_01, Person_02...)
    sorted_folders = sorted(os.listdir(foldername))

    for idx, person_folder in enumerate(sorted_folders, start=1):
        person_path = os.path.join(foldername, person_folder)
        if not os.path.isdir(person_path): continue

        # Ordenar ficheiros (Sensor1, Sensor2...)
        sorted_files = sorted(os.listdir(person_path))
        sensor_arrays = []

        # 1. Carregar todos os CSVs para memória
        for file in sorted_files:
            try:
                path = os.path.join(person_path, file)
                arr = np.loadtxt(path, delimiter=',')
                sensor_arrays.append(arr)
            except: continue
        
        if not sensor_arrays: continue

        try:
            # === PASSO CRÍTICO: IGUALAR O NÚMERO DE LINHAS ===
            
            # 1. Descobrir qual é o menor número de linhas entre os 5 sensores
            min_rows = min(arr.shape[0] for arr in sensor_arrays)
            
            # 2. Cortar todos os arrays para terem esse tamanho exato
            #    Isso resolve o erro "dimensions must match exactly"
            sensor_arrays = [arr[:min_rows, :] for arr in sensor_arrays]
            
            # --- Agora o HSTACK vai funcionar ---
            
            # Colunas de dados (Acc + Gyro): 1 a 6
            cols_data = [1, 2, 3, 4, 5, 6] 
            
            # Extrair dados e juntar lado a lado
            sensors_data = [s[:, cols_data] for s in sensor_arrays]
            merged_data = np.hstack(sensors_data) # (min_rows, 30)
            
            # Pegar Timestamp e Label (do primeiro sensor)
            # Como cortámos todos pelo min_rows, usamos min_rows aqui também
            ts_col = sensor_arrays[0][:min_rows, 10].reshape(-1, 1)
            label_col = sensor_arrays[0][:min_rows, 11].reshape(-1, 1)
            
            # ID da Pessoa
            id_col = np.full((min_rows, 1), idx)
            
            # Matriz Final: [ ID | TS | DADOS(30) | LABEL ]
            final_matrix = np.hstack((id_col, ts_col, merged_data, label_col))
            
            data_per_person.append(final_matrix)
            print(f"-> Pessoa {idx:02d}: Sucesso. Ajustado para {min_rows} linhas.")

        except Exception as e:
            print(f"ERRO CRÍTICO Pessoa {idx}: {e}")

    return data_per_person


def get_next_version_dir(base_path="data/results"):
    os.makedirs(base_path, exist_ok=True)
    existing = [d for d in os.listdir(base_path) if d.startswith("results_v")]
    if not existing: return os.path.join(base_path, "results_v1")
    
    versions = []
    for d in existing:
        try: v = int(d.split("v")[-1])
        except: continue
        versions.append(v)
    
    next_v = max(versions) + 1 if versions else 1
    return os.path.join(base_path, f"results_v{next_v}")

def load_best_features_from_version(version_int, top_n=50, results_root="data/results"):
    """
    Carrega o ranking do ReliefF de uma versão anterior e retorna os índices das Top N features.
    
    Args:
        version_int (int): O número da versão (ex: 3 para ler de 'results_v3').
        top_n (int): Quantas features quer selecionar.
        results_root (str): Caminho base dos resultados.
    
    Returns:
        numpy.array: Array com os índices das melhores features, ou None se falhar.
    """
    # Constrói o caminho: data/results/results_vX/relieff/relieff_ranking.csv
    version_dir = f"results_v{version_int}"
    file_path = os.path.join(results_root, version_dir, "relieff", "relieff_ranking.csv")
    
    if not os.path.exists(file_path):
        print(f"\n[AVISO] Ficheiro de ranking não encontrado em: {file_path}")
        print(" -> Verifique se o número da versão está correto ou se o ReliefF já foi executado nessa versão.")
        return None
        
    try:
        print(f"\n>>> [IO] A carregar Top {top_n} features da v{version_int}...")
        print(f"    Origem: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Validação simples
        if "Feature_Index" not in df.columns:
            print("[ERRO] O CSV não tem a coluna 'Feature_Index'.")
            return None
            
        # Seleciona as primeiras N linhas
        best_indices = df["Feature_Index"].values[:top_n]
        
        print(f"    [Sucesso] {len(best_indices)} features carregadas.")
        return best_indices

    except Exception as e:
        print(f"[ERRO] Falha ao ler o ficheiro de features: {e}")
        return None