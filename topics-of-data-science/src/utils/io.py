import os
import numpy as np

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