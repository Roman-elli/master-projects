import os
import numpy as np
from sklearn.model_selection import KFold

def calcProportion(data, n_activities=7, save_path=None): # Alterado default para 7
    n_persons = len(data)
    proportion = np.zeros((n_persons, n_activities))

    for i, person_data in enumerate(data):
        if person_data.shape[0] == 0: continue
        
        # Filtra apenas atividades de interesse (1 a 7)
        # Assumindo que a label está na última coluna (-1) ou 11
        labels = person_data[:, -1].astype(int) 
        
        # Filtro: Manter apenas labels <= n_activities
        valid_mask = labels <= n_activities
        valid_labels = labels[valid_mask]
        
        total_samples = len(valid_labels)
        if total_samples == 0: continue # Pessoa não tem dados válidos nestas atividades

        for act_id in range(1, n_activities + 1):
            count = np.sum(valid_labels == act_id)
            proportion[i, act_id-1] = count / total_samples
    
    # ... (Resto da função de salvar CSV igual) ...
    return proportion

def splitPerson(proportion, train_ratio=0.4, val_ratio=0.3, test_ratio=0.3):
    n_persons = proportion.shape[0]

    train_idx, val_idx, test_idx = [], [], []

    accum_train = np.zeros(proportion.shape[1])
    accum_val = np.zeros(proportion.shape[1])
    accum_test = np.zeros(proportion.shape[1])
    order = np.argsort(-np.max(proportion, axis=1))
    
    for i in order:
        # Calcula diferença atual de proporções caso adicionemos esse indivíduo a cada split
        diff_train = np.sum(np.abs(accum_train + proportion[i] - accum_train))
        diff_val = np.sum(np.abs(accum_val + proportion[i] - accum_val))
        diff_test = np.sum(np.abs(accum_test + proportion[i] - accum_test))
        
        # Escolhe o split com menor diferença de distribuição atual
        target = np.argmin([diff_train, diff_val, diff_test])
        
        if target == 0 and len(train_idx) < int(train_ratio * n_persons):
            train_idx.append(i)
            accum_train += proportion[i]
        elif target == 1 and len(val_idx) < int(val_ratio * n_persons):
            val_idx.append(i)
            accum_val += proportion[i]
        else:
            test_idx.append(i)
            accum_test += proportion[i]

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)

def kfold_split(data, n_splits=5, random_state=42, shuffle=True):
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    splits = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
        splits.append({
            "fold": fold+1,
            "train_idx": train_idx,
            "test_idx": test_idx
        })
        print(f"Fold {fold+1}: Train={train_idx}, Test={test_idx}")

    return splits

def splitFiles(data, folderDestiny, save_tvt=None, save_kfold=None, kfold_save_item=1):
    proportion = calcProportion(data, save_path=folderDestiny)
    
    train_idx, val_idx, test_idx = splitPerson(proportion)
    kfold_splits = kfold_split(np.arange(len(proportion)))
    
    if save_tvt is True:
        splits = {
            "tvt/train": train_idx,
            "tvt/valid": val_idx,
            "tvt/test": test_idx
        }

        for split_name, indices in splits.items():
            split_path = os.path.join(folderDestiny, split_name)
            os.makedirs(split_path, exist_ok=True)

            split_data = np.vstack([data[i] for i in indices])

            file_name = split_name.split('/')[-1] + ".csv"
            file_path = os.path.join(split_path, file_name)

            np.savetxt(file_path, split_data, delimiter=",", fmt="%.6f")
            print(f"[✔] {split_name} salvo em {file_path}, shape: {split_data.shape}")

        print("[✔] Todos os splits criados e concatenados.")

    if save_kfold is True:

        fold_idx = kfold_save_item - 1
        fold_data = kfold_splits[fold_idx]

        train_indices = fold_data["train_idx"]
        test_indices = fold_data["test_idx"]

        split_path = os.path.join(folderDestiny, f"kfold/fold_{kfold_save_item}")
        os.makedirs(split_path, exist_ok=True)

        train_data = np.vstack([data[i] for i in train_indices])
        test_data = np.vstack([data[i] for i in test_indices])

        np.savetxt(os.path.join(split_path, "train.csv"), train_data, delimiter=",", fmt="%.6f")
        np.savetxt(os.path.join(split_path, "test.csv"), test_data, delimiter=",", fmt="%.6f")

        print(f"[✔] Fold {kfold_save_item} salvo em {split_path}")
    return train_idx, val_idx, test_idx, kfold_split


import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

# Mantenha as funções calcProportion e splitPerson iguais...
# ...

def load_features_from_indices(person_indices, features_root_path, valid_activities=range(1, 8)):
    """
    Carrega features apenas das atividades permitidas (ex: 1 a 7).
    """
    X_list = []
    y_list = []

    print(f"   [Filtro] Carregando apenas atividades: {list(valid_activities)}")

    for p_idx in sorted(person_indices):
        p_id = p_idx + 1
        person_folder = os.path.join(features_root_path, f"Person_{p_id:02d}")

        if not os.path.exists(person_folder): continue

        for root, dirs, files in os.walk(person_folder):
            for file in files:
                if file == "features.csv":
                    folder_name = os.path.basename(root)
                    if "Activity_" in folder_name:
                        try:
                            # Extrair ID da atividade do nome da pasta
                            label = int(folder_name.split("_")[1])
                            
                            # === O FILTRO ACONTECE AQUI ===
                            if label not in valid_activities:
                                continue 
                            
                            csv_path = os.path.join(root, file)
                            
                            try:
                                data = np.loadtxt(csv_path, delimiter=';', skiprows=1)
                            except:
                                df = pd.read_csv(csv_path, sep=';')
                                data = df.values

                            if data.ndim == 1: data = data.reshape(1, -1)
                            
                            if data.shape[0] > 0:
                                X_list.append(data)
                                y_list.append(np.full(data.shape[0], label))
                                
                        except: continue

    if not X_list: return None, None

    return np.vstack(X_list), np.concatenate(y_list)

def apply_undersampling(X, y):
    """
    Balanceia o dataset cortando as classes majoritárias e imprime as estatísticas.
    """
    print("\n   [Undersampling] Analisando distribuição de classes no Treino:")
    
    # 1. Contar quantas amostras existem por atividade
    classes, counts = np.unique(y, return_counts=True)
    min_samples = np.min(counts)
    
    # 2. Imprimir tabela de contagem ANTES do corte
    print(f"      {'Atividade':<10} | {'Amostras (Original)':<20}")
    print("      " + "-"*35)
    
    for cls, count in zip(classes, counts):
        print(f"      {int(cls):<10} | {count:<20}")
        
    print("      " + "-"*35)
    print(f"      -> Classe minoritária determina o limite: {min_samples} amostras.")
    print(f"      -> Reduzindo TODAS as atividades para {min_samples} amostras...")
    
    # 3. Executar o corte
    balanced_indices = []
    
    for cls in classes:
        # Pegar índices desta classe
        cls_indices = np.where(y == cls)[0]
        
        # Escolher aleatoriamente 'min_samples' índices (sem reposição)
        # Random_state não fixo para variar a cada execução, ou fixo se quiser repetibilidade
        selected = np.random.choice(cls_indices, min_samples, replace=False)
        balanced_indices.extend(selected)
        
    # Converter para array e embaralhar
    balanced_indices = np.array(balanced_indices)
    np.random.shuffle(balanced_indices)
    
    # 4. Verificar total final
    X_bal, y_bal = X[balanced_indices], y[balanced_indices]
    print(f"      [Concluído] Novo tamanho do Treino: {len(y_bal)} amostras ({len(classes)} classes x {min_samples}).")
    
    return X_bal, y_bal

def create_dataset_final(raw_data_list, features_root, save_dir):
    print("\n=== Pipeline Final: Split -> Undersampling -> Normalização (Fit=Train, Trans=All) ===")
    
    TARGET_ACTIVITIES = range(1, 8) 

    # 1. Definir Split (TVT)
    proportion = calcProportion(raw_data_list, n_activities=7)
    train_idx, val_idx, test_idx = splitPerson(proportion)
    
    print(f"Pessoas Split -> Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # 2. Carregar Features
    print("1. Carregando dados...")
    X_train, y_train = load_features_from_indices(train_idx, features_root, valid_activities=TARGET_ACTIVITIES)
    X_val, y_val = load_features_from_indices(val_idx, features_root, valid_activities=TARGET_ACTIVITIES)
    X_test, y_test = load_features_from_indices(test_idx, features_root, valid_activities=TARGET_ACTIVITIES)

    if X_train is None: 
        print("Erro: Nenhum dado encontrado.")
        return

    # 3. Undersampling (Só Treino)
    X_train_bal, y_train_bal = apply_undersampling(X_train, y_train)
    print(f"   Treino Original: {X_train.shape} -> Balanceado: {X_train_bal.shape}")

    # 4. NORMALIZAÇÃO CORRETA
    print("4. Normalizando (Scaler fit no Treino, aplicado em TODOS)...")
    
    # Usar StandardScaler é geralmente melhor para sensores (Acc/Gyro têm distribuições Gaussianas)
    # Mas se preferir MinMaxScaler, mantenha.
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # A. Aprender escala com o Treino Balanceado
    X_train_FINAL = scaler.fit_transform(X_train_bal)
    
    # B. Aplicar a MESMA escala aos outros (SEM REAPRENDER!)
    # Se X_val ou X_test forem None (vazios), trate isso
    if X_val is not None:
        X_val_FINAL = scaler.transform(X_val)
    else: X_val_FINAL = None

    if X_test is not None:
        X_test_FINAL = scaler.transform(X_test)
    else: X_test_FINAL = None

    # 5. Salvar
    os.makedirs(save_dir, exist_ok=True)
    
    import joblib
    joblib.dump(scaler, f"{save_dir}/scaler_train.pkl")

    np.savetxt(f"{save_dir}/X_train.csv", X_train_FINAL, delimiter=";", fmt="%.6f")
    np.savetxt(f"{save_dir}/y_train.csv", y_train_bal, delimiter=";", fmt="%d")
    
    if X_val_FINAL is not None:
        np.savetxt(f"{save_dir}/X_val.csv", X_val_FINAL, delimiter=";", fmt="%.6f")
        np.savetxt(f"{save_dir}/y_val.csv", y_val, delimiter=";", fmt="%d")
        
    if X_test_FINAL is not None:
        np.savetxt(f"{save_dir}/X_test.csv", X_test_FINAL, delimiter=";", fmt="%.6f")
        np.savetxt(f"{save_dir}/y_test.csv", y_test, delimiter=";", fmt="%d")

    print(f"[✔] Dataset Final Normalizado salvo em: {save_dir}")