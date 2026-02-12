import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import joblib

# =============================================================================
# 1. FUNÇÕES DE DIVISÃO E ESTATÍSTICA (LEGACY / PART A)
# =============================================================================

def calcProportion(data, n_activities=7, save_path=None):
    n_persons = len(data)
    proportion = np.zeros((n_persons, n_activities))

    for i, person_data in enumerate(data):
        if person_data is None or len(person_data) == 0: continue
        
        # Assumindo que a label está na última coluna (-1)
        if person_data.ndim > 1 and person_data.shape[1] > 0:
            labels = person_data[:, -1].astype(int)
            
            # Filtro: Manter apenas labels <= n_activities
            valid_mask = labels <= n_activities
            valid_labels = labels[valid_mask]
            
            total_samples = len(valid_labels)
            if total_samples == 0: continue

            for act_id in range(1, n_activities + 1):
                count = np.sum(valid_labels == act_id)
                proportion[i, act_id-1] = count / total_samples
    
    return proportion

def splitPerson(proportion, train_ratio=0.4, val_ratio=0.3, test_ratio=0.3):
    n_persons = proportion.shape[0]
    train_idx, val_idx, test_idx = [], [], []

    accum_train = np.zeros(proportion.shape[1])
    accum_val = np.zeros(proportion.shape[1])
    accum_test = np.zeros(proportion.shape[1])
    
    # Ordenar por quem tem mais dados/atividades para distribuir melhor
    order = np.argsort(-np.max(proportion, axis=1))
    
    for i in order:
        diff_train = np.sum(np.abs(accum_train + proportion[i] - accum_train))
        diff_val = np.sum(np.abs(accum_val + proportion[i] - accum_val))
        diff_test = np.sum(np.abs(accum_test + proportion[i] - accum_test))
        
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
    """
    Gera índices para K-Fold Cross Validation.
    Mantido conforme solicitado.
    """
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    splits = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
        splits.append({
            "fold": fold+1,
            "train_idx": train_idx,
            "test_idx": test_idx
        })
        # print(f"Fold {fold+1}: Train={train_idx}, Test={test_idx}")

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
            
            # Proteção contra lista vazia
            arrays_to_stack = [data[i] for i in indices if data[i] is not None]
            if not arrays_to_stack: continue

            split_data = np.vstack(arrays_to_stack)
            file_name = split_name.split('/')[-1] + ".csv"
            file_path = os.path.join(split_path, file_name)

            np.savetxt(file_path, split_data, delimiter=",", fmt="%.6f")
            print(f"[✔] {split_name} salvo em {file_path}, shape: {split_data.shape}")

    if save_kfold is True:
        fold_idx = kfold_save_item - 1
        fold_data = kfold_splits[fold_idx]

        train_indices = fold_data["train_idx"]
        test_indices = fold_data["test_idx"]

        split_path = os.path.join(folderDestiny, f"kfold/fold_{kfold_save_item}")
        os.makedirs(split_path, exist_ok=True)

        tr_list = [data[i] for i in train_indices if data[i] is not None]
        ts_list = [data[i] for i in test_indices if data[i] is not None]

        if tr_list:
            train_data = np.vstack(tr_list)
            np.savetxt(os.path.join(split_path, "train.csv"), train_data, delimiter=",", fmt="%.6f")
        
        if ts_list:
            test_data = np.vstack(ts_list)
            np.savetxt(os.path.join(split_path, "test.csv"), test_data, delimiter=",", fmt="%.6f")

        print(f"[✔] Fold {kfold_save_item} salvo em {split_path}")
    
    return train_idx, val_idx, test_idx, kfold_split

# =============================================================================
# 2. FUNÇÕES DE CARREGAMENTO E SAMPLING (PART B - MACHINE LEARNING)
# =============================================================================

def load_features_from_indices(person_indices, features_root_path, valid_activities=range(1, 8)):
    """
    Carrega as features (X), labels (y) e IDs das pessoas (groups) dos índices fornecidos.
    """
    X_list = []
    y_list = []
    groups_list = [] # Importante para Oversampling por pessoa

    print(f"   [Filtro] Carregando atividades: {list(valid_activities)}")

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
                            label = int(folder_name.split("_")[1])
                            
                            # Filtro de atividades
                            if label not in valid_activities: continue 
                            
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
                                # Guardar ID da pessoa para cada amostra
                                groups_list.append(np.full(data.shape[0], p_id)) 
                                
                        except: continue

    if not X_list: return None, None, None

    return np.vstack(X_list), np.concatenate(y_list), np.concatenate(groups_list)

def apply_undersampling(X, y):
    """ Balanceia cortando pela classe minoritária. """
    print("\n   [Undersampling] Analisando distribuição...")
    classes, counts = np.unique(y, return_counts=True)
    min_samples = np.min(counts)
    
    balanced_indices = []
    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        selected = np.random.choice(cls_indices, min_samples, replace=False)
        balanced_indices.extend(selected)
        
    balanced_indices = np.array(balanced_indices)
    np.random.shuffle(balanced_indices)
    
    X_bal, y_bal = X[balanced_indices], y[balanced_indices]
    print(f"      [Concluído] Novo tamanho: {len(y_bal)} ({len(classes)} x {min_samples}).")
    return X_bal, y_bal

def apply_oversampling_per_person(X, y, groups):
    """ 
    Balanceia as classes INDIVIDUALMENTE para cada pessoa (Clonagem Intra-Sujeito).
    """
    print("\n   [Per-Person Oversampling] Balanceando cada pessoa individualmente...")
    
    unique_persons = np.unique(groups)
    X_final_list = []
    y_final_list = []
    
    for p_id in unique_persons:
        # 1. Isolar dados desta pessoa
        mask = (groups == p_id)
        X_p = X[mask]
        y_p = y[mask]
        
        # 2. Verificar classes existentes nesta pessoa
        classes, counts = np.unique(y_p, return_counts=True)
        if len(counts) == 0: continue
        
        max_samples = np.max(counts)
        
        # 3. Oversampling local
        for cls in classes:
            cls_indices = np.where(y_p == cls)[0]
            n_current = len(cls_indices)
            
            X_cls = X_p[cls_indices]
            y_cls = y_p[cls_indices]
            
            if n_current < max_samples:
                n_needed = max_samples - n_current
                # Clonar dados aleatórios desta pessoa para preencher
                extra_indices = np.random.choice(len(cls_indices), size=n_needed, replace=True)
                X_cls = np.vstack([X_cls, X_cls[extra_indices]])
                y_cls = np.concatenate([y_cls, y_cls[extra_indices]])
            
            X_final_list.append(X_cls)
            y_final_list.append(y_cls)
            
    # Juntar todas as pessoas de volta num dataset gigante
    X_final = np.vstack(X_final_list)
    y_final = np.concatenate(y_final_list)
    
    # Embaralhar tudo
    indices = np.arange(len(y_final))
    np.random.shuffle(indices)
    
    print(f"      Tamanho Final (Oversampled): {X_final.shape}")
    return X_final[indices], y_final[indices]

def create_dataset_final(raw_data_list, features_root, save_dir, sampling_method='UNDER'):
    """
    Pipeline principal de criação do dataset para treino.
    sampling_method: 'UNDER', 'OVER_PERSON', 'NONE'
    """
    print(f"\n=== Pipeline Final: Split -> Sampling ({sampling_method}) -> Normalização ===")
    
    TARGET_ACTIVITIES = range(1, 8) 

    # 1. Definir Split (TVT)
    proportion = calcProportion(raw_data_list, n_activities=7)
    train_idx, val_idx, test_idx = splitPerson(proportion)
    
    print(f"Pessoas Split -> Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # 2. Carregar Features
    print("1. Carregando dados...")
    # X_train recebe groups para poder usar o oversampling por pessoa
    X_train, y_train, groups_train = load_features_from_indices(train_idx, features_root, valid_activities=TARGET_ACTIVITIES)
    X_val, y_val, _ = load_features_from_indices(val_idx, features_root, valid_activities=TARGET_ACTIVITIES)
    X_test, y_test, _ = load_features_from_indices(test_idx, features_root, valid_activities=TARGET_ACTIVITIES)

    if X_train is None: 
        print("Erro: Nenhum dado encontrado.")
        return

    # 3. Aplicar Sampling (Apenas no Treino!)
    print(f"2. Aplicando Sampling: {sampling_method}...")
    
    if sampling_method == 'UNDER':
        X_train_bal, y_train_bal = apply_undersampling(X_train, y_train)
    elif sampling_method == 'OVER_PERSON':
        X_train_bal, y_train_bal = apply_oversampling_per_person(X_train, y_train, groups_train)
    else:
        print("   Nenhum sampling aplicado.")
        X_train_bal, y_train_bal = X_train, y_train

    print(f"   Treino Final: {X_train_bal.shape}")

    # 4. Normalização (Fit no Treino, Transform em Todos)
    print("4. Normalizando...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # A. Fit + Transform no Treino
    X_train_FINAL = scaler.fit_transform(X_train_bal)
    
    # B. Transform (sem fit) nos outros
    X_val_FINAL = scaler.transform(X_val) if X_val is not None else None
    X_test_FINAL = scaler.transform(X_test) if X_test is not None else None

    # 5. Salvar
    os.makedirs(save_dir, exist_ok=True)
    
    joblib.dump(scaler, f"{save_dir}/scaler_train.pkl")

    np.savetxt(f"{save_dir}/X_train.csv", X_train_FINAL, delimiter=";", fmt="%.6f")
    np.savetxt(f"{save_dir}/y_train.csv", y_train_bal, delimiter=";", fmt="%d")
    
    if X_val_FINAL is not None:
        np.savetxt(f"{save_dir}/X_val.csv", X_val_FINAL, delimiter=";", fmt="%.6f")
        np.savetxt(f"{save_dir}/y_val.csv", y_val, delimiter=";", fmt="%d")
        
    if X_test_FINAL is not None:
        np.savetxt(f"{save_dir}/X_test.csv", X_test_FINAL, delimiter=";", fmt="%.6f")
        np.savetxt(f"{save_dir}/y_test.csv", y_test, delimiter=";", fmt="%d")

    print(f"[✔] Dataset Final salvo em: {save_dir}")