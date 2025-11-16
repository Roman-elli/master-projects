import os
import numpy as np
from sklearn.model_selection import KFold

def calcProportion(data, n_activities=16, save_path=None):
    n_persons = len(data)
    proportion = np.zeros((n_persons, n_activities))

    for i, person_data in enumerate(data):
        if person_data.shape[0] == 0:
            continue
        
        activities = person_data[:, 11].astype(int)
        total_samples = len(activities)

        for act_id in range(1, n_activities + 1):
            count = np.sum(activities == act_id)
            proportion[i, act_id-1] = count / total_samples
    
     # --- Salvar CSV ---
    if(save_path != None):
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        
        header = ";".join([f"Activity_{i}" for i in range(1, n_activities+1)])
        np.savetxt(f"{save_path}/proportion_activities.csv", proportion, delimiter=";", header=header, fmt="%.6f", comments="")

        print(f"[✔] Matriz de proporção salva em {save_path}")

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
