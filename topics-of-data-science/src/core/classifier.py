import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score

def evaluate_and_save_metrics(y_true, y_pred, save_path, fold_name="tvt"):

    cm = confusion_matrix(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Criar DataFrame
    metrics_df = pd.DataFrame({
        "Recall": [recall],
        "Precision": [precision],
        "F1-score": [f1]
    })
    
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, f"{fold_name}_metrics.csv")
    metrics_df.to_csv(file_path, index=False)
    
    cm_file = os.path.join(save_path, f"{fold_name}_confusion_matrix.csv")
    np.savetxt(cm_file, cm, delimiter=";", fmt="%d")
    
    print(f"[✔] Métricas salvas em {file_path}")
    return metrics_df, cm

def baseline_classifier(save_path="split_assets/"):
    print("--- [MODO TESTE] A validar funções de métricas com dados fictícios ---")

    # ==============================================================================
    # 1. CÓDIGO DO CLASSIFICADOR (COMENTADO PARA USO FUTURO)
    # ==============================================================================
    # train_path = os.path.join(cfg.SPLIT_ASSETS_FOLDERS_PATH, "tvt/train/", "train.csv")
    # test_path = os.path.join(cfg.SPLIT_ASSETS_FOLDERS_PATH, "tvt/test/", "test.csv")

    # # Nota: header=None assume que o CSV não tem nomes nas colunas
    # train_df = pd.read_csv(train_path, header=None) 
    # test_df = pd.read_csv(test_path, header=None)

    # target_col = 11 
    
    # X_train = train_df.drop(columns=[target_col]).values
    # y_train = train_df[target_col].values 

    # X_test = test_df.drop(columns=[target_col]).values
    # y_true = test_df[target_col].values   

    # clf = RandomForestClassifier(n_estimators=100, random_state=42)
    # clf.fit(X_train, y_train)

    # # D. Fazer Previsões (Gerar y_pred)
    # y_pred = clf.predict(X_test)
    # ==============================================================================


    # ==============================================================================
    # 2. DADOS FICTÍCIOS (PARA TESTAR REQUISITOS 1.2 SEM MODELO)
    # ==============================================================================
    # Simulação: 3 classes de atividades (1, 2, 3)
    
    # Gabarito (A Realidade)
    y_true = [1, 2, 3, 1, 2, 3, 1, 2, 3, 3, 1, 2]
    
    # Previsão (O que o "modelo" chutou - coloquei alguns erros de propósito)
    y_pred = [1, 2, 3, 1, 1, 3, 1, 2, 3, 2, 1, 2]
    # Erros introduzidos:
    # Índice 4: Era 2, previu 1
    # Índice 9: Era 3, previu 2

    print(f"Dados Fictícios -> y_true: {y_true}")
    print(f"Dados Fictícios -> y_pred: {y_pred}")

    # E. Avaliar
    save_metrics_path = os.path.join(save_path, "baseline_metrics")
    
    # Chama a tua função de métricas passando os dados falsos
    evaluate_and_save_metrics(y_true, y_pred, save_metrics_path, fold_name="teste_fake")

