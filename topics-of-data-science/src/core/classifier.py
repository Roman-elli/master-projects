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
    
    print(f"[✔] Métricas salvas em {file_path}")
    return metrics_df

# Exemplo de uso para TVT
def train_and_evaluate_tvt(train_data, val_data, test_data, metrics_folder="split_assets/metrics"):
    # Concatenar train + val se quiser avaliar apenas no test, ou treinar no train completo
    X_train = np.vstack([train_data[:, :-2], val_data[:, :-2]])
    y_train = np.hstack([train_data[:, 11], val_data[:, 11]])
    
    X_test = test_data[:, :-2]
    y_test = test_data[:, 11]
    
    # Treinar modelo
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Predição
    y_pred = clf.predict(X_test)
    
    # Avaliação e salvar métricas
    metrics = evaluate_and_save_metrics(y_test, y_pred, metrics_folder, fold_name="tvt_test")
    return metrics
