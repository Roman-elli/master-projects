import pandas as pd
import numpy as np
import optuna
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import src.config as cfg

# Desativar logs verbosos do Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def preprocess_features(df):
    """Prepara as features para modelação (Encoding e Scaling)."""
    drop_cols = ['PatientId', 'ScheduledDay', 'AppointmentDay', 'No-show']
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = df['No-show'].astype(int)
    
    # One-Hot Encoding para variáveis categóricas
    X = pd.get_dummies(X, drop_first=True)
    
    # Scaling (fundamental para KNN e Regressão Logística)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X_scaled, y

def get_optuna_model(trial, model_name):
    """Define o espaço de pesquisa de hiperparâmetros para cada modelo."""
    if model_name == 'LogisticRegression':
        C = trial.suggest_float('C', 1e-4, 10.0, log=True)
        return LogisticRegression(C=C, max_iter=1000, random_state=42)
    
    elif model_name == 'KNN':
        n_neighbors = trial.suggest_int('n_neighbors', 3, 30)
        return KNeighborsClassifier(n_neighbors=n_neighbors)
    
    elif model_name == 'RandomForest':
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 3, 15)
        return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    
    elif model_name == 'XGBoost':
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
        return XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, use_label_encoder=False, eval_metric='logloss', random_state=42)
    
    elif model_name == 'LightGBM':
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
        return LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42, verbose=-1)

def tune_model(X_train, y_train, model_name, n_trials=20):
    """Executa o Optuna para encontrar os melhores hiperparâmetros num subset inicial."""
    def objective(trial):
        model = get_optuna_model(trial, model_name)
        # Split estático para manter as métricas de validação comparáveis no Optuna
        X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        model.fit(X_t, y_t)
        return roc_auc_score(y_v, model.predict_proba(X_v)[:, 1])

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    return get_optuna_model(study.best_trial, model_name)

def evaluate_robustness(X, y, best_model, n_seeds=100):
    """Treina o modelo nos melhores hiperparâmetros, variando a partição de dados 100 vezes."""
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []}
    last_cm = None
    
    for seed in range(n_seeds):
        # A seed altera a divisão dos dados a cada iteração (100 partições diferentes)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg.TEST_SIZE, random_state=seed)
        
        # A seed também garante aleatoriedade interna (ex: amostragem em Random Forest)
        if hasattr(best_model, 'random_state'):
            best_model.set_params(random_state=seed)
            
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        
        metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['precision'].append(precision_score(y_test, y_pred, zero_division=0))
        metrics['recall'].append(recall_score(y_test, y_pred))
        metrics['f1'].append(f1_score(y_test, y_pred))
        metrics['roc_auc'].append(roc_auc_score(y_test, y_proba))
        
        # Guarda a matriz de confusão da última iteração/partição do ciclo
        if seed == n_seeds - 1:
            last_cm = confusion_matrix(y_test, y_pred)
            
    return metrics, best_model, last_cm

def save_confusion_matrix(cm, model_name, output_dir):
    """Gera e guarda um gráfico da matriz de confusão."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Compareceu', 'Faltou'], 
                yticklabels=['Compareceu', 'Faltou'])
    plt.ylabel('Verdadeiro')
    plt.xlabel('Previsto')
    plt.title(f'Matriz de Confusão - {model_name}')
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_confusion_matrix.png", dpi=300)
    plt.close()