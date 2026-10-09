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
    """Prepara as features para modelação (Feature Engineering, Encoding e Scaling)."""
    
    # Garantir que não alteramos o dataframe original por engano
    df = df.copy()
    
    # ---------------------------------------------------------
    # CAMINHO 1: Histórico do Paciente (Comportamento Passado)
    # ---------------------------------------------------------
    # 1. Ordenar por data da consulta é OBRIGATÓRIO para não usar o futuro para prever o passado
    df = df.sort_values(by='AppointmentDay')
    
    df['NoShow_num'] = df['No-show'].astype(int)
    
    # 2. Contar quantas consultas o paciente já teve ANTES desta
    df['Past_Appointments'] = df.groupby('PatientId').cumcount()
    
    # 3. Contar a soma de faltas que o paciente teve ANTES desta (shift(1) garante que a consulta atual não conta)
    df['Past_NoShows'] = df.groupby('PatientId')['NoShow_num'].transform(lambda x: x.shift().cumsum()).fillna(0)
    
    # 4. Criar a feature de ouro: Taxa Histórica de Faltas do paciente
    # (Se é a primeira vez que o paciente vai à clínica, a taxa é 0)
    df['Patient_NoShow_Rate'] = np.where(df['Past_Appointments'] > 0, 
                                         df['Past_NoShows'] / df['Past_Appointments'], 
                                         0)
    
    # ---------------------------------------------------------
    # CAMINHO 2: Otimizar a Variável WaitingDays
    # ---------------------------------------------------------
    # Criar categorias de tempo baseadas na psicologia humana e logística
    bins = [-1, 0, 3, 7, 14, 30, float('inf')]
    labels = ['MesmoDia', '1-3_Dias', '4-7_Dias', '8-14_Dias', '15-30_Dias', 'Mais_de_30_Dias']
    
    # pd.cut agrupa os números contínuos nas categorias que definimos
    df['WaitingDays_Group'] = pd.cut(df['WaitingDays'], bins=bins, labels=labels)

    # Adicionar logo a seguir à criação das categorias do WaitingDays
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[-1, 12, 25, 45, 65, 120],
        labels=["Crianca", "Jovem", "Adulto", "MeiaIdade", "Idoso"]
    )

    # ---------------------------------------------------------
    # CAMINHO 3: Risco Histórico do Bairro (Target Encoding Temporal)
    # ---------------------------------------------------------
    # Em vez de 81 colunas (One-Hot), criamos 1 única coluna com a taxa de faltas acumulada daquele bairro.
    # O shift().expanding().mean() calcula a média de faltas do bairro até à data da consulta, sem olhar para a própria consulta.
    df['Neighbourhood_Risk'] = df.groupby('Neighbourhood')['NoShow_num'].transform(
        lambda x: x.shift().expanding().mean()
    ).fillna(0) # fillna(0) para o primeiro paciente de sempre de um bairro
    
    
    # ---------------------------------------------------------
    # PREPARAÇÃO FINAL (Encoding e Scaling)
    # ---------------------------------------------------------
    # IMPORTANTE: Temos de adicionar o 'Neighbourhood' à lista do que é apagado, 
    # para que o get_dummies não o transforme em 81 colunas!
    drop_cols = ['PatientId', 'ScheduledDay', 'AppointmentDay', 'No-show', 'NoShow_num', 'Neighbourhood']
        
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = df['No-show'].astype(int)
    
    # O pd.get_dummies vai transformar o nosso novo 'WaitingDays_Group' 
    # (e o Gender, etc.) em várias colunas de 0s e 1s automaticamente.
    X = pd.get_dummies(X, drop_first=True)
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Converter para float32 para poupar memória e acelerar (Downcast)
    X_scaled = X_scaled.astype('float32')
    
    return X_scaled, y

def get_optuna_model(trial, model_name):
    """Define o espaço de pesquisa de hiperparâmetros para cada modelo com pesos balanceados."""
    if model_name == 'LogisticRegression':
        C = trial.suggest_float('C', 1e-4, 10.0, log=True)
        return LogisticRegression(C=C, class_weight='balanced', max_iter=1000, random_state=42)
    
    elif model_name == 'KNN':
        n_neighbors = trial.suggest_int('n_neighbors', 3, 30)
        # O KNN não tem suporte direto para class_weight na sua implementação base
        return KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
    
    elif model_name == 'RandomForest':
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 3, 15)
        return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, class_weight='balanced', random_state=42, n_jobs=-1)
    
    elif model_name == 'XGBoost':
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
        return XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, scale_pos_weight=3.95, eval_metric='logloss', random_state=42, n_jobs=-1)
    
    elif model_name == 'LightGBM':
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
        return LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1)

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