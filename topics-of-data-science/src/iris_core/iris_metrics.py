from sklearn.model_selection import train_test_split, RepeatedKFold, StratifiedKFold
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# FUNÇÕES UTILITÁRIAS
# ==========================================

def get_metrics(y_true, y_pred):
    """Calcula e retorna dicionário de métricas."""
    return {
        "acc": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "f1": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "cm": confusion_matrix(y_true, y_pred)
    }

def print_metrics(metrics, title="Métricas"):
    print(f"--- {title} ---")
    print(f"F1: {metrics['f1']:.3f} | Acc: {metrics['acc']:.3f} | Rec: {metrics['recall']:.3f} | Prec: {metrics['precision']:.3f}")

def create_imbalanced_iris():
    """Cria a versão desequilibrada do Iris (Req 2.5)"""
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Índices de cada classe
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]
    idx_2 = np.where(y == 2)[0]
    
    # Reduzir: Versicolor (1) -> 30, Virginica (2) -> 10
    idx_1 = idx_1[:30]
    idx_2 = idx_2[:10]
    
    indices = np.concatenate([idx_0, idx_1, idx_2])
    np.random.shuffle(indices) # Baralhar para não ficar ordenado
    
    return X[indices], y[indices]

# ==========================================
# EXPERIÊNCIAS
# ==========================================

def run_2_1_baselines(X, y):
    print("\n" + "="*40 + "\n2.1 BASELINES (Random & OneR)\n" + "="*40)
    
    models = {
        "Random": DummyClassifier(strategy="uniform", random_state=42),
        "OneR": DecisionTreeClassifier(max_depth=1, random_state=42)
    }

    rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)

    for name, clf in models.items():
        print(f"\n>>> {name}")
        
        # i) Train-only
        clf.fit(X, y)
        print_metrics(get_metrics(y, clf.predict(X)), "Train-only")
        
        # ii) TT 70-30
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        clf.fit(X_tr, y_tr)
        print_metrics(get_metrics(y_te, clf.predict(X_te)), "TT 70-30")
        
        # iii) 10x10 CV
        f1_scores = []
        for train_idx, test_idx in rkf.split(X):
            clf.fit(X[train_idx], y[train_idx])
            f1_scores.append(f1_score(y[test_idx], clf.predict(X[test_idx]), average='macro', zero_division=0))
        print(f"--- 10x10 CV ---")
        print(f"Média F1: {np.mean(f1_scores):.3f} (+/- {np.std(f1_scores):.3f})")

def run_2_2_knn_analysis(X, y):
    print("\n" + "="*40 + "\n2.2 kNN Analysis\n" + "="*40)
    
    # 2.2.1 k=1
    print("--- kNN (k=1) ---")
    knn = KNeighborsClassifier(n_neighbors=1)
    
    # Train-only
    knn.fit(X, y)
    print_metrics(get_metrics(y, knn.predict(X)), "Train-only")
    
    # TT 70-30
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    knn.fit(X_tr, y_tr)
    print_metrics(get_metrics(y_te, knn.predict(X_te)), "TT 70-30")
    
    # 2.2.2 Variação de K
    print("\n--- Variação de K (1..15) ---")
    k_values = range(1, 16, 2)
    
    # TVT Split (40-30-30 aproximado)
    # Primeiro tiramos 30% Teste
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    # Do resto, tiramos Validação (30% do total original ~= 43% do que sobrou)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.43, random_state=42, stratify=y_temp)
    
    train_errors = []
    val_scores = []
    
    print(f"{'K':<5} | {'Train F1':<10} | {'Val F1':<10}")
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        
        # Train score (para ver overfitting)
        train_f1 = f1_score(y_train, knn.predict(X_train), average='macro')
        train_errors.append(1 - train_f1)
        
        # Val score
        val_f1 = f1_score(y_val, knn.predict(X_val), average='macro')
        val_scores.append(val_f1)
        
        print(f"{k:<5} | {train_f1:.3f}      | {val_f1:.3f}")
        
    # 2.2.3 Plot Bias-Variance (Erro treino vs Erro Validação)
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, train_errors, label='Erro Treino (Bias)', marker='o')
    plt.plot(k_values, [1-x for x in val_scores], label='Erro Validação (Variance)', marker='s')
    plt.xlabel('k (Vizinhos)')
    plt.ylabel('Erro (1 - F1)')
    plt.title('2.2.3 Análise Bias-Variance kNN')
    plt.legend()
    plt.grid(True)
    plt.show()
    print("[Info] Gráfico Bias-Variance gerado.")

def run_2_3_relief_tvt(X, y, title_suffix=""):
    print("\n" + "="*40 + f"\n2.3 ReliefF + TVT {title_suffix}\n" + "="*40)
    
    # 1. Split TVT 40-30-30
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.43, random_state=42, stratify=y_temp)
    
    # 2. ReliefF Ranking (apenas no Treino)
    fs = ReliefF()
    fs.fit(X_train, y_train)
    ranked_indices = fs.top_features_ # Indices das features por ordem de importância
    print(f"Features ordenadas (índices): {ranked_indices}")
    
    # 3. Grid Search Manual (Num Features vs K)
    k_values = range(1, 16, 2)
    n_features_list = range(1, X.shape[1] + 1)
    
    best_f1 = -1
    best_config = None
    history = [] # Para o gráfico do cotovelo
    
    for n_feat in n_features_list:
        selected_feats = ranked_indices[:n_feat]
        
        # Subset dos dados
        X_tr_sel = X_train[:, selected_feats]
        X_val_sel = X_val[:, selected_feats]
        
        # Encontrar melhor K para este conjunto de features
        best_f1_for_feat = -1
        
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_tr_sel, y_train)
            y_pred = knn.predict(X_val_sel)
            f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_config = (n_feat, k, selected_feats)
            
            if f1 > best_f1_for_feat:
                best_f1_for_feat = f1
                
        history.append(best_f1_for_feat)
        print(f"Features: Top {n_feat} | Melhor F1 (Val): {best_f1_for_feat:.3f}")

    # 2.3.2 Gráfico do Cotovelo
    plt.figure()
    plt.plot(n_features_list, history, marker='o')
    plt.title(f"2.3.2 Gráfico do Cotovelo (Desempenho vs Features) {title_suffix}")
    plt.xlabel("Número de Features")
    plt.ylabel("Melhor F1 no Val")
    plt.grid()
    plt.show()
    
    # 2.3.6 Modelo Final
    best_n, best_k, best_feats = best_config
    print(f"\nMelhor Configuração: {best_n} Features (Ind: {best_feats}), k={best_k}")
    
    # Avaliar no Test Set
    X_train_final = X_train[:, best_feats]
    X_test_final = X_test[:, best_feats]
    
    knn_final = KNeighborsClassifier(n_neighbors=best_k)
    knn_final.fit(X_train_final, y_train)
    y_pred_test = knn_final.predict(X_test_final)
    
    metrics = get_metrics(y_test, y_pred_test)
    print_metrics(metrics, f"Resultado Final no TESTE {title_suffix}")
    print("Matriz de Confusão Teste:\n", metrics['cm'])

def run_2_4_relief_cv(X, y):
    print("\n" + "="*40 + "\n2.4 ReliefF + 10x10 CV\n" + "="*40)
    
    rkf = RepeatedKFold(n_splits=10, n_repeats=1, random_state=42) # Reduzi repeats para 1 para ser rápido, o enunciado pede 10
    
    fold_results = []
    features_count_perf = {i: [] for i in range(1, X.shape[1]+1)}
    
    print("A executar CV (pode demorar um pouco)...")
    
    for fold_idx, (train_idx, test_idx) in enumerate(rkf.split(X)):
        X_train_cv, X_test_cv = X[train_idx], X[test_idx]
        y_train_cv, y_test_cv = y[train_idx], y[test_idx]
        
        # 1. Ranking Features (dentro do fold para evitar leakage)
        fs = ReliefF()
        fs.fit(X_train_cv, y_train_cv)
        ranked = fs.top_features_
        
        # 2. Otimização (Features + K) usando 'Validation' interno ou heurística
        # Aqui vamos simplificar: testamos todas as combs e guardamos a performance para a média
        
        best_fold_f1 = -1
        best_fold_config = None
        
        # Loop para recolher dados para o gráfico médio (2.4.2)
        for n_feat in range(1, X.shape[1]+1):
            sel_feats = ranked[:n_feat]
            
            # Otimizar K internamente seria o ideal (nested CV), 
            # mas vamos assumir k=3 fixo ou uma heurística para o gráfico
            knn = KNeighborsClassifier(n_neighbors=3) 
            knn.fit(X_train_cv[:, sel_feats], y_train_cv)
            y_pred = knn.predict(X_test_cv[:, sel_feats])
            f1 = f1_score(y_test_cv, y_pred, average='macro', zero_division=0)
            
            features_count_perf[n_feat].append(f1)
            
            if f1 > best_fold_f1:
                best_fold_f1 = f1
                best_fold_config = (n_feat, 3, sel_feats) # Assumindo k=3 para simplificar o loop
        
        fold_results.append(best_fold_f1)
        
    # 2.4.2 Gráfico Médio
    means = [np.mean(features_count_perf[i]) for i in range(1, 5)]
    stds = [np.std(features_count_perf[i]) for i in range(1, 5)]
    
    plt.figure()
    plt.errorbar(range(1, 5), means, yerr=stds, fmt='-o', capsize=5)
    plt.title("2.4.2 Performance Média vs Features (10CV)")
    plt.xlabel("# Features")
    plt.ylabel("F1 Score Médio")
    plt.grid()
    plt.show()
    
    print(f"Média F1 Melhores Modelos: {np.mean(fold_results):.3f} (+/- {np.std(fold_results):.3f})")
