import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from iris_core.iris_metrics import get_metrics, print_metrics
from skrebate import ReliefF
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

def evaluate_and_save_metrics(y_true, y_pred, save_path, fold_name="tvt"):

    os.makedirs(save_path, exist_ok=True)

    # Calcular Métricas
    cm = confusion_matrix(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Criar DataFrame com escalares
    metrics_df = pd.DataFrame({
        "Recall": [recall],
        "Precision": [precision],
        "F1-score": [f1]
    })
    
    # Salvar Métricas
    file_path = os.path.join(save_path, f"{fold_name}_metrics.csv")
    metrics_df.to_csv(file_path, index=False)
    
    # Salvar Matriz de Confusão
    cm_file = os.path.join(save_path, f"{fold_name}_confusion_matrix.csv")
    np.savetxt(cm_file, cm, delimiter=";", fmt="%d")
    
    print(f"[✔] Métricas salvas em {file_path}")
    return metrics_df, cm

def baseline_classifier(X_train, y_train, X_test, y_test, save_dir="data/results/baseline"):
    """
    Executa os 3 cenários pedidos para o OneR:
    i) Train-only (Treina e testa em tudo)
    ii) Train-Test 70-30 (Já vem dividido do main)
    iii) 10x10 Cross Validation (Usa tudo)
    """
    print("\n" + "="*40 + "\n3.1 CLASSIFICADOR OneR (Baseline)\n" + "="*40)
    
    # Definir Modelo OneR (Árvore profundidade 1)
    clf = DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=42)
    
    # --- CENÁRIO ii) TT 70-30 (O mais comum) ---
    print("\n>>> Cenário ii) Train-Test (70-30)...")
    clf.fit(X_train, y_train)
    y_pred_tt = clf.predict(X_test)
    
    evaluate_and_save_metrics(y_test, y_pred_tt, save_dir, fold_name="OneR_TT_70_30")

    # --- Preparar dados completos para i) e iii) ---
    X_full = np.vstack([X_train, X_test])
    y_full = np.concatenate([y_train, y_test])

    # --- CENÁRIO i) Train-only ---
    print("\n>>> Cenário i) Train-only (Overfitting check)...")
    clf.fit(X_full, y_full)
    y_pred_full = clf.predict(X_full)
    
    evaluate_and_save_metrics(y_full, y_pred_full, save_dir, fold_name="OneR_Train_Only")

    # --- CENÁRIO iii) 10x10 Cross-Validation ---
    print("\n>>> Cenário iii) 10x10 CV...")
    rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
    
    f1_scores = []
    acc_scores = []
    
    for i, (train_idx, val_idx) in enumerate(rkf.split(X_full)):
        X_cv_tr, X_cv_val = X_full[train_idx], X_full[val_idx]
        y_cv_tr, y_cv_val = y_full[train_idx], y_full[val_idx]
        
        clf.fit(X_cv_tr, y_cv_tr)
        y_pred_cv = clf.predict(X_cv_val)
        
        f1_scores.append(f1_score(y_cv_val, y_pred_cv, average='macro', zero_division=0))
        acc_scores.append(accuracy_score(y_cv_val, y_pred_cv))
        
    print(f"Média F1 (100 folds): {np.mean(f1_scores):.4f} (+/- {np.std(f1_scores):.4f})")
    print(f"Média Acc (100 folds): {np.mean(acc_scores):.4f}")
    
def knn_analysis(X_train, y_train, X_val, y_val, X_test, y_test, save_dir="data/results/knn"):
    os.makedirs(save_dir, exist_ok=True)
    print("\n" + "="*40 + "\n3.2 kNN Analysis\n" + "="*40)
    
    print("--- kNN (k=1) ---")
    knn = KNeighborsClassifier(n_neighbors=1)
    
    # Train-only
    X_full = np.vstack([X_train, X_val, X_test])
    y_full = np.concatenate([y_train, y_val, y_test])
    
    knn.fit(X_full, y_full)
    y_pred_full = knn.predict(X_full)
    evaluate_and_save_metrics(y_full, y_pred_full, save_dir, fold_name="knn_k1_train")    
    
    # TT 70-30
    X_train_tt = np.vstack([X_train, X_val])
    y_train_tt = np.concatenate([y_train, y_val])
    
    knn.fit(X_train_tt, y_train_tt)
    y_pred_tt = knn.predict(X_test)
    evaluate_and_save_metrics(y_test, y_pred_tt, save_dir, fold_name="knn_k1_tt70x30") 
    
    # 10x10 CV
    rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
    f1_cv = []
    for tr_idx, ts_idx in rkf.split(X_full):
        knn.fit(X_full[tr_idx], y_full[tr_idx])
        pred = knn.predict(X_full[ts_idx])
        f1_cv.append(f1_score(y_full[ts_idx], pred, average='macro', zero_division=0))
    print(f"      Média F1 (CV k=1): {np.mean(f1_cv):.3f}")
    
    
    # 3.2.2 Variação de K
    print("\n--- Variação de K (1..15) ---")
    k_values = range(1, 16, 2)
    train_errors = [] # Para análise de Bias (1 - F1)
    val_errors = []   # Para análise de Variance (1 - F1)
    cv_means = []
    
    
    print(f"{'K':<3} | {'Train F1':<8} | {'Val F1':<8} | {'CV F1 (Média)':<12}")
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        # A. Train-only (Bias check) - Treinar no Train, Avaliar no Train
        knn.fit(X_train, y_train)
        pred_train = knn.predict(X_train)
        f1_train = f1_score(y_train, pred_train, average='macro', zero_division=0)
        train_errors.append(1 - f1_train)
        
        # B. TVT (Variance check) - Treinar no Train, Avaliar no Valid
        # O modelo já está treinado no X_train, só precisamos prever no X_val
        pred_val = knn.predict(X_val)
        f1_val = f1_score(y_val, pred_val, average='macro', zero_division=0)
        val_errors.append(1 - f1_val)
        
        # C. 10x10 CV (Simplificado para performance)
        cv_k_scores = []
        rkf_fast = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42) 
        for tr_idx, ts_idx in rkf_fast.split(X_full):
            knn.fit(X_full[tr_idx], y_full[tr_idx])
            cv_k_scores.append(f1_score(y_full[ts_idx], knn.predict(X_full[ts_idx]), average='macro', zero_division=0))
        
        mean_cv = np.mean(cv_k_scores)
        cv_means.append(mean_cv)
        
        print(f"{k:<3} | {f1_train:.3f}    | {f1_val:.3f}    | {mean_cv:.3f}")        
    
    # 3.2.3 - Gráfico Bias-Variance
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, train_errors, label='Erro Treino (Bias)', marker='o', linestyle='--')
    plt.plot(k_values, val_errors, label='Erro Validação (Variance)', marker='s')
    
    plt.title('3.2.3 Análise Bias-Variance (kNN)')
    plt.xlabel('Valor de k (Vizinhos) -> (Complexidade diminui ->)')
    plt.ylabel('Erro (1 - F1 Score)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Salvar gráfico
    plot_path = os.path.join(save_dir, "knn_bias_variance.png")
    plt.savefig(plot_path)
    plt.show() # Remove isto se estiveres a correr em servidor sem ecrã
    print(f"[✔] Gráfico Bias-Variance salvo em {plot_path}")
'''
def relieff_tvt(X_train, y_train, X_val, y_val, X_test, y_test, save_dir="data/results/relief"):
    os.makedirs(save_dir, exist_ok=True)
    print("\n" + "="*40 + "\n3.3 OTIMIZAÇÃO RELIEFF + kNN (TVT)\n" + "="*40)    
    
    #ReliefF Ranking (apenas no Treino)
    fs = ReliefF(n_features_to_select=X_train.shape[1], n_neighbors=20, n_jobs=-1)
    fs.fit(X_train, y_train)
    
    ranked_indices = fs.top_features_ # Indices das features por ordem de importância
    features_scores = fs.feature_importances_
    
    print(f"Features ordenadas (índices): {ranked_indices}")
    pd.DataFrame({"Feature_Index": ranked_indices, "Score": feature_scores[ranked_indices]}).to_csv(
        os.path.join(save_dir, "relieff_ranking.csv"), index=False
    )
    
    # 3. Grid Search Manual (Num Features vs K)
    k_values = range(1, 16, 2)
    n_features_list = range(1, X_train.shape[1] + 1)
    
    best_f1 = -1
    best_config = None
    elbow_scores = [] # Para o gráfico do cotovelo
    all_results = []
    
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
            
            # Guardar histórico
            all_results.append({
                "Num_Features": n_feat,
                "k": k,
                "F1_Validation": f1
            })
            
            if f1 > best_f1:
                best_f1 = f1
                best_config = (n_feat, k, selected_feats)
            
            if f1 > best_f1_for_feat:
                best_f1_for_feat = f1
                
        elbow_scores.append(best_f1_for_feat)
        print(f"Features: Top {n_feat} | Melhor F1 (Val): {best_f1_for_feat:.3f}")
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(save_dir, "grid_search_results.csv"), index=False)

    # =========================================================================
    # 3. GRÁFICO DO COTOVELO (3.3.2)
    # =========================================================================
    plt.figure(figsize=(10, 6))
    plt.plot(n_features_list, elbow_scores, marker='o', linestyle='-', color='b')
    plt.title('3.3.2 Gráfico do Cotovelo (Desempenho vs Nº Features)')
    plt.xlabel('Número de Features (Top ReliefF)')
    plt.ylabel('Melhor F1-Score (Validação)')
    plt.grid(True)
    plt.xticks(n_features_list)
    
    plot_path = os.path.join(save_dir, "elbow_plot.png")
    plt.savefig(plot_path)
    # plt.show() # Descomenta se quiseres ver a janela pop-up
    print(f"[✔] Gráfico salvo em {plot_path}")

    # =========================================================================
    # 4. AVALIAÇÃO FINAL NO TESTE (3.3.4)
    # =========================================================================
    best_n, best_k, best_feats_indices = best_config
    print(f"\n>>> CONFIGURAÇÃO IDEAL: {best_n} Features | k={best_k} (Val F1: {best_f1:.4f})")
    
    # Treinar modelo final com a melhor configuração
    # Nota: Podemos usar Train ou Train+Valid. Para estimativa rigorosa usamos Train.
    X_tr_final = X_train[:, best_feats_indices]
    X_ts_final = X_test[:, best_feats_indices]
    
    knn_final = KNeighborsClassifier(n_neighbors=best_k)
    knn_final.fit(X_tr_final, y_train)
    y_pred_test = knn_final.predict(X_ts_final)
    
    # Comparar Val vs Test
    print("\n--- 3.3.4 Resultados Finais (Modelo Ideal) ---")
    metrics, cm = evaluate_and_save_metrics(y_test, y_pred_test, save_dir, fold_name="ideal_model_test")
    
    print(f"Validação F1: {best_f1:.4f}")
    print(f"Teste F1:     {metrics['F1-score'].iloc[0]:.4f}")
    
    diff = abs(best_f1 - metrics['F1-score'].iloc[0])
    if diff > 0.10:
        print("ALERT: Grande diferença entre Val e Test. Possível Overfitting ao conjunto de Validação.")
    else:
        print("Resultado consistente entre Val e Test.")
    
    return best_feats_indices
'''

from sklearn.utils import resample # <--- IMPORTANTE: Adiciona este import

def relieff_tvt(X_train, y_train, X_val, y_val, X_test, y_test, save_dir="data/results/relieff"):
    os.makedirs(save_dir, exist_ok=True)
    print("\n" + "="*40 + "\n3.3 OTIMIZAÇÃO RELIEFF + kNN (TVT)\n" + "="*40)
    
    # =========================================================================
    # 1. RANKING DE FEATURES (ReliefF) COM DOWNSAMPLING
    # =========================================================================
    print(">>> 1. A calcular Ranking ReliefF...")
    
    # --- CORREÇÃO DE MEMÓRIA ---
    # O ReliefF crasha com muitos dados. Vamos usar no máximo 2000 amostras para o ranking.
    # Isto é suficiente para saber quais as features importantes.
    MAX_SAMPLES_RELIEF = 2000 
    
    if len(X_train) > MAX_SAMPLES_RELIEF:
        print(f"   [Aviso] Dataset muito grande ({len(X_train)}). Usando apenas {MAX_SAMPLES_RELIEF} amostras para o ReliefF para evitar crash de RAM.")
        # stratify=y_train garante que mantemos as proporções das atividades
        X_relief, y_relief = resample(X_train, y_train, n_samples=MAX_SAMPLES_RELIEF, random_state=42, stratify=y_train)
    else:
        X_relief, y_relief = X_train, y_train

    # n_neighbors=100 é mais seguro para ReliefF com ruído
    fs = ReliefF(n_features_to_select=X_train.shape[1], n_neighbors=100, n_jobs=-1)
    
    # Treinamos APENAS com o subconjunto reduzido
    fs.fit(X_relief, y_relief)
    
    ranked_indices = fs.top_features_
    feature_scores = fs.feature_importances_
    
    print(f"   Features Ordenadas (indices): {ranked_indices}")
    
    # Guardar ranking
    pd.DataFrame({"Feature_Index": ranked_indices, "Score": feature_scores[ranked_indices]}).to_csv(
        os.path.join(save_dir, "relieff_ranking.csv"), index=False
    )

    # =========================================================================
    # 2. GRID SEARCH (Features + k) - Continua a usar dados COMPLETOS aqui
    # =========================================================================
    # NOTA: O kNN é rápido, por isso aqui voltamos a usar o X_train COMPLETO,
    # não usamos o X_relief reduzido.
    
    print("\n>>> 2. A Otimizar (Grid Search: Features vs k)...")
    
    k_values = range(1, 16, 2) 
    n_features_list = range(1, X_train.shape[1] + 1)
    
    best_f1_val = -1
    best_config = None 
    elbow_scores = [] 
    all_results = []
    
    for n_feat in n_features_list:
        selected_feats = ranked_indices[:n_feat]
        
        # Aqui usamos o X_train original (grande) porque o kNN aguenta bem
        X_tr_sel = X_train[:, selected_feats]
        X_val_sel = X_val[:, selected_feats]
        
        best_f1_for_this_n_feat = -1
        
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1) # n_jobs=-1 ajuda na velocidade
            knn.fit(X_tr_sel, y_train)
            
            y_pred_val = knn.predict(X_val_sel)
            f1_val = f1_score(y_val, y_pred_val, average='macro', zero_division=0)
            
            all_results.append({
                "Num_Features": n_feat,
                "k": k,
                "F1_Validation": f1_val
            })
            
            if f1_val > best_f1_val:
                best_f1_val = f1_val
                best_config = (n_feat, k, selected_feats)
            
            if f1_val > best_f1_for_this_n_feat:
                best_f1_for_this_n_feat = f1_val
                
        elbow_scores.append(best_f1_for_this_n_feat)
        print(f"   Top {n_feat} Features -> Melhor Val F1: {best_f1_for_this_n_feat:.4f}")

    # Salvar resultados
    pd.DataFrame(all_results).to_csv(os.path.join(save_dir, "grid_search_results.csv"), index=False)

    # =========================================================================
    # 3. GRÁFICO DO COTOVELO
    # =========================================================================
    plt.figure(figsize=(10, 6))
    plt.plot(n_features_list, elbow_scores, marker='o', linestyle='-', color='b')
    plt.title('3.3.2 Gráfico do Cotovelo (ReliefF Ranking)')
    plt.xlabel('Número de Features')
    plt.ylabel('Melhor F1-Score (Validação)')
    plt.grid(True)
    plt.xticks(n_features_list)
    
    plot_path = os.path.join(save_dir, "elbow_plot.png")
    plt.savefig(plot_path)
    print(f"[✔] Gráfico salvo em {plot_path}")

    # =========================================================================
    # 4. AVALIAÇÃO FINAL NO TESTE
    # =========================================================================
    best_n, best_k, best_feats_indices = best_config
    print(f"\n>>> CONFIGURAÇÃO IDEAL: {best_n} Features | k={best_k} (Val F1: {best_f1_val:.4f})")
    
    X_tr_final = X_train[:, best_feats_indices]
    X_ts_final = X_test[:, best_feats_indices]
    
    knn_final = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
    knn_final.fit(X_tr_final, y_train)
    y_pred_test = knn_final.predict(X_ts_final)
    
    print("\n--- 3.3.4 Resultados Finais (Modelo Ideal) ---")
    metrics, cm = evaluate_and_save_metrics(y_test, y_pred_test, save_dir, fold_name="ideal_model_test")
    
    print(f"Validação F1: {best_f1_val:.4f}")
    print(f"Teste F1:     {metrics['F1-score'].iloc[0]:.4f}")
    
    return best_feats_indices

def mlp_experiment(X_train, y_train, X_val, y_val, X_test, y_test, selected_features, lr_mode='constant'):

    if lr_mode == 'constant':
        title = "4.1 MLP - Taxa Fixa"
        save_dir = "data/results/mlp_fixed"
    else:
        title = "4.2 MLP - Taxa Variável (Adaptive)"
        save_dir = "data/results/mlp_variable"

    os.makedirs(save_dir, exist_ok=True)
    print("\n" + "="*40 + f"\n{title}\n" + "="*40)
    
    # 1. FILTRAR FEATURES (Usar apenas as selecionadas no ponto 3.3)
    print(f">>> A selecionar as {len(selected_features)} melhores features...")
    X_tr_sel = X_train[:, selected_features]
    X_val_sel = X_val[:, selected_features]
    X_ts_sel = X_test[:, selected_features]

    # 2. NORMALIZAR DADOS (Obrigatório para MLP)
    # Ajustamos o scaler apenas no treino e aplicamos nos outros
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_sel)
    X_val_scaled = scaler.transform(X_val_sel)
    X_ts_scaled = scaler.transform(X_ts_sel)

    # 3. GRID SEARCH
    neuron_configs = [10, 50, 100, 200]
    activations = ['logistic', 'relu']
    
    results = []
    best_val_f1 = -1
    best_model = None
    best_params = None

    print(f"{'Ativação':<10} | {'Neurónios':<10} | {'LR Mode':<10} | {'Val F1':<10}")
    print("-" * 55)

    for act in activations:
        for neurons in neuron_configs:
            # Definição do Modelo
            mlp = MLPClassifier(hidden_layer_sizes=(neurons,), activation=act, learning_rate=lr_mode, learning_rate_init=0.01, batch_size=200, random_state=42, max_iter=500)
            
            mlp.fit(X_tr_scaled, y_train)
            
            # Validar
            y_pred_val = mlp.predict(X_val_scaled)
            f1_val = f1_score(y_val, y_pred_val, average='macro', zero_division=0)
            
            print(f"{act:<10} | {neurons:<10} | {lr_mode:<10} | {f1_val:.4f}")
            
            results.append({
                "activation": act,
                "neurons": neurons,
                "learning_rate": lr_mode,
                "f1_val": f1_val
            })
            
            if f1_val > best_val_f1:
                best_val_f1 = f1_val
                best_model = mlp
                best_params = (act, neurons)

    # 4. AVALIAÇÃO FINAL NO TESTE
    print(f"\n>>> MELHOR CONFIGURAÇÃO ({lr_mode}):")
    print(f"Ativação: {best_params[0]} | Neurónios: {best_params[1]} | Val F1: {best_val_f1:.4f}")
    
    y_pred_test = best_model.predict(X_ts_scaled)
    
    print(f"\n--- Resultados Finais MLP {lr_mode} (Test Set) ---")
    evaluate_and_save_metrics(y_test, y_pred_test, save_dir, fold_name=f"mlp_{lr_mode}_best")
    
    # Salvar CSV com histórico
    pd.DataFrame(results).to_csv(os.path.join(save_dir, "grid_search_results.csv"), index=False)