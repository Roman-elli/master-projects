import os
import numpy as np
import config as cfg
from scipy import stats
from scipy.stats import zscore, pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import pickle
import pandas as pd

def _get_fft(d, fs):
    return np.abs(np.fft.rfft(d)), np.fft.rfftfreq(len(d), 1/fs)

def _safe_stat(func, d):
    if np.std(d) < 1e-9: return 0.0
    return func(d)

def _pairwise_corr(v1, v2):
    if np.std(v1) < 1e-9 or np.std(v2) < 1e-9: return 0.0
    return np.corrcoef(v1, v2)[0, 1]

def _zcr(d):
    if len(d) < 2: return 0.0
    d_centered = d - np.mean(d)
    return ((d_centered[:-1] * d_centered[1:]) < 0).sum() / len(d)

def _mcr(d):
    if len(d) < 2: return 0.0
    mean_val = np.mean(d)
    return (((d[:-1] - mean_val) * (d[1:] - mean_val)) < 0).sum() / len(d)

def _spec_entropy(d, fs):
    fft_vals, _ = _get_fft(d, fs)
    # Avoid division by zero
    sum_sq = np.sum(fft_vals**2)
    if sum_sq < 1e-12: return 0.0
    psd = fft_vals**2 / sum_sq
    return -np.sum(psd * np.log(psd + 1e-12))

# ==============================================================================
# 2. CORE EXTRACTION UNIT (110 Features)
# ==============================================================================

def _extract_unit_110(Ax, Ay, Az, Gx, Gy, Gz, fs):
    """
    Core function: Calculates 110 features for a 6-axis block (Acc+Gyro).
    """
    dt = 1.0 / fs
    channels = [Ax, Ay, Az, Gx, Gy, Gz]
    features = []

    # A. STATISTICS (84 features: 14 stats * 6 channels)
    for ch in channels:
        fft_vals, fft_freqs = _get_fft(ch, fs)
        features.extend([
            np.mean(ch), np.median(ch), np.std(ch), np.var(ch), np.sqrt(np.mean(ch**2)),
            np.mean(np.abs(np.diff(ch))),
            _safe_stat(stats.skew, ch), _safe_stat(stats.kurtosis, ch),
            stats.iqr(ch), _zcr(ch), _mcr(ch),
            fft_freqs[np.argmax(fft_vals)] if len(fft_vals) > 0 else 0,
            np.sum(fft_vals**2) / len(ch),
            _spec_entropy(ch, fs)
        ])

    # B. CORRELATIONS (15 features)
    pairs = [(Ax,Ay), (Ax,Az), (Ay,Az), (Gx,Gy), (Gx,Gz), (Gy,Gz),
             (Ax,Gx), (Ay,Gy), (Az,Gz), (Ax,Gy), (Ax,Gz), (Ay,Gx), (Ay,Gz), (Az,Gx), (Az,Gy)]
    for v1, v2 in pairs:
        features.append(_pairwise_corr(v1, v2))

    # C. PHYSICAL FEATURES (11 features)
    mi = np.sqrt(Ax**2 + Ay**2 + Az**2)
    features.extend([np.mean(mi), np.var(mi)])
    features.append(np.sum(np.abs(Ax) + np.abs(Ay) + np.abs(Az)) / len(Ax)) # SMA
    
    try:
        eigvals = np.linalg.eigvalsh(np.cov(np.vstack((Ax, Ay, Az))))
        features.extend([eigvals[-1], eigvals[-2]])
    except: features.extend([0.0, 0.0])

    heading = np.sqrt(Ay**2 + Az**2)
    features.append(_pairwise_corr(Ax, heading)) # CAGH
    
    vy, vz = np.cumsum(Ay) * dt, np.cumsum(Az) * dt
    features.append(np.mean(np.sqrt(vy**2 + vz**2))) # AVH
    features.append(np.mean(np.cumsum(Ax) * dt)) # AVG
    features.append(np.mean(np.cumsum(Gx) * dt)) # ARATG
    
    aae = np.mean(np.sum([Ax**2, Ay**2, Az**2], axis=0))
    are = np.mean(np.sum([Gx**2, Gy**2, Gz**2], axis=0))
    features.extend([aae, are])

    return features

def get_feature_names_110(prefix=""):
    """Generates names for the 110 features."""
    ch = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
    st = ['Mean', 'Median', 'Std', 'Var', 'RMS', 'AvgDeriv', 'Skew', 'Kurt', 'IQR', 'ZCR', 'MCR', 'DomFreq', 'Energy', 'SpecEnt']
    base_names = [f"{c}_{s}" for c in ch for s in st]
    
    corr_names = ['AxAy', 'AxAz', 'AyAz', 'GxGy', 'GxGz', 'GyGz', 'AxGx', 'AyGy', 'AzGz', 'AxGy', 'AxGz', 'AyGx', 'AyGz', 'AzGx', 'AzGy']
    base_names += [f"Corr_{p}" for p in corr_names]
    
    phys_names = ['AI', 'VI', 'SMA', 'EVA1', 'EVA2', 'CAGH', 'AVH', 'AVG', 'ARATG', 'AAE', 'ARE']
    base_names += phys_names
    if prefix:
        return [f"{prefix}_{n}" for n in base_names]
    return base_names

# ==============================================================================
# 3. EXTRACTION FUNCTIONS (PART A COMPATIBILITY)
# ==============================================================================

def build_feature_matrix_activity(data_array, activity_id, fs=50, window_ms=1000, overlap=0.5, body_part_idx=None, body_part_name=None, sensor=None):
    """
    Extrai features para uma atividade específica de uma parte do corpo.
    AGORA PEGA SEMPRE ACC + GYRO JUNTOS (110 Features).
    Ignora o argumento 'sensor' para a extração, mas usa-o apenas para log/caminho.
    """
    X = []
    
    # Define as colunas fixas: 0,1,2 (Acc) e 3,4,5 (Gyro)
    # Assumindo que os dados brutos vêm sempre nesta ordem: [Ax, Ay, Az, Gx, Gy, Gz, Mx, My, Mz, ...]
    cols_acc_gyro = [0, 1, 2, 3, 4, 5]

    # Cálculos de janela
    win_len = int(fs * (window_ms / 1000))
    step_len = int(win_len * (1 - overlap))

    for person in data_array:
        if body_part_idx >= len(person): continue
        
        # Pega a matriz completa do sensor (todas as colunas)
        activity_data = person[body_part_idx]

        # Identificar coluna da Label (geralmente 11 ou a última)
        lbl_col = 11 if activity_data.shape[1] > 11 else -1
        
        # Filtrar atividade
        mask = activity_data[:, lbl_col].astype(int) == activity_id
        act_data = activity_data[mask]

        if len(act_data) < win_len: continue

        # Loop de Janela Deslizante
        for start in range(0, len(act_data) - win_len + 1, step_len):
            # Recorta janela apenas com as 6 colunas de interesse (Acc+Gyro)
            window = act_data[start : start + win_len, cols_acc_gyro]
            
            # Separa os canais
            Ax, Ay, Az = window[:, 0], window[:, 1], window[:, 2]
            Gx, Gy, Gz = window[:, 3], window[:, 4], window[:, 5]

            try:
                # Calcula as 110 features UNIFICADAS
                feats = _extract_unit_110(Ax, Ay, Az, Gx, Gy, Gz, fs)
                X.append(feats)
            except Exception:
                continue

    X = np.array(X)
    
    # Nomes das features
    prefix = f"{body_part_name}" if body_part_name else "Sensor"
    feat_names = get_feature_names_110(prefix)
    
    return X, feat_names

# ==============================================================================
# 4. EXTRACTION FUNCTIONS
# ==============================================================================

def extract_features_550(window_data, fs=50):
    """
    Used in Part B: Extracts 110 features for each of the 5 sensors.
    Input: Window (N, 33) -> [ID, TS, S1(6), S2(6)... S5(6), Label]
    """
    all_features = [] 
    start_col = 2 # Skip ID and TS
    
    for i in range(5):
        end_col = start_col + 6
        if end_col > window_data.shape[1]:
            all_features.extend([0.0] * 110)
            continue

        block = window_data[:, start_col:end_col]
        
        # 0,1,2=Acc | 3,4,5=Gyro
        Ax, Ay, Az = block[:, 0], block[:, 1], block[:, 2]
        Gx, Gy, Gz = block[:, 3], block[:, 4], block[:, 5]

        feats_110 = _extract_unit_110(Ax, Ay, Az, Gx, Gy, Gz, fs)
        all_features.extend(feats_110)
        
        start_col = end_col

    return np.nan_to_num(np.array(all_features))

def get_feature_names_550():
    # Helper to generate the 550 names [S1_Ax_Mean ... S5_ARE]
    base_names = get_feature_names_110("") # Get base names without prefix
    base_names = [n.lstrip("_") for n in base_names] # clean
    
    final_names = []
    sensor_labels = ['S1', 'S2', 'S3', 'S4', 'S5']
    for s_label in sensor_labels:
        final_names += [f"{s_label}_{n}" for n in base_names]
    return final_names

# ==============================================================================
# 5. ANALYSIS FUNCTIONS (PCA, FEATURE SELECTION, STATS)
# ==============================================================================

def statistical_significance(data_array, body_part_idx=None, body_part_name=None, save_dir="data/statistics"):
    """
    Calcula a significância estatística (Teste KS de Normalidade) para Acc e Gyro.
    Gera um relatório único por Parte do Corpo.
    """
    # Pasta unificada
    save_dir = os.path.join(save_dir, "combined_acc_gyro", body_part_name)
    os.makedirs(save_dir, exist_ok=True)
    
    results_file = os.path.join(save_dir, "normality_test.txt")

    with open(results_file, "w", encoding='utf-8') as f:
        f.write(f"=== Análise Estatística (Normality Check) - {body_part_name} ===\n")
        f.write("Nota: p-value < 0.05 indica que os dados NÃO seguem uma distribuição normal.\n\n")

        # Listas para acumular magnitudes
        acc_mods, gyro_mods, all_activities = [], [], []

        # Índices das colunas (Fixo: 0-2 Acc, 3-5 Gyro)
        acc_cols = [0, 1, 2]
        gyro_cols = [3, 4, 5]

        for person in data_array:
            if body_part_idx >= len(person): continue
            
            activity = person[body_part_idx]
            
            # 1. Módulo Acelerómetro
            ax, ay, az = activity[:, acc_cols[0]], activity[:, acc_cols[1]], activity[:, acc_cols[2]]
            acc_mods.extend(np.sqrt(ax**2 + ay**2 + az**2))

            # 2. Módulo Giroscópio
            gx, gy, gz = activity[:, gyro_cols[0]], activity[:, gyro_cols[1]], activity[:, gyro_cols[2]]
            gyro_mods.extend(np.sqrt(gx**2 + gy**2 + gz**2))

            # Label
            lbl_col = 11 if activity.shape[1] > 11 else -1
            all_activities.extend(activity[:, lbl_col].astype(int))

        # Converter para numpy arrays para processamento rápido
        acc_mods = np.array(acc_mods)
        gyro_mods = np.array(gyro_mods)
        all_activities = np.array(all_activities)
        unique_acts = np.unique(all_activities)

        # Análise por Atividade
        for a in unique_acts:
            f.write(f"--- Atividade {a} ---\n")
            mask = (all_activities == a)
            
            # --- ACELERÓMETRO ---
            curr_acc = acc_mods[mask]
            mean_acc = np.mean(curr_acc)
            if len(curr_acc) > 1 and np.std(curr_acc) > 0:
                _, p_acc = stats.kstest(zscore(curr_acc), 'norm')
            else: p_acc = 1.0 # Dados constantes ou insuficientes
            
            f.write(f"  [ACC]  Média: {mean_acc:.4f} | KS p-value: {p_acc:.4e}\n")

            # --- GIROSCÓPIO ---
            curr_gyro = gyro_mods[mask]
            mean_gyro = np.mean(curr_gyro)
            if len(curr_gyro) > 1 and np.std(curr_gyro) > 0:
                _, p_gyro = stats.kstest(zscore(curr_gyro), 'norm')
            else: p_gyro = 1.0

            f.write(f"  [GYRO] Média: {mean_gyro:.4f} | KS p-value: {p_gyro:.4e}\n")
            f.write("\n")

    print(f"[OK] Estatísticas salvas em: {results_file}")


# ==============================================================================
# 5. ANÁLISE (PCA e FEATURE SELECTION) - ATUALIZADO PARA DADOS COMBINADOS
# ==============================================================================

def relieff_single_activity(X, k=10):
    """
    ReliefF adaptado para atividade única (não supervisionado/local).
    Mede a relevância da feature baseada na distância aos vizinhos locais.
    Se a feature muda drasticamente entre vizinhos próximos, é instável (score baixo).
    """
    n_samples, n_features = X.shape
    scores = np.zeros(n_features)
    
    # Encontra os k vizinhos mais próximos para cada ponto
    # Usa apenas uma amostra dos dados se for muito grande para ser rápido
    if n_samples > 2000:
        idx = np.random.choice(n_samples, 2000, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    nbrs = NearestNeighbors(n_neighbors=min(k+1, len(X_sample))).fit(X_sample)
    
    for i in range(len(X_sample)):
        distances, indices = nbrs.kneighbors([X_sample[i]])
        # indices[0][0] é o próprio ponto, indices[0][1:] são os vizinhos
        neighbors_idx = indices[0][1:]
        
        # Calcula diferença média para os vizinhos em cada feature
        diffs = np.abs(X_sample[i] - X_sample[neighbors_idx])
        # Soma a diferença média (quanto maior a diferença, "pior" a feature para manter padrão)
        scores += np.mean(diffs, axis=0)
        
    # Inverter: Queremos features que variem POUCO localmente (estáveis)
    scores = 1.0 / (scores + 1e-9)
    return scores

def apply_feature_selection_to_activity(sensor, body_part, activity_id):
    print(f"   -> Calculando Fisher e ReliefF: {body_part} | Act {activity_id}...")
    
    # 1. Definir Caminho
    act_dir = os.path.join("data", "features_analysis", sensor, body_part.replace(" ","_"), f"act_{activity_id}")
    fpath = os.path.join(act_dir, "features.csv")

    if not os.path.exists(fpath): 
        print(f"      [!] Arquivo não encontrado: {fpath}")
        return

    try:
        # Ler Header e Dados
        with open(fpath, 'r') as f:
            feat_names = f.readline().strip().split(';')
        X = np.loadtxt(fpath, delimiter=';', skiprows=1)
        if X.ndim == 1: X = X.reshape(1, -1)
        
        # Se tiver poucas amostras, ignora
        if X.shape[0] < 5: return

    except Exception as e: 
        print(f"      [!] Erro leitura: {e}")
        return
    
    # 1. Fisher Simplificado (Signal-to-Noise Ratio)
    # NÃO use StandardScaler aqui. Queremos a magnitude real.
    # Score alto = Sinal forte e estável. Score baixo = Ruído.
    means = np.mean(np.abs(X), axis=0)
    stds = np.std(X, axis=0)
    
    # Evitar divisão por zero
    valid = stds > 1e-9
    fisher_scores = np.zeros(X.shape[1])
    fisher_scores[valid] = means[valid] / stds[valid]

    # 2. ReliefF (Local Consistency)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    relief_scores = relieff_single_activity(X_scaled, k=10)

    # === SALVAR ===
    save_dir = os.path.join(act_dir, "feature_selection")
    os.makedirs(save_dir, exist_ok=True)
    
    # Salvar Fisher
    with open(os.path.join(save_dir, "fisher_scores.csv"), 'w') as f:
        f.write("Feature;Score\n")
        for n, s in zip(feat_names, fisher_scores):
            f.write(f"{n};{s:.6f}\n")
            
    # Salvar Relief
    with open(os.path.join(save_dir, "relief_scores.csv"), 'w') as f:
        f.write("Feature;Score\n")
        for n, s in zip(feat_names, relief_scores):
            f.write(f"{n};{s:.6f}\n")
            
    print(f"      [✔] Scores salvos em: {save_dir}")

def apply_pca_to_activity(sensor, body_part, activity_id):
    """
    Aplica PCA nas 110 features combinadas para ver a redução de dimensionalidade.
    """
    print(f"\n=== PCA -> {body_part} | Act {activity_id} ===")
    
    # CAMINHO CRÍTICO: Pasta de análise da Parte A
    act_dir = os.path.join("data", "features_analysis", sensor, body_part.replace(" ","_"), f"act_{activity_id}")
    fpath = os.path.join(act_dir, "features.csv")

    if not os.path.exists(fpath): return

    try:
        X = np.loadtxt(fpath, delimiter=';', skiprows=1)
        if X.ndim == 1: X = X.reshape(1, -1)
    except: return

    # 1. Normalizar (Obrigatório para PCA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Aplicar PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Variância Explicada Acumulada
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    
    # Quantos componentes explicam 95%?
    n_95 = np.argmax(cum_var >= 0.95) + 1

    # 3. Salvar Resultados e Gráfico
    pca_dir = os.path.join(act_dir, "pca_results")
    os.makedirs(pca_dir, exist_ok=True)
    
    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(cum_var*100, linewidth=2)
    plt.axvline(n_95, color='r', linestyle='--', label=f'95% Var ({n_95} Comps)')
    plt.axhline(95, color='r', linestyle=':', alpha=0.5)
    
    plt.xlabel("Número de Componentes Principais")
    plt.ylabel("Variância Explicada Acumulada (%)")
    plt.title(f"PCA Analysis: {body_part} - Act {activity_id}")
    plt.legend()
    plt.grid(True)
    
    img_path = os.path.join(pca_dir, "pca_variance.png")
    plt.savefig(img_path)
    plt.close()
    
    # Salvar modelo (opcional)
    with open(os.path.join(pca_dir, "pca_model.pkl"), "wb") as f:
        pickle.dump(pca, f)
    
    print(f"[✔] PCA concluído. 95% var com {n_95}/{X.shape[1]} componentes. Gráfico em {img_path}")
