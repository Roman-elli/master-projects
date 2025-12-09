import os
import numpy as np
import config as cfg
from scipy import stats
from scipy.stats import zscore, ttest_ind, f_oneway, kruskal, spearmanr, pearsonr, mode
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import pickle

# =========================================
# 0 — Função auxiliar para verificar NaN/Inf
# =========================================

def safe_stat(func, arr):
    try:
        val = func(arr)
        if np.isnan(val) or np.isinf(val):
            return 0.0
        return float(val)
    except Exception:
        return 0.0

# =========================================
# 1 — Teste de significância e gaussianidade
# =========================================
def statistical_significance(data_array, sensor='acc', body_part_idx=None, body_part_name=None, save_dir="data/statistics"):
    save_dir = os.path.join(save_dir, sensor, body_part_name)
    os.makedirs(save_dir, exist_ok=True)

    cols = cfg.SENSOR_COLS[sensor]
    results_file = os.path.join(save_dir, f"{sensor}_significance.txt")

    with open(results_file, "w", encoding='utf-8') as f:
        f.write(f"=== Significância Estatística ({sensor.upper()} - {body_part_name}) ===\n")

        all_modules = []
        all_activities = []

        for person in data_array:
            activity = person[body_part_idx]
            x, y_sig, z = activity[:, cols[0]], activity[:, cols[1]], activity[:, cols[2]]
            module = np.sqrt(x**2 + y_sig**2 + z**2)
            act = activity[:,11].astype(int)
            all_modules.extend(module)
            all_activities.extend(act)

        all_modules = np.array(all_modules)
        all_activities = np.array(all_activities)
        unique_acts = np.unique(all_activities)

        for a in unique_acts:
            vals = all_modules[all_activities==a]
            mean_val = np.mean(vals)
            ks_stat, p_val = stats.kstest(zscore(vals), 'norm')
            f.write(f"Atividade {a}: média={mean_val:.4f}, KS p={p_val:.4f}\n")

    print(f"[OK] Resultados básicos salvos em {results_file}")

# =========================================
# 2 — Testes estatísticos avançados (ANOVA / Kruskal Wallis)
# =========================================
def advanced_statistical_tests(X, y, feature_names, sensor='acc', body_part_name=None, save_dir="data/statistics"):
    save_dir = os.path.join(save_dir, sensor, body_part_name)
    os.makedirs(save_dir, exist_ok=True)
    unique_acts = np.unique(y)
    results_file = os.path.join(save_dir, "advanced_stats.txt")

    with open(results_file, "w", encoding='utf-8') as f:
        f.write("=== Testes Estatísticos Avançados ===\n")

        for i, feat in enumerate(feature_names):
            f.write(f"\nFeature: {feat}\n")
            groups = [X[y==a,i] for a in unique_acts]

            if len(unique_acts) == 2:
                g1, g2 = groups
                try:
                    t_stat, t_p = ttest_ind(g1,g2)
                    mw_stat, mw_p = stats.mannwhitneyu(g1,g2)
                    f.write(f"T-test p={t_p:.4f}, Mann-Whitney p={mw_p:.4f}\n")
                except Exception as e:
                    f.write(f"[!] Erro nos testes 2 a 2: {e}\n")
            elif len(unique_acts) > 2:
                try:
                    if np.all([np.std(g)<1e-8 for g in groups]):
                        raise ValueError("Sem variação entre grupos")
                    f_stat, f_p = f_oneway(*groups)
                    try:
                        kw_stat, kw_p = kruskal(*groups)
                    except ValueError as e:
                        kw_p = np.nan
                    f.write(f"ANOVA p={f_p:.4f}, Kruskal-Wallis p={kw_p:.4f}\n")
                except Exception as e:
                    f.write(f"[!] Erro ANOVA/Kruskal: {e}\n")

        # Correlações entre features
        f.write("\n=== Correlações entre features ===\n")
        for i in range(len(feature_names)):
            for j in range(i+1, len(feature_names)):
                if np.std(X[:,i])<1e-8 or np.std(X[:,j])<1e-8:
                    continue
                try:
                    p_corr, p_p = pearsonr(X[:,i], X[:,j])
                    s_corr, s_p = spearmanr(X[:,i], X[:,j])
                    f.write(f"{feature_names[i]} vs {feature_names[j]} → "
                            f"Pearson r={p_corr:.3f} (p={p_p:.4e}), "
                            f"Spearman ρ={s_corr:.3f} (p={s_p:.4e})\n")
                except:
                    continue
    print(f"[OK] Resultados avançados salvos em {results_file}")

# =========================================
# 3 — Extração de features
# =========================================

def temporal_features(x,y,z,fs=50):
    feats = {}
    for name, arr in zip(['X','Y','Z'], [x,y,z]):
        feats[f'Mean_{name}'] = np.mean(arr)
        feats[f'Median_{name}'] = np.median(arr)
        feats[f'Std_{name}'] = np.std(arr)
        feats[f'Variance_{name}'] = np.var(arr)
        feats[f'RMS_{name}'] = np.sqrt(np.mean(arr**2))
        feats[f'AvgDerivative_{name}'] = np.mean(np.abs(np.diff(arr)))
        feats[f'Skewness_{name}'] = safe_stat(stats.skew, arr)
        feats[f'Kurtosis_{name}'] = safe_stat(stats.kurtosis, arr)
        feats[f'IQR_{name}'] = np.percentile(arr,75) - np.percentile(arr,25)
        feats[f'ZeroCross_{name}'] = np.sum((arr[:-1]*arr[1:])<0)/(len(arr)/fs)
        feats[f'MeanCross_{name}'] = np.sum((arr[:-1]-np.mean(arr))*(arr[1:]-np.mean(arr))<0)/(len(arr)/fs)
        feats[f'MedianAbsDev_{name}'] = np.median(np.abs(arr - np.median(arr)))
        feats[f'Range_{name}'] = np.max(arr) - np.min(arr)
    # Correlation
    feats['Correlation_XY'] = np.corrcoef(x,y)[0,1]
    feats['Correlation_XZ'] = np.corrcoef(x,z)[0,1]
    feats['Correlation_YZ'] = np.corrcoef(y,z)[0,1]
    return feats

def fft_features(x,y,z,fs=50):
    feats = {}
    for name, arr in zip(['X','Y','Z'], [x,y,z]):

        fft_vals = np.fft.rfft(arr)
        fft_freq = np.fft.rfftfreq(len(arr), 1/fs)
        feats[f'FFT_Max_{name}'] = np.max(np.abs(fft_vals))
        feats[f'FFT_Min_{name}'] = np.min(np.abs(fft_vals))
        feats[f'MeanFFT_{name}'] = np.mean(np.abs(fft_vals))
        feats[f'StdFFT_{name}'] = np.std(np.abs(fft_vals))
        feats[f'FFT_Energy_{name}'] = np.sum(np.abs(fft_vals)**2)/len(arr)
        feats[f'FFT_DominantFreq_{name}'] = fft_freq[np.argmax(np.abs(fft_vals))]
        feats[f'SkewnessFFT_{name}'] = stats.skew(np.abs(fft_vals))
        feats[f'KurtosisFFT_{name}'] = stats.kurtosis(np.abs(fft_vals))
        # Spectral entropy
        P = np.abs(fft_vals)**2
        P /= np.sum(P)
        feats[f'SpectralEntropy_{name}'] = -np.sum(P*np.log(P+1e-12))
    return feats

def physical_features(x, y, z, fs=50, sensor='acc'):
    feats = {}
    MI = np.sqrt(x**2 + y**2 + z**2)
    feats['AI'] = np.mean(MI)
    feats['VI'] = np.var(MI)
    feats['SMA'] = np.mean(np.abs(x)+np.abs(y)+np.abs(z))
    cov = np.cov(np.vstack([x,y,z]))
    eigvals = np.sort(np.linalg.eigvals(cov))[::-1]
    feats['EVA1'], feats['EVA2'] = eigvals[:2]
    heading = np.sqrt(y**2 + z**2)
    feats['CAGH'] = np.corrcoef(x, heading)[0,1]
    vx, vy, vz = np.cumsum(x)/fs, np.cumsum(y)/fs, np.cumsum(z)/fs
    feats['AVH'] = np.sqrt(np.mean(vy)**2 + np.mean(vz)**2)
    feats['AVG'] = np.mean(vx)
    feats['ARATG'] = np.arctan2(np.mean(y), np.mean(z))
    fft_mag = np.abs(np.fft.rfft(MI))**2
    freqs = np.fft.rfftfreq(len(MI),1/fs)
    feats['DF'] = freqs[np.argmax(fft_mag[1:])]
    feats['ENERGY'] = np.sum(fft_mag)/len(MI)
    energy_x = np.sum(np.abs(np.fft.rfft(x))**2)/len(x)
    energy_y = np.sum(np.abs(np.fft.rfft(y))**2)/len(y)
    energy_z = np.sum(np.abs(np.fft.rfft(z))**2)/len(z)
    feats['AAE'] = np.mean([energy_x, energy_y, energy_z])
    if sensor=='gyro':
        feats['ARE'] = np.mean([energy_x, energy_y, energy_z])
    feats['MaxAccel_X'] = np.max(x)
    feats['MaxAccel_Y'] = np.max(y)
    feats['MaxAccel_Z'] = np.max(z)
    feats['MinAccel_X'] = np.min(x)
    feats['MinAccel_Y'] = np.min(y)
    feats['MinAccel_Z'] = np.min(z)
    feats['RMSAccel_X'] = np.sqrt(np.mean(x**2))
    feats['RMSAccel_Y'] = np.sqrt(np.mean(y**2))
    feats['RMSAccel_Z'] = np.sqrt(np.mean(z**2))
    feats['TotalEnergy'] = energy_x + energy_y + energy_z


    # --- Magnitude dos sinais ---
    feats['SignalMagnitude_X'] = np.sqrt(np.mean(x**2))
    feats['SignalMagnitude_Y'] = np.sqrt(np.mean(y**2))
    feats['SignalMagnitude_Z'] = np.sqrt(np.mean(z**2))

    # --- Covariâncias ---
    if np.all(np.std([x, y, z], axis=1) < 1e-8):
        feats['Cov_XY'] = feats['Cov_XZ'] = feats['Cov_YZ'] = 0.0
    else:
        feats['Cov_XY'] = np.cov(x, y)[0, 1]
        feats['Cov_XZ'] = np.cov(x, z)[0, 1]
        feats['Cov_YZ'] = np.cov(y, z)[0, 1]

    # --- Entropia temporal ---
    def entropy_time(arr):
        arr = arr - np.mean(arr)
        hist, _ = np.histogram(arr, bins=30, density=True)
        hist += 1e-12
        P = hist/np.sum(hist)
        return -np.sum(P*np.log(P))

    feats['EntropyTime_X'] = entropy_time(x)
    feats['EntropyTime_Y'] = entropy_time(y)
    feats['EntropyTime_Z'] = entropy_time(z)

    # --- Máximos e mínimos ---
    feats['Max_X'] = np.max(x)
    feats['Max_Y'] = np.max(y)
    feats['Max_Z'] = np.max(z)
    feats['Min_X'] = np.min(x)
    feats['Min_Y'] = np.min(y)
    feats['Min_Z'] = np.min(z)

    # --- Máximos FFT ---
    feats['MaxFFT_X'] = np.max(np.abs(np.fft.rfft(x)))
    feats['MaxFFT_Y'] = np.max(np.abs(np.fft.rfft(y)))
    feats['MaxFFT_Z'] = np.max(np.abs(np.fft.rfft(z)))

    # --- Energia rotacional (ARE) para giroscópio ---
    if sensor == 'gyro':
        feats['ARE'] = np.mean(np.abs(np.fft.rfft(x))**2 + np.abs(np.fft.rfft(y))**2 + np.abs(np.fft.rfft(z))**2)

    return feats

def extract_features(x, y, z, fs=50, vector='acc'):
    feats = {}
    feats.update(temporal_features(x, y, z, fs))
    feats.update(fft_features(x, y, z, fs))
    feats.update(physical_features(x, y, z, fs, vector))
    feat_names = list(feats.keys())
    feat_vector = np.array([feats[k] for k in feat_names], dtype=float)
    feat_vector = np.nan_to_num(feat_vector, nan=0.0, posinf=0.0, neginf=0.0)
    return feat_vector, feat_names

# =========================================
# 4 — Construção da matriz com intervalos temporais por sensor
# =========================================

def build_feature_matrix(data_array, sensor='acc', fs=50, body_part_idx=None, body_part_name=None):
    X, y = [], []
    cols = cfg.SENSOR_COLS[sensor]
    for person in data_array:
        activity = person[body_part_idx]
        x, y_sig, z = activity[:,cols[0]], activity[:,cols[1]], activity[:,cols[2]]
        label = int(mode(activity[:,11], keepdims=True)[0])
        feat_vec, feat_names = extract_features(x,y_sig,z,fs,vector=sensor)
        X.append(feat_vec)
        y.append(label)
    X = np.array(X)
    y = np.array(y)
    print(f"[OK] Sensor: {sensor} | Parte: {body_part_name} | labels únicos: {np.unique(y)}")
    return X, y, feat_names

# =========================================
# 4 — Construção de matriz / calculo features com intervalos temporais por atividade
# =========================================

def build_feature_matrix_activity(data_array, activity_id, sensor='acc', fs=51.2, window_ms=2000, overlap=0.5, body_part_idx=None, body_part_name=None):
    X = []
    cols = cfg.SENSOR_COLS[sensor]

    step_ms = window_ms * (1 - overlap)
    min_samples = int(fs * (window_ms / 1000) * 0.5)  # pelo menos 50% da janela (ex: 1s)

    for person in data_array:
        activity = person[body_part_idx]

        # Filtrar só a atividade desejada
        activity = activity[activity[:, 11].astype(int) == activity_id]
        if len(activity) < min_samples:
            continue

        timestamps = activity[:, 10]
        timestamps = timestamps - np.min(timestamps)
        t_max = np.max(timestamps)
        if t_max < window_ms:
            continue  # sinal muito curto

        n_windows = int(np.floor((t_max - window_ms) / step_ms)) + 1

        for w in range(n_windows):
            t_start = w * step_ms
            t_end = t_start + window_ms
            mask = (timestamps >= t_start) & (timestamps < t_end)
            subset = activity[mask]

            if subset.shape[0] < min_samples:
                continue

            x, y_sig, z = subset[:, cols[0]], subset[:, cols[1]], subset[:, cols[2]]

            # Substituir valores inválidos
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            y_sig = np.nan_to_num(y_sig, nan=0.0, posinf=0.0, neginf=0.0)
            z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

            try:
                feat_vec, feat_names = extract_features(x, y_sig, z, fs=fs, vector=sensor)
                if np.any(np.isnan(feat_vec)) or np.any(np.isinf(feat_vec)):
                    continue  # ignora janelas problemáticas
                X.append(feat_vec)
            except Exception as e:
                print(f"[!] Erro ao calcular features: {e}")
                continue

        # Última janela parcial
        if t_max > n_windows * step_ms:
            t_start = n_windows * step_ms
            mask = (timestamps >= t_start) & (timestamps <= t_max)
            subset = activity[mask]
            if subset.shape[0] >= min_samples:
                x, y_sig, z = subset[:, cols[0]], subset[:, cols[1]], subset[:, cols[2]]
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                y_sig = np.nan_to_num(y_sig, nan=0.0, posinf=0.0, neginf=0.0)
                z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
                try:
                    feat_vec, feat_names = extract_features(x, y_sig, z, fs=fs, vector=sensor)
                    if not np.any(np.isnan(feat_vec)) and not np.any(np.isinf(feat_vec)):
                        X.append(feat_vec)
                except:
                    pass

    if len(X) == 0:
        raise ValueError(f"Nenhuma janela válida encontrada para atividade {activity_id} ({body_part_name})")

    X = np.array(X)
    print(f"[OK] {sensor} | {body_part_name} | Atividade {activity_id}: {len(X)} janelas válidas")
    return X, feat_names

# =========================================
# 6 — PCA
# =========================================

def apply_pca_to_activity(sensor, body_part, activity_id):
    print(f"\n=== PCA -> Sensor: {sensor.upper()} | Parte: {body_part} | Atividade {activity_id} ===")
    
    act_dir = os.path.join(cfg.BASE_DATA_PATH, sensor, body_part, f"act_{activity_id}")
    fpath = os.path.join(act_dir, "features.csv")

    if not os.path.exists(fpath):
        print(f"[!] Arquivo não encontrado: {fpath}")
        return

    # Carregar CSV
    try:
        X = np.loadtxt(fpath, delimiter=';', skiprows=1)
        if X.ndim == 1:
            X = X.reshape(1, -1)
    except Exception as e:
        print(f"[!] Erro ao ler {fpath}: {e}")
        return

    print(f"[OK] Dados: {X.shape[0]} janelas × {X.shape[1]} features")

    # Padronização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.searchsorted(cum_var, cfg.PCA_THRESHOLD) + 1
    print(f"[OK] {cfg.PCA_THRESHOLD*100:.0f}% da variância com {n_components} PCs")

    # Criar pasta de resultados
    pca_dir = os.path.join(act_dir, "pca_results")
    os.makedirs(pca_dir, exist_ok=True)

    # Salvar projeção reduzida
    X_proj = X_pca[:, :n_components]
    np.savetxt(os.path.join(pca_dir, "X_pca.csv"), X_proj, delimiter=';', fmt='%.6f')

    # Salvar modelo e scaler
    with open(os.path.join(pca_dir, "pca_model.pkl"), "wb") as f:
        pickle.dump(pca, f)
    with open(os.path.join(pca_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # --- Plot PCA (sem marca de 0.75) ---
    plt.figure(figsize=(8,5))
    plt.plot(cum_var*100)
    plt.xlabel("Número de Componentes")
    plt.ylabel("Variância Explicada (%)")
    plt.title(f"PCA - {sensor.upper()} | {body_part} | Atividade {activity_id}")
    plt.grid(True)
    plt.savefig(os.path.join(pca_dir, f"{sensor}_{body_part}_act{activity_id}_pca_variance.png"), dpi=200)
    plt.close()

    print(f"[✔] PCA salvo em {pca_dir}")

# =========================================
# 6 — Seleção de top 10 (fisher / relieff)
# =========================================

def apply_feature_selection_to_sensor(sensor, body_parts, top_k=10):
    fisher_scores_all = []
    relief_scores_all = []

    for part in body_parts:
        part_dir = os.path.join(cfg.BASE_DATA_PATH, sensor, part)
        X_all = []
        y_all = []
        feat_names = None  # vai armazenar os nomes das features

        for act_folder in sorted(os.listdir(part_dir)):
            act_path = os.path.join(part_dir, act_folder, "features.csv")
            if not os.path.exists(act_path):
                continue
            
            # Ler os nomes das features (primeira linha)
            if feat_names is None:
                feat_names = np.loadtxt(act_path, delimiter=';', max_rows=1, dtype=str)

            # Carregar os valores
            X = np.loadtxt(act_path, delimiter=';', skiprows=1)
            if X.ndim == 1:
                X = X.reshape(1,-1)
            
            labels = np.repeat(int(act_folder.split("_")[1]), X.shape[0])
            X_all.append(X)
            y_all.append(labels)

        if len(X_all) == 0:
            print(f"[!] Nenhum dado para {sensor} | {part}")
            continue

        X_all = np.vstack(X_all)
        y_all = np.hstack(y_all)

        # Calcular scores
        fisher_scores = fisher_score(X_all, y_all)
        relief_scores = relieff(X_all, y_all)

        # Salvar scores com nomes das features
        save_dir = os.path.join(cfg.BASE_DATA_PATH, sensor, part)
        os.makedirs(save_dir, exist_ok=True)

        # Fisher
        with open(os.path.join(save_dir, "fisher_scores.csv"), 'w', encoding='utf-8') as f:
            f.write("Feature;Score\n")
            for name, score in zip(feat_names, fisher_scores):
                f.write(f"{name};{score:.6f}\n")

        # Relief
        with open(os.path.join(save_dir, "relief_scores.csv"), 'w', encoding='utf-8') as f:
            f.write("Feature;Score\n")
            for name, score in zip(feat_names, relief_scores):
                f.write(f"{name};{score:.6f}\n")

        print(f"[✔] Scores salvos: {sensor} | {part}")
        fisher_scores_all.append(fisher_scores)
        relief_scores_all.append(relief_scores)

    return fisher_scores_all, relief_scores_all

def fisher_score(X, y):
    n_features = X.shape[1]
    scores = np.zeros(n_features)
    classes = np.unique(y)
    overall_mean = np.mean(X, axis=0)
    for i in range(n_features):
        num, den = 0, 0
        for c in classes:
            Xc = X[y==c, i]
            num += len(Xc)*(np.mean(Xc)-overall_mean[i])**2
            den += np.sum((Xc - np.mean(Xc))**2)
        scores[i] = num/(den+1e-12)
    return scores

def relieff(X, y, k=10):
    n_samples, n_features = X.shape
    scores = np.zeros(n_features)
    nn = NearestNeighbors(n_neighbors=min(k+1, n_samples)).fit(X)
    for i in range(n_samples):
        distances, neighbors = nn.kneighbors([X[i]])
        hit_mask = y[neighbors[0][1:]] == y[i]
        miss_mask = ~hit_mask
        for f in range(n_features):
            diff_hit = np.mean((X[i,f]-X[neighbors[0][1:][hit_mask],f])**2) if np.any(hit_mask) else 0
            diff_miss = np.mean((X[i,f]-X[neighbors[0][1:][miss_mask],f])**2) if np.any(miss_mask) else 0
            scores[f] += diff_miss - diff_hit
    scores /= n_samples
    return scores


def fisher_score_single_activity(X):
    """Versão estável do Fisher Score sem comparação entre classes."""
    # Fisher = variância total / variância interna aproximada
    # Aqui usamos o inverso da variância para medir discriminatividade dentro da própria atividade
    var = np.var(X, axis=0)
    mean_abs = np.abs(np.mean(X, axis=0))
    scores = mean_abs / (var + 1e-12)
    return scores


def relieff_single_activity(X, k=10):
    """ReliefF não supervisionado — mede relevância por variação local."""
    n_samples, n_features = X.shape
    scores = np.zeros(n_features)
    nn = NearestNeighbors(n_neighbors=min(k+1, n_samples)).fit(X)
    for i in range(n_samples):
        _, neighbors = nn.kneighbors([X[i]])
        diffs = X[i] - X[neighbors[0][1:], :]
        scores += np.mean(np.abs(diffs), axis=0)
    # menor média de diferença → mais estável → maior score
    scores = 1 / (scores + 1e-12)
    return scores

def apply_feature_selection_to_activity(sensor, body_part, activity_id):
    print(f"\n=== Feature Selection (individual) -> {sensor.upper()} | {body_part} | Act {activity_id} ===")

    act_dir = os.path.join(cfg.BASE_DATA_PATH, sensor, body_part, f"act_{activity_id}")
    fpath = os.path.join(act_dir, "features.csv")

    if not os.path.exists(fpath):
        print(f"[!] Arquivo não encontrado: {fpath}")
        return

    # Lê features
    feat_names = np.loadtxt(fpath, delimiter=';', max_rows=1, dtype=str)
    X = np.loadtxt(fpath, delimiter=';', skiprows=1)

    if X.ndim == 1:
        X = X.reshape(1, -1)

    # --- Normaliza para evitar escalas absurdas ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Calcula os scores ---
    fisher_scores = fisher_score_single_activity(X_scaled)
    relief_scores = relieff_single_activity(X_scaled)

    # --- Salva ---
    save_dir = os.path.join(act_dir, "feature_selection")
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "fisher_scores.csv"), 'w', encoding='utf-8') as f:
        f.write("Feature;Score\n")
        for name, score in zip(feat_names, fisher_scores):
            f.write(f"{name};{score:.6f}\n")

    with open(os.path.join(save_dir, "relief_scores.csv"), 'w', encoding='utf-8') as f:
        f.write("Feature;Score\n")
        for name, score in zip(feat_names, relief_scores):
            f.write(f"{name};{score:.6f}\n")

    print(f"[✔] Fisher + Relief salvos em {save_dir}")



import numpy as np
from scipy import stats

# ==========================================
# 1. FUNÇÕES AUXILIARES (Matemática Pura)
# ==========================================
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
    psd = fft_vals**2 / (np.sum(fft_vals**2) + 1e-12)
    return -np.sum(psd * np.log(psd + 1e-12))

# ==========================================
# 2. EXTRAÇÃO UNITÁRIA (O bloco de 110 features)
# ==========================================
def _extract_unit_110(Ax, Ay, Az, Gx, Gy, Gz, fs):
    """
    Calcula as 110 features para UM conjunto de 6 eixos (Acc+Gyro).
    Retorna uma lista.
    """
    dt = 1.0 / fs
    channels = [Ax, Ay, Az, Gx, Gy, Gz]
    features = []

    # A. ESTATÍSTICAS (84 features)
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

    # B. CORRELAÇÕES (15 features)
    pairs = [(Ax,Ay), (Ax,Az), (Ay,Az), (Gx,Gy), (Gx,Gz), (Gy,Gz),
             (Ax,Gx), (Ay,Gy), (Az,Gz), (Ax,Gy), (Ax,Gz), (Ay,Gx), (Ay,Gz), (Az,Gx), (Az,Gy)]
    for v1, v2 in pairs:
        features.append(_pairwise_corr(v1, v2))

    # C. FÍSICAS (11 features)
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

# ==========================================
# 3. EXTRAÇÃO 550 (O loop pelos sensores)
# ==========================================

def extract_features_550(window_data, fs=50):
    """
    Percorre os 5 sensores sequencialmente, extrai 110 features de cada
    e concatena horizontalmente. Total: 550 colunas.
    """
    all_features = [] 
    
    # Assumindo: Coluna 0 é ID, dados começam na 1.
    # Cada sensor tem 6 eixos (3 Acc + 3 Gyro).
    start_col = 1 
    
    # Loop 5 vezes (para os 5 sensores)
    for i in range(5):
        end_col = start_col + 6
        
        # Proteção: Se o CSV não tiver colunas suficientes, preenche com zeros
        if end_col > window_data.shape[1]:
            all_features.extend([0.0] * 110)
            continue

        # Pegar os dados brutos APENAS deste sensor
        sensor_block = window_data[:, start_col:end_col]
        
        # Separar canais
        Ax, Ay, Az = sensor_block[:, 0], sensor_block[:, 1], sensor_block[:, 2]
        Gx, Gy, Gz = sensor_block[:, 3], sensor_block[:, 4], sensor_block[:, 5]

        # Calcular as 110 features deste bloco
        feats_110 = _extract_unit_110(Ax, Ay, Az, Gx, Gy, Gz, fs)
        
        # === CONCATENAÇÃO HORIZONTAL ===
        all_features.extend(feats_110)
        
        # Avança para o próximo sensor
        start_col = end_col

    return np.nan_to_num(np.array(all_features))


def get_feature_names_550():
    # 1. Gera nomes base para 110 features
    ch = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
    st = ['Mean', 'Median', 'Std', 'Var', 'RMS', 'AvgDeriv', 'Skew', 'Kurt', 'IQR', 'ZCR', 'MCR', 'DomFreq', 'Energy', 'SpecEnt']
    base_names = [f"{c}_{s}" for c in ch for s in st]
    
    corr_names = ['AxAy', 'AxAz', 'AyAz', 'GxGy', 'GxGz', 'GyGz', 'AxGx', 'AyGy', 'AzGz', 'AxGy', 'AxGz', 'AyGx', 'AyGz', 'AzGx', 'AzGy']
    base_names += [f"Corr_{p}" for p in corr_names]
    
    phys_names = ['AI', 'VI', 'SMA', 'EVA1', 'EVA2', 'CAGH', 'AVH', 'AVG', 'ARATG', 'AAE', 'ARE']
    base_names += phys_names
    
    # 2. Multiplica pelos 5 sensores (S1...S5)
    final_names = []
    sensor_labels = ['S1', 'S2', 'S3', 'S4', 'S5']
    
    for s_label in sensor_labels:
        # Ex: S1_Ax_Mean, S1_Corr_AxAy, ..., S5_ARE
        final_names += [f"{s_label}_{n}" for n in base_names]
        
    return final_names # Total 550