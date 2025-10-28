import os
import numpy as np
from scipy import stats
from scipy.stats import zscore, ttest_ind, f_oneway, kruskal, spearmanr, pearsonr
from scipy.signal import welch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# =========================================
# 1 — Teste de significância e gaussianidade
# =========================================
def statistical_significance(data_array, sensor='acc', body_part_idx=None, body_part_name=None, save_dir="data/statistics"):
    save_dir = os.path.join(save_dir, sensor, body_part_name)
    os.makedirs(save_dir, exist_ok=True)

    sensor_cols = {'acc':[1,2,3],'gyro':[4,5,6],'mag':[7,8,9]}
    cols = sensor_cols[sensor]
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
# 2 — Testes estatísticos avançados
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
    v = np.vstack([x,y,z]).T
    mag = np.linalg.norm(v,axis=1)
    feats = {}
    eps = 1e-12

    def avg_derivative(sig):
        return np.mean(np.abs(np.diff(sig)))

    for name, arr in zip(['x','y','z'], [x,y,z]):
        feats[f'{name}_mean'] = np.mean(arr)
        feats[f'{name}_median'] = np.median(arr)
        feats[f'{name}_std'] = np.std(arr)
        feats[f'{name}_var'] = np.var(arr)
        feats[f'{name}_rms'] = np.sqrt(np.mean(arr**2))
        feats[f'{name}_avg_deriv'] = avg_derivative(arr)
        feats[f'{name}_skew'] = stats.skew(arr)
        feats[f'{name}_kurtosis'] = stats.kurtosis(arr)
        feats[f'{name}_iqr'] = np.percentile(arr,75)-np.percentile(arr,25)
        zc = np.sum((arr[:-1]*arr[1:])<0)
        feats[f'{name}_zcr'] = zc/(len(arr)/fs + eps)

    feats['corr_xy'] = np.corrcoef(x,y)[0,1]
    feats['corr_xz'] = np.corrcoef(x,z)[0,1]
    feats['corr_yz'] = np.corrcoef(y,z)[0,1]
    return feats

def spectral_entropy(x,y,z,fs=50):
    v = np.vstack([x,y,z]).T
    mag = np.linalg.norm(v,axis=1)
    feats = {}

    for name, arr in zip(['x','y','z','mag'], [x,y,z,mag]):
        f, Pxx = welch(arr, fs=fs, nperseg=min(256,len(arr)))
        Pxx += 1e-12
        Pnorm = Pxx/np.sum(Pxx)
        feats[f'{name}_spec_entropy'] = -np.sum(Pnorm*np.log(Pnorm))
    return feats

def physical_features(x,y,z,fs=50,sensor='acc'):
    feats = {}
    T = len(x)
    eps = 1e-12
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
    freqs = np.fft.rfftfreq(T, d=1/fs)
    fft_mag = np.abs(np.fft.rfft(MI))**2
    feats['DF'] = freqs[np.argmax(fft_mag[1:])]
    feats['ENERGY'] = np.sum(fft_mag)/T
    energy_x = np.sum(np.abs(np.fft.rfft(x))**2)/T
    energy_y = np.sum(np.abs(np.fft.rfft(y))**2)/T
    energy_z = np.sum(np.abs(np.fft.rfft(z))**2)/T
    feats['AAE'] = np.mean([energy_x, energy_y, energy_z])
    if sensor=='gyro':
        energy_gx = np.sum(np.abs(np.fft.rfft(x))**2)/T
        energy_gy = np.sum(np.abs(np.fft.rfft(y))**2)/T
        energy_gz = np.sum(np.abs(np.fft.rfft(z))**2)/T
        feats['ARE'] = np.mean([energy_gx,energy_gy,energy_gz])
    return feats

def extract_features(x,y,z,fs=50,vector='acc'):
    feats_temp = temporal_features(x,y,z,fs)
    feats_spec = spectral_entropy(x,y,z,fs)
    feats_phys = physical_features(x,y,z,fs,sensor=vector)
    feats = {**feats_temp, **feats_spec, **feats_phys}
    feat_names = sorted(feats.keys())
    feat_vector = np.array([feats[k] for k in feat_names],dtype=float)
    return feat_vector, feat_names

# =========================================
# 4 — Construção da matriz de features
# =========================================
def build_feature_matrix(data_array, sensor='acc', fs=50, body_part_idx=None, body_part_name=None):
    X, y = [], []
    for person in data_array:
        activity = person[body_part_idx]
        sensor_cols = {'acc':[1,2,3],'gyro':[4,5,6],'mag':[7,8,9]}
        cols = sensor_cols[sensor]
        x, y_sig, z = activity[:,cols[0]], activity[:,cols[1]], activity[:,cols[2]]
        label = int(stats.mode(activity[:,11], keepdims=True)[0])
        feat_vec, feat_names = extract_features(x,y_sig,z,fs,vector=sensor)
        X.append(feat_vec)
        y.append(label)
    X = np.array(X)
    y = np.array(y)
    print(f"[OK] Sensor: {sensor} | Parte: {body_part_name} | labels únicos: {np.unique(y)}")
    return X, y, feat_names

# =========================================
# 5 — PCA
# =========================================
def compute_pca(X, sensor='acc', body_part_name=None, save_dir="data/pca"):
    save_dir = os.path.join(save_dir, sensor, body_part_name)
    os.makedirs(save_dir, exist_ok=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(8,5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_)*100)
    plt.xlabel("Número de componentes")
    plt.ylabel("Variância explicada (%)")
    plt.title(f"PCA ({sensor.upper()} - {body_part_name})")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"{sensor}_{body_part_name}_pca.png"), dpi=200)
    plt.close()
    return X_pca, pca

def pca_variance_analysis(pca, threshold=0.75):
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_feats = np.searchsorted(cumvar, threshold)+1
    print(f"[OK] Número de features para {threshold*100:.0f}% variância: {n_feats}")
    return n_feats

def project_features(X_pca, n_features):
    return X_pca[:,:n_features]

# =========================================
# 6 — Seleção de features (Fisher + ReliefF)
# =========================================
def fisher_score(X, y):
    n_features = X.shape[1]
    scores = np.zeros(n_features)
    classes = np.unique(y)
    overall_mean = np.mean(X, axis=0)
    for i in range(n_features):
        num, den = 0, 0
        for c in classes:
            Xc = X[y==c,i]
            num += len(Xc)*(np.mean(Xc)-overall_mean[i])**2
            den += np.sum((Xc - np.mean(Xc))**2)
        scores[i] = num/(den+1e-12)
    return scores

def relieff(X, y, k=10):
    n_samples, n_features = X.shape
    scores = np.zeros(n_features)
    nn = NearestNeighbors(n_neighbors=k+1).fit(X)
    for i in range(n_samples):
        distances, neighbors = nn.kneighbors([X[i]])
        hit_mask = y[neighbors[0][1:]] == y[i]
        miss_mask = ~hit_mask
        for f in range(n_features):
            diff_hit = np.mean((X[i,f]-X[neighbors[0][1:][hit_mask],f])**2)
            diff_miss = np.mean((X[i,f]-X[neighbors[0][1:][miss_mask],f])**2)
            scores[f] += diff_miss - diff_hit
    scores /= n_samples
    return scores

def top_features(X, y, method='fisher', top_k=10):
    if method=='fisher':
        scores = fisher_score(X,y)
    elif method=='relieff':
        scores = relieff(X,y)
    else:
        raise ValueError("Método inválido")
    idx = np.argsort(scores)[::-1][:top_k]
    return idx, scores[idx]
