import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score

def get_tls_beta(x, y):
    """Calcula o Beta usando Total Least Squares (ortogonal e simétrico)."""
    pca = PCA(n_components=1)
    data = np.vstack((x, y)).T
    pca.fit(data)
    return pca.components_[0][1] / pca.components_[0][0]

def calculate_hurst(ts):
    """Calcula o Expoente de Hurst para avaliar a velocidade de reversão à média."""
    lags = range(2, 20)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

def estimate_ou_parameters(spread_series):
    """Estima os parâmetros do Processo Ornstein-Uhlenbeck (Velocidade, Média, Meia-vida)."""
    y = spread_series.diff().dropna()
    x = spread_series.shift(1).dropna()
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    theta = -model.params.iloc[1]
    mu = model.params.iloc[0] / theta if theta > 0 else spread_series.mean()
    halflife = np.log(2) / theta if theta > 0 else 20
    return theta, mu, halflife

def tune_threshold(probs_sucesso, y_real, thresholds=np.arange(0.5, 0.9, 0.05)):
    """
    Analisa o impacto de diferentes limiares de confiança no Win-Rate e Recall.
    Retorna um DataFrame com as métricas calculadas.
    """
    precisions, recalls, trades_aprovados = [], [], []

    for t in thresholds:
        # Se a probabilidade for maior que o limiar, o Portão 2 abre (Previsão = 1)
        y_pred_custom = (probs_sucesso >= t).astype(int)
        
        # Prevenir divisão por zero se o limiar for tão alto que não aprova nada
        if y_pred_custom.sum() == 0:
            precisions.append(0)
            recalls.append(0)
        else:
            precisions.append(precision_score(y_real, y_pred_custom))
            recalls.append(recall_score(y_real, y_pred_custom))
            
        trades_aprovados.append(y_pred_custom.sum())

    # Construir Tabela
    df_thresholds = pd.DataFrame({
        'Limiar (%)': [f"{t*100:.0f}%" for t in thresholds],
        'Trades Aprovados': trades_aprovados,
        'Precision (Win-Rate)': [f"{p*100:.1f}%" for p in precisions],
        'Recall (Oportunidades)': [f"{r*100:.1f}%" for r in recalls]
    })
    
    return df_thresholds