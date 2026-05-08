import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
from tqdm import tqdm
from utils.metrics import estimate_ou_parameters
import config as cfg

def build_global_dataset(valid_pairs_df, spreads_dict, split_date, window, lower_pct, upper_pct, sl_mult):
    """Gera o dataset global calculando ECDF, parâmetros OU e aplicando a Tripla Barreira."""
    global_dataset_list = []

    for _, row in tqdm(valid_pairs_df.iterrows(), total=len(valid_pairs_df)):
        s1, s2 = row['ativo_y'], row['ativo_x']
        pair_name = f"{s1}_{s2}"
        spread = spreads_dict[pair_name].loc[:split_date]
        
        # 1. Feature Engineering
        df_feat = pd.DataFrame(index=spread.index)
        df_feat['spread'] = spread
        df_feat['ecdf'] = spread.rolling(window=window).apply(lambda x: percentileofscore(x, x.iloc[-1]) / 100.0, raw=False)
        df_feat['volatility'] = spread.rolling(window=window).std()
        
        df_feat['theta'], df_feat['mu'], df_feat['half_life'] = np.nan, np.nan, np.nan
        for i in range(window, len(spread)):
            window_data = spread.iloc[i-window:i]
            theta, mu, hl = estimate_ou_parameters(window_data)
            df_feat.iloc[i, df_feat.columns.get_loc('theta')] = theta
            df_feat.iloc[i, df_feat.columns.get_loc('mu')] = mu
            df_feat.iloc[i, df_feat.columns.get_loc('half_life')] = hl
        df_feat = df_feat.dropna()
        
        # 2. Portão 1 (Eventos Base)
        events, pos = [], 0
        for idx, r in df_feat.iterrows():
            p = r['ecdf']
            if p <= lower_pct and pos == 0:
                events.append({'date': idx, 'side': 1, 'mu': r['mu'], 'half_life': r['half_life']})
                pos = 1
            elif p >= upper_pct and pos == 0:
                events.append({'date': idx, 'side': -1, 'mu': r['mu'], 'half_life': r['half_life']})
                pos = -1
            elif 0.40 <= p <= 0.60:
                pos = 0
                
        if not events: continue
        events_df = pd.DataFrame(events).set_index('date')
        
        # 3. Triple Barrier Method
        labels = []
        for idx, event in events_df.iterrows():
            start_loc, side = idx, event['side']
            time_limit_days = int(max(event['half_life'] * cfg.HALF_LIFE_TIME_LIMIT, 5))
            end_loc = start_loc + pd.Timedelta(days=time_limit_days)
            
            path = df_feat.loc[start_loc:end_loc, 'spread']
            if path.empty: continue
                
            entry_price, target_mu, entry_vol = path.iloc[0], df_feat.loc[start_loc, 'mu'], df_feat.loc[start_loc, 'volatility']
            
            if side == 1:
                hit_pt = path[path >= target_mu]
                hit_sl = path[path <= entry_price - (entry_vol * sl_mult)]
            else:
                hit_pt = path[path <= target_mu]
                hit_sl = path[path >= entry_price + (entry_vol * sl_mult)]
            
            time_pt = hit_pt.index[0] if not hit_pt.empty else path.index[-1] + pd.Timedelta(days=1)
            time_sl = hit_sl.index[0] if not hit_sl.empty else path.index[-1] + pd.Timedelta(days=1)
            first_hit = min(time_pt, time_sl, path.index[-1])
            
            label = 1 if first_hit == time_pt else 0
            labels.append({'date': idx, 'target_bin': label})
            
        labels_df = pd.DataFrame(labels).set_index('date')
        
        # Consolidação do Par
        pair_dataset = df_feat.loc[events_df.index].copy()
        pair_dataset['side'] = events_df['side']
        pair_dataset['target'] = labels_df['target_bin']
        pair_dataset['pair_id'] = pair_name
        global_dataset_list.append(pair_dataset.dropna())

    # Agrupar todos os pares e ordenar cronologicamente
    global_meta_dataset = pd.concat(global_dataset_list).sort_index()
    return global_meta_dataset