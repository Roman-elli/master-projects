import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
from tqdm import tqdm
from utils.metrics import estimate_ou_parameters
import config as cfg

def run_oos_backtest(valid_pairs_df, spreads_dict, meta_agent, split_date, 
                     window, lower_pct, upper_pct, sl_mult, limiar_final, alavancagem_fundo):
    """
    Simula o passar do tempo num ambiente Out-of-Sample (OOS).
    Para cada dia, calcula a ECDF, valida as regras de entrada (Portões 1 e 2) 
    e gere as posições ativas usando o Método da Tripla Barreira.
    """
    trade_log = []

    # Iterar sobre todos os pares válidos
    for _, row in tqdm(valid_pairs_df.iterrows(), total=len(valid_pairs_df), desc="OOS Backtest"):
        s1, s2 = row['ativo_y'], row['ativo_x']
        pair_name = f"{s1}_{s2}"
        
        # Recuperar spread total para podermos olhar "para trás" no dia 1 do OOS
        spread_full = spreads_dict[pair_name]
        
        # 1. Feature Engineering Dia-a-Dia
        df_test = pd.DataFrame(index=spread_full.index)
        df_test['spread'] = spread_full
        df_test['ecdf'] = spread_full.rolling(window=window).apply(lambda x: percentileofscore(x, x.iloc[-1]) / 100.0, raw=False)
        df_test['volatility'] = spread_full.rolling(window=window).std()
        
        df_test['theta'], df_test['mu'], df_test['half_life'] = np.nan, np.nan, np.nan
        for i in range(window, len(spread_full)):
            window_data = spread_full.iloc[i-window:i]
            theta, mu, hl = estimate_ou_parameters(window_data)
            df_test.iloc[i, df_test.columns.get_loc('theta')] = theta
            df_test.iloc[i, df_test.columns.get_loc('mu')] = mu
            df_test.iloc[i, df_test.columns.get_loc('half_life')] = hl
            
        # Isolar SÓ o período de Trading (O Out-of-Sample), ignorando a formação
        df_test = df_test.loc[split_date:].dropna()
        if df_test.empty: continue
            
        pos = 0 # 0 = Sem posição aberta
        entry_data = {}
        
        # 2. Simulação Temporal
        for idx, r in df_test.iterrows():
            p = r['ecdf']
            
            # --- LÓGICA DE SAÍDA (Agente Executor / Tripla Barreira) ---
            if pos != 0:
                current_price = r['spread']
                days_held = (idx - entry_data['date']).days
                
                # Verificar se bateu nalguma barreira
                hit_tp = (pos == 1 and current_price >= entry_data['tp']) or (pos == -1 and current_price <= entry_data['tp'])
                hit_sl = (pos == 1 and current_price <= entry_data['sl']) or (pos == -1 and current_price >= entry_data['sl'])
                hit_ts = days_held >= entry_data['time_limit']
                
                if hit_tp or hit_sl or hit_ts:
                    ret = (current_price - entry_data['price']) * pos
                    motivo = "Take-Profit" if hit_tp else ("Stop-Loss" if hit_sl else "Time-Stop")
                    
                    # Registar trade finalizado
                    trade_log.append({
                        'pair': pair_name, 'entry_date': entry_data['date'], 'exit_date': idx,
                        'side': pos, 'prob_ml': entry_data['prob'], 'kelly_fraction': entry_data['kelly'],
                        'duration': days_held, 'exit_reason': motivo, 'pnl_spread_pts': ret
                    })
                    pos = 0 # Posição encerrada, bot livre novamente
                    continue
                    
            # --- LÓGICA DE ENTRADA (Portões Hierárquicos) ---
            if pos == 0:
                side_signal = 0
                if p <= lower_pct: side_signal = 1
                elif p >= upper_pct: side_signal = -1

                # Portão 1 Extra: Filtro "Anti-Zombi". Abortar se demorar mais de 12 dias a reverter.
                if side_signal != 0 and r['half_life'] > 12:
                    side_signal = 0
                
                if side_signal != 0:
                    # Preparar os dados exatos para o Random Forest prever
                    X_atual = pd.DataFrame([{
                        'ecdf': r['ecdf'], 'theta': r['theta'], 'half_life': r['half_life'], 
                        'volatility': r['volatility'], 'side': side_signal
                    }])
                    
                    # Portão 2: O Meta-Agente aprova?
                    prob_sucesso = meta_agent.predict_proba(X_atual)[0][1]
                    
                    if prob_sucesso >= limiar_final:
                        pos = side_signal
                        
                        # Kelly Dinâmico com Alavancagem e limites de segurança
                        kelly_f = prob_sucesso - (1 - prob_sucesso)
                        kelly_f_alavancado = kelly_f * alavancagem_fundo
                        kelly_f_final = max(0.01, min(kelly_f_alavancado, cfg.MAX_ALLOCATION))
                        
                        # Construir Barreiras Muro (Take Profit = Média, Stop Loss = Volatilidade extrema)
                        tp_price = r['mu']
                        sl_price = r['spread'] - (r['volatility'] * sl_mult) if pos == 1 else r['spread'] + (r['volatility'] * sl_mult)
                        
                        entry_data = {
                            'date': idx, 'price': r['spread'], 'tp': tp_price, 'sl': sl_price,
                            'time_limit': int(max(r['half_life'] * 2, 5)),
                            'prob': prob_sucesso, 'kelly': kelly_f_final
                        }

    return pd.DataFrame(trade_log)