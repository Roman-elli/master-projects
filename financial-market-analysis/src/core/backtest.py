import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
from tqdm import tqdm
import random
from utils.metrics import estimate_ou_parameters
import config as cfg

def run_oos_backtest(valid_pairs_df, spreads_dict, models_dict, mode, test_start, test_end, 
                     window, lower_pct, upper_pct, sl_mult, limiar_final, alavancagem_fundo):
    """
    Simula o passar do tempo num ambiente Out-of-Sample (OOS).
    Modos de Inteligência suportados:
    - 'global': Usa um modelo único treinado com todos os dados.
    - 'specific': Cada par usa o seu modelo exclusivo.
    - 'cross_testing': Cada par é operado por um modelo treinado num par DIFERENTE (Transfer Learning).
    """
    trade_log = []

    # Iterar sobre todos os pares válidos
    for _, row in tqdm(valid_pairs_df.iterrows(), total=len(valid_pairs_df), desc=f"OOS Backtest ({mode.upper()})"):
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
            
        # Isolar SÓ o período de Teste
        df_test = df_test.loc[test_start:test_end].dropna()
        if df_test.empty: continue
            
        # 2. SELEÇÃO DO CÉREBRO (Agente) PARA ESTE PAR
        agent_name = "Global" # Nome default para logs
        
        if mode == 'global':
            agent = models_dict['global']
            
        elif mode == 'specific':
            if pair_name in models_dict:
                agent = models_dict[pair_name]
                agent_name = pair_name
            else:
                continue # Ignorar se não tem modelo próprio
                
        elif mode == 'cross_testing':
            # 1. Criar uma lista estática de todos os modelos específicos disponíveis
            chaves_modelos = [k for k in models_dict.keys() if k != 'global']
            
            if pair_name not in chaves_modelos or len(chaves_modelos) < 2:
                continue
                
            # 2. Descobrir em que posição da "roda" este par está
            meu_index = chaves_modelos.index(pair_name)
            
            # 3. Passar o cérebro para a "direita" (Shift + 1). 
            # O último da lista usa o cérebro do primeiro (usando o resto da divisão %)
            index_vizinho = (meu_index + 1) % len(chaves_modelos)
            
            modelo_vizinho = chaves_modelos[index_vizinho]
            
            agent = models_dict[modelo_vizinho]
            agent_name = modelo_vizinho # Registar quem foi o intruso
            
        pos = 0 # 0 = Sem posição aberta
        entry_data = {}
        
        # 3. Simulação Temporal
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
                    
                    # Registar trade finalizado (adicionada a coluna 'agent_used' para saberes quem operou no Cross-Test)
                    trade_log.append({
                        'pair': pair_name, 'agent_used': agent_name, 
                        'entry_date': entry_data['date'], 'exit_date': idx,
                        'side': pos, 'prob_ml': entry_data['prob'], 'kelly_fraction': entry_data['kelly'],
                        'duration': days_held, 'exit_reason': motivo, 'pnl_spread_pts': ret
                    })
                    pos = 0 # Posição encerrada
                    continue
                    
            # --- LÓGICA DE ENTRADA (Portões Hierárquicos) ---
            if pos == 0:
                side_signal = 0
                if p <= lower_pct: side_signal = 1
                elif p >= upper_pct: side_signal = -1

                # Portão 1 Extra: Filtro "Anti-Zombi"
                if side_signal != 0 and r['half_life'] > 12:
                    side_signal = 0
                
                if side_signal != 0:
                    X_atual = pd.DataFrame([{
                        'ecdf': r['ecdf'], 'theta': r['theta'], 'half_life': r['half_life'], 
                        'volatility': r['volatility'], 'side': side_signal
                    }])
                    
                    # Portão 2: O Agente aprova? (Pode ser Global, Específico ou Estrangeiro)
                    prob_sucesso = agent.predict_proba(X_atual)[0][1]
                    
                    if prob_sucesso >= limiar_final:
                        pos = side_signal
                        
                        # Kelly Dinâmico com Alavancagem
                        kelly_f = prob_sucesso - (1 - prob_sucesso)
                        kelly_f_alavancado = kelly_f * alavancagem_fundo
                        kelly_f_final = max(0.01, min(kelly_f_alavancado, cfg.MAX_ALLOCATION))
                        
                        tp_price = r['mu']
                        sl_price = r['spread'] - (r['volatility'] * sl_mult) if pos == 1 else r['spread'] + (r['volatility'] * sl_mult)
                        
                        entry_data = {
                            'date': idx, 'price': r['spread'], 'tp': tp_price, 'sl': sl_price,
                            'time_limit': int(max(r['half_life'] * 2, 5)),
                            'prob': prob_sucesso, 'kelly': kelly_f_final
                        }

    return pd.DataFrame(trade_log)