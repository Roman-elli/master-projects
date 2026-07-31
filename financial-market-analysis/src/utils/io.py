# Data
import yfinance as yf
import matplotlib.pyplot as plt
import config as cfg
import os
import pandas as pd

# Ignore warnings (useful for statistical tests)
import warnings
warnings.filterwarnings("ignore")

# Get the prices with a function
def get_stock_data(tickers):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Descarregar os dados
    data = yf.download(tickers, start=cfg.DATA_START, end=cfg.DATA_END)['Close']    
    print(f"-> Dados descarregados: {data.shape[0]} dias, {data.shape[1]} ações.")
    
    # 1. Definir o limite mínimo de dados válidos que uma ação deve ter (ex: 95% dos dias)
    min_valid_days = int(len(data) * 0.95)
    
    # 2. Remover AÇÕES (axis=1) que não cumpram este limite (as que fizeram IPO recentemente)
    data_clean = data.dropna(axis=1, thresh=min_valid_days)
    acoes_removidas = data.shape[1] - data_clean.shape[1]
    print(f"-> Ações recentes/com falhas removidas: {acoes_removidas}")
    
    # 3. Agora sim, removemos os poucos DIAS (axis=0) que sobraram com NaNs 
    data_clean = data_clean.dropna(axis=0)
    
    print(f"-> DADOS FINAIS PARA O BOT: {data_clean.shape[0]} dias, {data_clean.shape[1]} ações.")
    
    return data_clean

# def get_stock_data(tickers):
#     """
#     Busca dados usando Databento com sistema de Cache Local em Parquet.
#     Lê a resolução ('1d' ou '1h') e as datas diretamente do config.py.
#     """
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
        
#         # 1. Criar uma pasta segura para os dados
#         data_dir = "../data"
#         os.makedirs(data_dir, exist_ok=True)
        
#         # Ler a resolução do config
#         res = cfg.RESOLUTION 
        
#         # 2. Definir o nome exato do ficheiro cache baseado nas datas e resolução
#         cache_file = f"{data_dir}/market_data_{res}_{cfg.TRAIN_START}_to_{cfg.TEST_END}.parquet"
        
#         # 3. O "Guarda-Redes": Verificar se já temos os dados no disco
#         if os.path.exists(cache_file):
#             print(f"-> A carregar dados do Cache Local SOTA ({res})... Custo: $0.00!")
#             data_clean = pd.read_parquet(cache_file)
#             print(f"-> DADOS FINAIS PARA O BOT: {data_clean.shape[0]} barras, {data_clean.shape[1]} ações.")
#             return data_clean
            
#         # 4. Se não temos o cache, vamos ao Databento descarregar
#         print(f"-> Cache não encontrado. A descarregar dados ({res}) pelo Databento...")
        
#         # Inicia o cliente (Nota: Adiciona DATABENTO_API_KEY="tua_chave" no teu config.py)
#         client = db.Historical(cfg.DATABENTO_API_KEY)
#         schema_db = "ohlcv-1d" if res == '1d' else "ohlcv-1h"
        
#         # Faz o pedido dos dados à API
#         data = client.timeseries.get_range(
#             dataset="XNAS.ITCH",  # Ajusta para o dataset que usares (ex: XNAS.ITCH para Nasdaq)
#             symbols=tickers,
#             stype_in="raw_symbol",
#             start=cfg.TRAIN_START,
#             end=cfg.TEST_END,
#             schema=schema_db
#         )
        
#         # Converter a resposta para DataFrame do Pandas e isolar a coluna 'close'
#         df = data.to_df()
        
#         # Dependendo do schema, o nome da coluna pode vir minúsculo no Databento
#         close_col = 'close' if 'close' in df.columns else 'Close'
#         data_raw = df.pivot(columns='symbol', values=close_col)
        
#         print(f"-> Dados descarregados: {data_raw.shape[0]} barras, {data_raw.shape[1]} ações.")
        
#         # 5. A TUA LÓGICA DE LIMPEZA SOTA
#         # Definir o limite mínimo de dados válidos que uma ação deve ter (ex: 95% do tempo)
#         min_valid_days = int(len(data_raw) * 0.95)
        
#         # Remover AÇÕES (axis=1) que não cumpram este limite (as que fizeram IPO recentemente)
#         data_clean = data_raw.dropna(axis=1, thresh=min_valid_days)
#         acoes_removidas = data_raw.shape[1] - data_clean.shape[1]
#         print(f"-> Ações recentes/com falhas removidas: {acoes_removidas}")
        
#         # Agora sim, removemos os poucos DIAS/HORAS (axis=0) que sobraram com NaNs 
#         data_clean = data_clean.dropna(axis=0)
        
#         print(f"-> DADOS FINAIS PARA O BOT: {data_clean.shape[0]} barras, {data_clean.shape[1]} ações.")
        
#         # 6. GUARDAR NO DISCO (Cache Local)
#         data_clean.to_parquet(cache_file)
#         print(f"-> Base de dados blindada e guardada no disco em: {cache_file}")
        
#         return data_clean


def generate_tearsheet(trades_df, capital_inicial=100000):
    """
    Calcula as métricas de performance da estratégia OOS,
    incluindo Win-Rate, R/R, Retorno Total e Max Drawdown.
    Gera o gráfico da Curva de Capital (Equity Curve) e Drawdown.
    """
    if trades_df.empty:
        print("Sem trades para analisar. O mercado de 2024-2026 pode não ter apresentado anomalias extremas ou o modelo foi demasiado conservador.")
        return

    # 1. Calcular o Lucro Financeiro em % (Pontos ganhos * Fração de Kelly)
    trades_df['pnl_percent'] = trades_df['pnl_spread_pts'] * trades_df['kelly_fraction']
    
    # 2. Estatísticas Base
    total_trades = len(trades_df)
    win_rate = len(trades_df[trades_df['exit_reason'] == 'Take-Profit']) / total_trades
    sl_rate = len(trades_df[trades_df['exit_reason'] == 'Stop-Loss']) / total_trades
    ts_rate = len(trades_df[trades_df['exit_reason'] == 'Time-Stop']) / total_trades
    
    lucro_medio = trades_df[trades_df['pnl_percent'] > 0]['pnl_percent'].mean() * 100
    perda_media = trades_df[trades_df['pnl_percent'] < 0]['pnl_percent'].mean() * 100
    
    print("=== PERFORMANCE OUT-OF-SAMPLE (2024-2026) ===")
    print(f"-> Total Trades: {total_trades}")
    print(f"-> Win-Rate: {win_rate*100:.2f}% (Trades que bateram no mu)")
    print(f"-> Stop-Loss Rate: {sl_rate*100:.2f}%")
    print(f"-> Time-Stop Rate: {ts_rate*100:.2f}%")
    print(f"-> Lucro Médio p/ Trade Vencedor: {lucro_medio:.2f}% do capital")
    print(f"-> Perda Média p/ Trade Perdedor: {perda_media:.2f}% do capital")
    
    if abs(perda_media) > 0:
        reward_risk_ratio = lucro_medio / abs(perda_media)
        print(f"-> Risco/Recompensa Realizado: {reward_risk_ratio:.2f}")

    # 3. Construir a Curva de Capital (Equity Curve)
    trades_df = trades_df.sort_values(by='exit_date')
    trades_df['capital_acumulado'] = (1 + trades_df['pnl_percent']).cumprod() * capital_inicial
    
    # Drawdown (A maior queda a partir do topo histórico)
    trades_df['peak'] = trades_df['capital_acumulado'].cummax()
    trades_df['drawdown'] = (trades_df['capital_acumulado'] - trades_df['peak']) / trades_df['peak']
    max_drawdown = trades_df['drawdown'].min() * 100
    
    retorno_total = ((trades_df['capital_acumulado'].iloc[-1] / capital_inicial) - 1) * 100
    print(f"-> Retorno Total da Estratégia: {retorno_total:.2f}%")
    print(f"-> Max Drawdown (A maior dor financeira): {max_drawdown:.2f}%")

    # 4. Gráfico SOTA
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    ax1.plot(trades_df['exit_date'], trades_df['capital_acumulado'], color='blue', linewidth=2)
    ax1.axhline(y=capital_inicial, color='r', linestyle='--')
    ax1.set_title(f'Evolução do Capital (${capital_inicial:,.0f} Iniciais) | Retorno: {retorno_total:.2f}%')
    ax1.set_ylabel('Capital em $')
    ax1.grid(True, alpha=0.3)
    
    ax2.fill_between(trades_df['exit_date'], trades_df['drawdown'] * 100, 0, color='red', alpha=0.5)
    ax2.set_title(f'Drawdown Histórico | Máximo: {max_drawdown:.2f}%')
    ax2.set_ylabel('Queda %')
    ax2.set_xlabel('Data')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def setup_results_dir(cfg):
    """Cria uma pasta única baseada nas datas e timestamp para armazenar os outputs."""
    dir_name = f"../results/Train_{cfg.TRAIN_START}_to_{cfg.TRAIN_END}__Test_{cfg.TEST_START}_to_{cfg.TEST_END}"
    os.makedirs(dir_name, exist_ok=True)
    print(f"-> Diretório de resultados criado: {dir_name}")
    return dir_name

def compare_strategies(trades_global, trades_specific, trades_cross, capital_inicial, save_dir):
    """
    Gera gráficos separados comparando a Curva de Capital e a precisão das 3 abordagens (Global, Específica e Cross)
    e exporta um relatório completo de métricas para CSV e TXT.
    """
    if trades_global.empty and trades_specific.empty and trades_cross.empty:
        print("Sem trades para comparar e gerar métricas.")
        return

    # --- 1. FUNÇÃO INTERNA PARA CALCULAR MÉTRICAS EXATAS ---
    def calc_metrics(df):
        if df is None or df.empty:
            return {"Total Trades": 0, "Win-Rate (%)": 0.0, "Stop-Loss Rate (%)": 0.0,
                    "Time-Stop Rate (%)": 0.0, "Lucro Médio (%)": 0.0, "Perda Média (%)": 0.0,
                    "Risco/Recompensa": 0.0, "Retorno Total (%)": 0.0, "Max Drawdown (%)": 0.0}

        df = df.copy()
        df['pnl_percent'] = df['pnl_spread_pts'] * df['kelly_fraction']
        total_trades = len(df)
        
        win_rate = len(df[df['exit_reason'] == 'Take-Profit']) / total_trades * 100
        sl_rate = len(df[df['exit_reason'] == 'Stop-Loss']) / total_trades * 100
        ts_rate = len(df[df['exit_reason'] == 'Time-Stop']) / total_trades * 100

        lucro_medio = df[df['pnl_percent'] > 0]['pnl_percent'].mean() * 100
        if pd.isna(lucro_medio): lucro_medio = 0
        perda_media = df[df['pnl_percent'] < 0]['pnl_percent'].mean() * 100
        if pd.isna(perda_media): perda_media = 0

        rr_ratio = lucro_medio / abs(perda_media) if perda_media != 0 else 0

        df = df.sort_values(by='exit_date')
        capital_acumulado = (1 + df['pnl_percent']).cumprod() * capital_inicial
        peak = capital_acumulado.cummax()
        drawdown = (capital_acumulado - peak) / peak
        max_drawdown = drawdown.min() * 100
        retorno_total = ((capital_acumulado.iloc[-1] / capital_inicial) - 1) * 100

        return {
            "Total Trades": total_trades,
            "Win-Rate (%)": round(win_rate, 2),
            "Stop-Loss Rate (%)": round(sl_rate, 2),
            "Time-Stop Rate (%)": round(ts_rate, 2),
            "Lucro Médio (%)": round(lucro_medio, 2),
            "Perda Média (%)": round(perda_media, 2),
            "Risco/Recompensa": round(rr_ratio, 2),
            "Retorno Total (%)": round(retorno_total, 2),
            "Max Drawdown (%)": round(max_drawdown, 2)
        }

    # --- 2. CALCULAR MÉTRICAS E EXPORTAR (CSV e TXT) ---
    metrics_global = calc_metrics(trades_global)
    metrics_specific = calc_metrics(trades_specific)
    metrics_cross = calc_metrics(trades_cross)

    # Criar um DataFrame de comparação
    nomes_modelos = ['Modelo Global', 'Modelos Específicos', 'Cross-Testing']
    df_metrics = pd.DataFrame([metrics_global, metrics_specific, metrics_cross], index=nomes_modelos).T
    
    # Mostrar no ecrã 
    print("\n=== RESUMO DE PERFORMANCE (OUT-OF-SAMPLE) ===")
    print(df_metrics)

    # Guardar em CSV
    df_metrics.to_csv(f"{save_dir}/performance_metrics.csv")
    
    # Guardar num ficheiro de Texto estilo "Relatório Clássico"
    with open(f"{save_dir}/performance_report.txt", "w", encoding='utf-8') as f:
        f.write("=== PERFORMANCE OUT-OF-SAMPLE ===\n\n")
        for model_name, metrics in zip(nomes_modelos, [metrics_global, metrics_specific, metrics_cross]):
            f.write(f"--- {model_name.upper()} ---\n")
            for key, value in metrics.items():
                f.write(f"-> {key}: {value}\n")
            f.write("\n")
            
    print(f"-> Relatórios de métricas salvos em: {save_dir}/performance_metrics.csv (e .txt)")

    # --- 3. GRÁFICOS SOTA (Evolução de Capital e Barras SEPARADOS) ---
    def process_equity(df):
        if df is None or df.empty: return pd.Series(dtype=float)
        df = df.sort_values(by='exit_date')
        eq = (1 + (df['pnl_spread_pts'] * df['kelly_fraction'])).cumprod() * capital_inicial
        eq.index = df['exit_date']
        return eq.resample('D').last().ffill() 

    eq_global = process_equity(trades_global)
    eq_specific = process_equity(trades_specific)
    eq_cross = process_equity(trades_cross)

    df_compare = pd.DataFrame({'Global': eq_global, 'Específico': eq_specific, 'Cross': eq_cross}).ffill().fillna(capital_inicial)

    # ---------------------------------------------------------
    # GRÁFICO 1: EVOLUÇÃO DO CAPITAL
    # ---------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df_compare.index, df_compare['Global'], label=f"Global ({metrics_global['Retorno Total (%)']}%)", color='#1f77b4', linewidth=2)
    ax1.plot(df_compare.index, df_compare['Específico'], label=f"Específico ({metrics_specific['Retorno Total (%)']}%)", color='#ff7f0e', linewidth=2)
    ax1.plot(df_compare.index, df_compare['Cross'], label=f"Cross-Testing ({metrics_cross['Retorno Total (%)']}%)", color='#2ca02c', linewidth=2)
    ax1.axhline(y=capital_inicial, color='gray', linestyle='--', alpha=0.5)
    
    ax1.set_title('Evolução de Capital', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Capital ($)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    fig1.tight_layout()
    fig1.savefig(f"{save_dir}/comparison_equity.png", dpi=300, transparent=False)
    plt.show()

    # ---------------------------------------------------------
    # GRÁFICO 2: PROPORÇÃO DE WIN-RATE
    # ---------------------------------------------------------
    metricas_barras = {
        'Modelo': ['Global', 'Específico', 'Cross'],
        'Trades': [metrics_global['Total Trades'], metrics_specific['Total Trades'], metrics_cross['Total Trades']],
        'WinRate': [metrics_global['Win-Rate (%)'], metrics_specific['Win-Rate (%)'], metrics_cross['Win-Rate (%)']]
    }
    
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.bar(['Global', 'Específico', 'Cross'], metricas_barras['WinRate'], color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
    for i, v in enumerate(metricas_barras['WinRate']):
        ax2.text(i, v + 1, f"{v:.1f}%\n({metricas_barras['Trades'][i]} trades)", ha='center', fontweight='bold')
    
    ax2.set_title('Precisão (Win-Rate) e Volume', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Win-Rate (%)')
    # Ajusta o eixo Y para não cortar o texto do topo da barra
    ax2.set_ylim(0, max(metricas_barras['WinRate']) + 15 if metricas_barras['WinRate'] else 100)

    fig2.tight_layout()
    fig2.savefig(f"{save_dir}/comparison_winrate.png", dpi=300, transparent=False)
    plt.show()
    
    print(f"-> Gráficos guardados separadamente em: {save_dir}/comparison_equity.png e comparison_winrate.png")

def generate_trade_plots(trades_df, spreads_dict, save_dir, model_name):
    """
    Gera um gráfico com a linha do spread, sobrepondo os sinais de entrada 
    (Long/Short) e de saída para cada par operado, guardando tudo numa subpasta.
    """
    if trades_df.empty:
        return
        
    # Criar uma subpasta dentro dos resultados para não gerar desorganização
    plots_dir = f"{save_dir}/graficos_trades_{model_name}"
    os.makedirs(plots_dir, exist_ok=True)
    
    pares_operados = trades_df['pair'].unique()
    
    for pair in pares_operados:
        pair_trades = trades_df[trades_df['pair'] == pair]
        spread = spreads_dict[pair]
        
        # Limitar o gráfico à janela temporal onde ocorreram trades (+/- 30 dias para margem)
        start_dt = pair_trades['entry_date'].min() - pd.Timedelta(days=30)
        end_dt = pair_trades['exit_date'].max() + pd.Timedelta(days=30)
        spread_cut = spread.loc[start_dt:end_dt]
        
        plt.figure(figsize=(15, 7))
        # Plot da Linha do Spread
        plt.plot(spread_cut.index, spread_cut.values, label=f'Spread {pair}', color='#5c5c5c', alpha=0.6, linewidth=1.5)
        
        # 1. ENTRADAS LONG (Verde, Triângulo para cima)
        longs = pair_trades[pair_trades['side'] == 1]
        if not longs.empty:
            plt.scatter(longs['entry_date'], spread.loc[longs['entry_date']].values,
                        marker='^', color='green', s=150, label='Long (Compra Spread)', zorder=5)
                        
        # 2. ENTRADAS SHORT (Vermelho, Triângulo para baixo)
        shorts = pair_trades[pair_trades['side'] == -1]
        if not shorts.empty:
            plt.scatter(shorts['entry_date'], spread.loc[shorts['entry_date']].values,
                        marker='v', color='red', s=150, label='Short (Venda Spread)', zorder=5)
                        
        # 3. SAÍDAS COM LUCRO (Take-Profit: Estrela Dourada)
        tps = pair_trades[pair_trades['exit_reason'] == 'Take-Profit']
        if not tps.empty:
            plt.scatter(tps['exit_date'], spread.loc[tps['exit_date']].values,
                        marker='*', color='gold', edgecolors='black', s=250, label='Take-Profit', zorder=6)
                        
        # 4. SAÍDAS COM PREJUÍZO (Stop-Loss / Time-Stop: X Preto)
        fails = pair_trades[pair_trades['exit_reason'].isin(['Stop-Loss', 'Time-Stop'])]
        if not fails.empty:
            plt.scatter(fails['exit_date'], spread.loc[fails['exit_date']].values,
                        marker='X', color='black', s=100, label='Loss / Time-Stop', zorder=6)
                        
        # 5. LINHAS A LIGAR A ENTRADA À SAÍDA (Opcional, mas muito SOTA)
        for _, row in pair_trades.iterrows():
            y_in = spread.loc[row['entry_date']]
            y_out = spread.loc[row['exit_date']]
            cor_linha = 'green' if row['pnl_spread_pts'] > 0 else 'red'
            plt.plot([row['entry_date'], row['exit_date']], [y_in, y_out], color=cor_linha, linestyle='--', alpha=0.4)
            
        plt.title(f'Tracking de Sinais ({model_name}): {pair}', fontsize=14, fontweight='bold')
        plt.xlabel('Data')
        plt.ylabel('Valor do Spread (Z-Score / Resíduo)')
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        # Guardar gráfico e fechar figura da memória
        plt.savefig(f"{plots_dir}/{pair}.png", dpi=150)
        plt.close() 
        
    print(f"-> {len(pares_operados)} gráficos de pares detalhados guardados na pasta: {plots_dir}/")