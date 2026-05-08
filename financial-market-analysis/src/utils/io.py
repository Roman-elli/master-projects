# Data
import yfinance as yf
import matplotlib.pyplot as plt
import config as cfg

# Ignore warnings (useful for statistical tests)
import warnings
warnings.filterwarnings("ignore")

# Get the prices with a function
def get_stock_data(tickers):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Descarregar os dados
    data = yf.download(tickers, start=cfg.START_DATE, end=cfg.END_DATE)['Close']    
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