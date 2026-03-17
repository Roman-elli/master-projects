# Data
import yfinance as yf

# Ignore warnings (useful for statistical tests)
import warnings
warnings.filterwarnings("ignore")

# Get the prices with a function
def get_stock_data(tickers):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Descarregar os dados
        data = yf.download(tickers, start="2020-01-01", end="2026-01-01")['Close']
    
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