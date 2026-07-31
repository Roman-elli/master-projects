# Variables, paths, parameters... definition

# config.py
DOWNLOAD_STOCKS = False
TRAIN_MODELS = True

# config.py
# 1. Descarregar todos os dados necessários de uma vez
# (Se a tua função de descarregar dados não baixar de 2008 a 2015, não haverá dados para testar)
DATA_START = "2008-01-01" 
DATA_END = "2026-05-31" 

# 2. Período de Treino (Onde a IA aprende a física do elástico)
TRAIN_START = "2022-01-01" 
TRAIN_END = "2025-12-31"

# 3. Período de Teste Cego / Backcasting (Apanha a Crise de 2008)
TEST_START = "2008-01-01"
TEST_END = "2020-12-31"

HALF_LIFE_TIME_LIMIT = 3
P_VALUE_ADF = 0.05
HURST_VALUE = 0.4
WINDOW = 60
LOWER_PERCENTILE, UPPER_PERCENTILE = 0.03, 0.97
SL_MULTIPLIER = 1.2
LIMIAR_FINAL = 0.7 # 0.6 0.70
CAPITAL_INICIAL = 100000
ALAVANCAGEM_FUNDO = 3  
MAX_ALLOCATION = 0.2

# --- CONFIGURAÇÃO DE TEMPO E RESOLUÇÃO ---
# RESOLUTION = '1h'  # Opções suportadas: '1d' (Diário) ou '1h' (Horário)
# DATABENTO_API_KEY =

# # Assumimos 7 barras de 1 hora por cada dia de pregão (ex: 09:30 às 16:00)
# BARS_PER_DAY = 7 if RESOLUTION == '1h' else 1

# # Parâmetros Dinâmicos (Multiplicam automaticamente consoante a resolução)
# WINDOW = 60 * BARS_PER_DAY            # Memória do elástico (60 dias vs 420 horas)
# MAX_HALF_LIFE = 12 * BARS_PER_DAY     # Filtro Anti-Zombi (12 dias vs 84 horas)
# MIN_HOLD_TIME = 5 * BARS_PER_DAY      # Tempo mínimo de aposta (5 dias vs 35 horas)

# Variável Stock
TICKERS = ["NVDA", "AAPL", "GOOG", "GOOGL", "MSFT", "AMZN", "META", "AVGO", "TSLA",
    "WMT", "LLY", "JPM", "XOM", "V", "JNJ", "MA", "MU", "COST", "ORCL",
    "NFLX", "ABBV", "CVX", "PG", "PLTR", "HD", "BAC", "GE", "KO", "CAT",
    "AMD", "CSCO", "MRK", "RTX", "AMAT", "PM", "LRCX", "UNH", "MS", "GS",
    "WFC", "TMUS", "IBM", "INTC", "MCD", "GEV", "LIN", "PEP", "VZ", "AXP",
    "AMGN", "T", "ABT", "NEE", "C", "KLAC", "TMO", "GILD", "CRM", "DIS",
    "TXN", "TJX", "ANET", "ISRG", "BA", "APH", "SCHW", "APP", "DE", "BLK",
    "ADI", "PFE", "HON", "LMT", "UBER", "UNP", "WELL", "QCOM", "LOW", "ETN",
    "COP", "BKNG", "DHR", "SYK", "PANW", "BX", "SPGI", "PLD", "NEM", "CB",
    "VRTX", "INTU", "ACN", "BMY", "NOW", "PGR", "HCA", "PH", "GLW", "IBKR",
    "MDT", "COF", "SBUX", "CEG", "MCK", "ADBE", "MO", "CMCSA", "CRWD", "CME",
    "SO", "NOC", "BSX", "HWM", "DUK", "CVS", "GD", "TT", "DELL", "WM",
    "EQIX", "SNDK", "ICE", "WDC", "WMB", "FCX", "ADP", "AMT", "MAR", "UPS",
    "FDX", "PWR", "MRSH", "STX", "PNC", "NKE", "SNPS", "JCI", "SHW", "MMM",
    "REGN", "USB", "ABNB", "KKR", "MCO", "CDNS", "ORLY", "ECL", "BK", "EMR",
    "ITW", "CTAS", "CMI", "RCL", "MSI", "CSX", "MNST", "CL", "DASH", "KMI",
    "MDLZ", "SLB", "TDG", "AEP", "CRH", "HOOD", "CVNA", "CI", "RSG", "ROST",
    "WBD", "AON", "EOG", "COR", "HLT", "GM", "LHX", "NSC", "TRV", "PSX",
    "VLO", "MPC", "PCAR", "APO", "SPG", "ELV", "FTNT", "DLR", "AZO", "APD",
    "SRE", "O", "TEL", "BKR", "TFC", "AFL", "VST", "D", "TGT", "AJG",
    "COIN", "ALL", "PSA", "ADSK", "OKE", "FAST", "GWW", "OXY", "MPWR", "AME",
    "CTVA", "NXPI", "XEL", "CAH", "ZTS", "FANG", "EXC", "EA", "TRGP", "EW",
    "NDAQ", "KEYS", "URI", "CARR", "F", "FIX", "CIEN", "IDXX", "ETR", "TER",
    "BDX", "GRMN", "MET", "KR", "CMG", "HSY", "YUM", "DDOG", "AXON", "DHI",
    "WAB", "ROK", "FITB", "AIG", "VTR", "AMP", "PEG", "PYPL", "EBAY", "ODFL",
    "MSCI", "SYY", "ED", "PCG", "TKO", "CBRE", "XYZ", "NUE", "TTWO", "DAL",
    "CCI", "EQT", "LYV", "KDP", "HIG", "WEC", "MLM", "WDAY", "LVS", "ROP",
    "CCL", "RMD", "TPL", "VMC", "MCHP", "CPRT", "KVUE", "ACGL", "STT", "PAYX",
    "EL", "IR", "KMB", "OTIS", "ADM", "NRG", "PRU", "GEHC", "A", "HBAN",
    "EME", "DG", "IRM", "FISV", "VICI", "CHTR", "EXR", "DTE", "AEE", "FICO",
    "MTB", "ATO", "TDY", "CTSH", "TPR", "CBOE", "XYL", "HAL", "UAL", "WAT",
    "RJF", "FE", "IQV", "ULTA", "PPL", "EXPE", "DOV", "CNP", "HPE", "KHC",
    "VRSK", "BIIB", "ES", "EIX", "WTW", "DVN", "ROL", "JBL", "TSCO", "STLD",
    "DXCM", "STZ", "FIS", "AWK", "NTRS", "CINF", "HUBB", "WRB", "EXE", "OMC",
    "MTD", "CFG", "AVB", "DOW", "LEN", "ARES", "CHD", "PHM", "Q", "PPG",
    "FOXA", "FOX", "EFX", "ON", "DRI", "CMS", "BRO", "DLTR", "EQR", "BG",
    "RF", "CTRA", "SYF", "VLTO", "GIS", "L", "WSM", "CPAY", "SW", "NI",
    "LH", "DGX", "VRSN", "LDOS", "BR", "STE", "MRNA", "KEY", "FSLR", "LYB",
    "LUV", "RL", "HUM", "CHRW", "TSN", "IP", "GPN", "SBAC", "JBHT", "LULU",
    "PKG", "ALB", "PFG", "AMCR", "CSGP", "TROW", "SNA", "NTAP", "INCY", "SMCI",
    "PTC", "NVR", "EXPD", "EVRG", "DD", "IFF", "LNT", "ZBH", "CNC", "LII",
    "WY", "FTV", "HPQ", "MKC", "CF", "HOLX", "WST", "PODD", "BALL", "ESS",
    "HII", "VTRS", "FFIV", "TRMB", "TXT", "INVH", "KIM", "J", "APTV", "CDW",
    "TYL", "MAA", "NDSN", "AKAM", "GPC", "DECK", "PNR", "IEX", "COO", "CPT",
    "REG", "CLX", "NWS", "NWSA", "BBY", "DPZ", "AVY", "HAS", "HST", "EG",
    "GEN", "TTD", "BEN", "MAS", "HRL", "ALLE", "DOC", "GNRC", "JKHY", "PNW",
    "UDR", "ALGN", "GDDY", "SOLV", "SJM", "SWK", "PSKY", "UHS", "ERIE",
    "APA", "GL", "IT", "AIZ", "WYNN", "IVZ", "ZBRA", "BLDR", "AES", "DVA",
    "RVTY", "AOS", "FRT", "NCLH", "TAP", "BAX", "MGM", "ARE", "HSIC", "CAG",
    "BXP", "TECH", "SWKS", "CRL", "FDS", "EPAM", "POOL", "CPB", "MOH", "MTCH",
    "PAYC", "LW", "MOS"]