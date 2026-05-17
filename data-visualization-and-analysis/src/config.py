# Path variables
RAW_COURSES_PATH = 'assets/raw_courses_data.csv'
RAW_INSTITUTION_PATH = 'assets/raw_institution_data.csv'
CLEAN_DATA_PATH = 'data/clean_data.csv'
MERGED_DATA_PATH = 'data/merged_data.csv'

# Extraction / cleaning variables

colunas_roi = [
    # Identificação base
    'UNITID','INSTNM', 'CIPCODE', 'CIPDESC', 'CREDLEV', 'IPEDSCOUNT1', 'DISTANCE',
    
    # Dívida Global e Desagregada
    'DEBT_ALL_STGP_ANY_MDN',
    'DEBT_MALE_STGP_ANY_MDN', 'DEBT_NOTMALE_STGP_ANY_MDN',
    'DEBT_PELL_STGP_ANY_MDN', 'DEBT_NOPELL_STGP_ANY_MDN',
    
    # ----------------------------------------------------
    # SALÁRIOS: GLOBAIS
    # ----------------------------------------------------
    'EARN_MDN_1YR', 'EARN_MDN_4YR', 'EARN_MDN_5YR',
    
    # ----------------------------------------------------
    # SALÁRIOS: ANO 1 (Género e Pell)
    # ----------------------------------------------------
    'EARN_MALE_WNE_MDN_1YR', 'EARN_NOMALE_WNE_MDN_1YR',
    'EARN_PELL_WNE_MDN_1YR', 'EARN_NOPELL_WNE_MDN_1YR',
    
    # ----------------------------------------------------
    # SALÁRIOS: ANO 3 (Género e Pell) 
    # ----------------------------------------------------
    'EARN_MALE_NE_MDN_3YR', 'EARN_NOMALE_NE_MDN_3YR',
    'EARN_PELL_NE_MDN_3YR', 'EARN_NOPELL_NE_MDN_3YR',
    
    # ----------------------------------------------------
    # SALÁRIOS: ANO 4 (Género e Pell)
    # ----------------------------------------------------
    'EARN_MALE_WNE_MDN_4YR', 'EARN_NOMALE_WNE_MDN_4YR',
    'EARN_PELL_WNE_MDN_4YR', 'EARN_NOPELL_WNE_MDN_4YR',

    # ----------------------------------------------------
    # SALÁRIOS: ANO 5 (Género e Pell)
    # ----------------------------------------------------
    'EARN_MALE_WNE_MDN_5YR', 'EARN_NOMALE_WNE_MDN_5YR',
    'EARN_PELL_WNE_MDN_5YR', 'EARN_NOPELL_WNE_MDN_5YR',
    
    # Sucesso a pagar a dívida (2 anos depois)
    'BBRR2_FED_COMP_MAKEPROG', 'BBRR2_FED_COMP_DFLT'
]

colunas_para_converter = [
    'DEBT_ALL_STGP_ANY_MDN', 
    'EARN_MDN_1YR', 'EARN_MDN_4YR', 'EARN_MDN_5YR', 
    'IPEDSCOUNT1',
    'BBRR2_FED_COMP_DFLT',
    'DEBT_MALE_STGP_ANY_MDN', 'DEBT_NOTMALE_STGP_ANY_MDN',
    'DEBT_PELL_STGP_ANY_MDN', 'DEBT_NOPELL_STGP_ANY_MDN'
]

colunas_genero = [
        'EARN_MALE_WNE_MDN_1YR', 'EARN_NOMALE_WNE_MDN_1YR',
        'EARN_MALE_NE_MDN_3YR', 'EARN_NOMALE_NE_MDN_3YR',
        'EARN_MALE_WNE_MDN_4YR', 'EARN_NOMALE_WNE_MDN_4YR',
        'EARN_MALE_WNE_MDN_5YR', 'EARN_NOMALE_WNE_MDN_5YR'
    ]

colunas_pell = [
        'EARN_PELL_WNE_MDN_1YR', 'EARN_NOPELL_WNE_MDN_1YR',
        'EARN_PELL_NE_MDN_3YR', 'EARN_NOPELL_NE_MDN_3YR',
        'EARN_PELL_WNE_MDN_4YR', 'EARN_NOPELL_WNE_MDN_4YR',
        'EARN_PELL_WNE_MDN_5YR', 'EARN_NOPELL_WNE_MDN_5YR'
    ]

important_institution_cols = ['UNITID', 'LATITUDE', 'LONGITUDE', 'STABBR', 'CITY', 'CONTROL', 'ADM_RATE_ALL']

# Layout variables
# =========================================================
# 2. DICIONÁRIOS DE MAPEAMENTO
# =========================================================
MAPA_ESTADOS = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
    'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
    'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
    'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
    'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri',
    'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
    'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
    'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont',
    'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming',
    'DC': 'District of Columbia', 'PR': 'Puerto Rico', 'VI': 'Virgin Islands'
}