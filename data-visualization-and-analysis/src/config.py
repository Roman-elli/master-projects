# Path variables
RAW_COURSES_PATH = 'raw_data/raw_courses_data.csv'
RAW_INSTITUTION_PATH = 'raw_data/raw_institution_data.csv'
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
    'IPEDSCOUNT1'
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
