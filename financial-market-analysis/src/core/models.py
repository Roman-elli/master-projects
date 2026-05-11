from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import pandas as pd

def train_meta_agent(dataset):
    """Treina o Random Forest aplicando o Purged K-Fold (gap temporal)."""
    X = dataset[['ecdf', 'theta', 'half_life', 'volatility', 'side']]
    y = dataset['target']
    
    # TimeSeriesSplit com Embargo (Prevenção de Fugas de Informação)
    tscv = TimeSeriesSplit(n_splits=5, gap=20)
    
    meta_agent = RandomForestClassifier(
        n_estimators=150, max_depth=4, min_samples_leaf=10, 
        random_state=42, class_weight='balanced_subsample'
    )
    
    cv_scores = cross_val_score(meta_agent, X, y, cv=tscv, scoring='accuracy')
    meta_agent.fit(X, y)
    
    # Feature Importances
    importances = pd.Series(meta_agent.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    return meta_agent, cv_scores, importances

def train_specific_agents(dataset):
    """Treina um modelo Random Forest dedicado para cada par institucional."""
    specific_agents = {}
    
    # Agrupar dados por par
    for pair_id, df_pair in dataset.groupby('pair_id'):
        # Regra de Segurança: O par tem de ter exemplos de Vitória (1) e Derrota (0)
        # e um mínimo de amostras para não criar árvores viciadas.
        if len(df_pair) < 15 or len(df_pair['target'].unique()) < 2:
            print(f"Par {pair_id} ignorado no modelo específico (Dados insuficientes/Sem variância).")
            continue
            
        X = df_pair[['ecdf', 'theta', 'half_life', 'volatility', 'side']]
        y = df_pair['target']
        
        # Modelo ligeiramente mais leve para evitar overfitting em amostras pequenas
        agent = RandomForestClassifier(
            n_estimators=100, max_depth=3, min_samples_leaf=5, 
            random_state=42, class_weight='balanced_subsample'
        )
        agent.fit(X, y)
        specific_agents[pair_id] = agent
        
    print(f"-> {len(specific_agents)} Modelos Específicos treinados com sucesso.")
    return specific_agents