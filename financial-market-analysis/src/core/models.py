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
