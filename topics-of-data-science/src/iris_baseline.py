from sklearn.datasets import load_iris
from iris_core.iris_metrics import run_2_1_baselines, run_2_2_knn_analysis, run_2_3_relief_tvt, run_2_4_relief_cv, create_imbalanced_iris
import numpy as np

def main():
    # Carregar dados originais
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # --- 2.1 & 2.2 (Já feitos na resposta anterior, mas incluídos aqui) ---
    run_2_1_baselines(X, y)
    run_2_2_knn_analysis(X, y)
    
    # --- 2.3 ReliefF + TVT ---
    run_2_3_relief_tvt(X, y, title_suffix="(Original)")
    
    # --- 2.4 ReliefF + CV ---
    run_2_4_relief_cv(X, y)
    
    # --- 2.5 Imbalanced Data ---
    print("\n\n>>> GERANDO DATASET DESEQUILIBRADO PARA 2.5...")
    X_imb, y_imb = create_imbalanced_iris()
    # Check distribuição
    unique, counts = np.unique(y_imb, return_counts=True)
    print(f"Distribuição Imbalanced: {dict(zip(unique, counts))}")
    
    # Repetir 2.3 com dados desequilibrados
    run_2_3_relief_tvt(X_imb, y_imb, title_suffix="(Imbalanced)")

if __name__ == '__main__':
    main()