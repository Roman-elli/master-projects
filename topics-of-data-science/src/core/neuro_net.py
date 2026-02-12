from sklearn.preprocessing import OneHotEncoder
import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from core.classifier import evaluate_and_save_metrics

# =============================================================================
# 5. IMPLEMENTAÇÃO DE REDE NEURONAL DE RAIZ (FROM SCRATCH)
# =============================================================================
class CustomMLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate
        
        # Inicialização de Pesos (Aleatório pequeno) e Bias (Zeros)
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.1
        self.b1 = np.zeros((1, self.hidden_size))
        
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.1
        self.b2 = np.zeros((1, self.output_size))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def train(self, X, y, X_val=None, y_val=None, epochs=1000):
        loss_history = []
        val_loss_history = [] 
        m = X.shape[0]
        
        # Barra de progresso
        pbar = tqdm(range(epochs), desc=">>> Treinando", unit="epoch")
        
        for epoch in pbar:
            # --- A. TREINO (TRAIN SET) ---
            # 1. Feedforward
            input_layer = X
            hidden_input = np.dot(input_layer, self.W1) + self.b1
            hidden_output = self.sigmoid(hidden_input)
            
            final_input = np.dot(hidden_output, self.W2) + self.b2
            final_output = self.sigmoid(final_input)
            
            # 2. Erro e Loss de Treino
            error = y - final_output
            loss = np.mean(np.square(error))
            loss_history.append(loss)
            
            # --- B. VALIDAÇÃO (VALID SET - Monitorização apenas) ---
            current_val_loss = 0
            if X_val is not None and y_val is not None:
                # Apenas Feedforward (sem Backprop)
                v_hidden = self.sigmoid(np.dot(X_val, self.W1) + self.b1)
                v_final = self.sigmoid(np.dot(v_hidden, self.W2) + self.b2)
                
                v_error = y_val - v_final
                current_val_loss = np.mean(np.square(v_error))
                val_loss_history.append(current_val_loss)
            
            # --- C. BACKPROPAGATION (Apenas no Treino!) ---
            d_output = error * self.sigmoid_derivative(final_output)
            error_hidden = d_output.dot(self.W2.T)
            d_hidden = error_hidden * self.sigmoid_derivative(hidden_output)
            
            # --- D. UPDATE PESOS (Média do Batch) ---
            self.W2 += (hidden_output.T.dot(d_output) / m) * self.lr
            self.b2 += (np.sum(d_output, axis=0, keepdims=True) / m) * self.lr
            
            self.W1 += (input_layer.T.dot(d_hidden) / m) * self.lr
            self.b1 += (np.sum(d_hidden, axis=0, keepdims=True) / m) * self.lr
            
            # Atualizar barra: Mostra Loss de Treino e Validação
            logs = {"loss": f"{loss:.4f}"}
            if X_val is not None:
                logs["val_loss"] = f"{current_val_loss:.4f}"
                
            pbar.set_postfix(logs)
                
        return loss_history, val_loss_history

    def predict_proba(self, X):
        hidden_input = np.dot(X, self.W1) + self.b1
        hidden_output = self.sigmoid(hidden_input)
        final_input = np.dot(hidden_output, self.W2) + self.b2
        return self.sigmoid(final_input)

def run_custom_mlp(X_train, y_train, X_val, y_val, X_test, y_test, selected_features, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    print("\n" + "="*60 + "\n5. REDE NEURONAL CUSTOM (10 Execuções)\n" + "="*60)
    
    # 1. Preparar Dados (Scaling)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train[:, selected_features])
    X_va = scaler.transform(X_val[:, selected_features])   # Validação Normalizada
    X_ts = scaler.transform(X_test[:, selected_features])  # Teste Normalizado
    
    # 2. One-Hot Encoding (Target)
    enc = OneHotEncoder(sparse_output=False)
    y_tr_encoded = enc.fit_transform(y_train.reshape(-1, 1)) # Treino codificado
    y_va_encoded = enc.transform(y_val.reshape(-1, 1))       # Validação codificada
    
    # Configurações
    INPUT_SIZE = X_tr.shape[1]
    HIDDEN_SIZE = 50
    OUTPUT_SIZE = y_tr_encoded.shape[1]
    LR = 0.5         
    EPOCHS = 5000    
    N_RUNS = 10      
    
    print(f"Config: Hidden={HIDDEN_SIZE}, LR={LR}, Epochs={EPOCHS}, Runs={N_RUNS}")

    custom_scores = []
    
    # Variáveis para guardar a MELHOR execução
    best_overall_f1 = -1
    best_overall_pred = None
    best_loss_history = None
    best_val_loss_history = None
    
    print(f">>> Iniciando bateria de testes...")
    
    for i in range(N_RUNS):
        nn = CustomMLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, learning_rate=LR)
        
        print(f"   Run {i+1}/{N_RUNS}...", end="\r")
        
        # TREINO (passando validação)
        loss_hist, val_loss_hist = nn.train(X_tr, y_tr_encoded, X_val=X_va, y_val=y_va_encoded, epochs=EPOCHS)
        
        # Prever no Teste
        raw_output = nn.predict_proba(X_ts)
        predicted_indices = np.argmax(raw_output, axis=1)
        y_pred = enc.categories_[0][predicted_indices]
        
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        custom_scores.append(f1)
        
        if f1 > best_overall_f1:
            best_overall_f1 = f1
            best_overall_pred = y_pred
            best_loss_history = loss_hist
            best_val_loss_history = val_loss_hist

        final_val_loss = val_loss_hist[-1] if val_loss_hist else 0
        print(f"   Run {i+1}/{N_RUNS}: Val Loss={final_val_loss:.4f} | Test F1={f1:.4f}")

    # --- ESTATÍSTICAS ---
    mean_score = np.mean(custom_scores)
    std_score = np.std(custom_scores)
    print(f"\n>>> Custom MLP (10 Runs): Média={mean_score:.4f} (+/- {std_score:.4f})")

    # --- BENCHMARK SCIKIT-LEARN ---
    print("\n>>> Calculando Benchmark Scikit-Learn...")
    from sklearn.neural_network import MLPClassifier
    mlp_sk = MLPClassifier(hidden_layer_sizes=(HIDDEN_SIZE,), activation='logistic', 
                           solver='sgd', learning_rate='constant', momentum=0.0,
                           learning_rate_init=LR, max_iter=EPOCHS, random_state=42)
    mlp_sk.fit(X_tr, y_train)
    y_pred_sk = mlp_sk.predict(X_ts)
    f1_sk = f1_score(y_test, y_pred_sk, average='macro', zero_division=0)
    print(f"    Scikit-Learn F1 (Ref): {f1_sk:.4f}")

    # --- GRÁFICO 1: ESTABILIDADE (RUNS) ---
    plt.figure(figsize=(10, 6))
    runs_x = range(1, N_RUNS + 1)
    plt.plot(runs_x, custom_scores, marker='o', linestyle='-', color='b', label='Custom MLP')
    plt.axhline(y=f1_sk, color='r', linestyle='--', label=f'Scikit-Learn ({f1_sk:.3f})')
    plt.axhline(y=mean_score, color='g', linestyle=':', label=f'Média Custom ({mean_score:.3f})')
    plt.title(f'Estabilidade do Custom MLP ({N_RUNS} Execuções)')
    plt.xlabel('Run')
    plt.ylabel('F1-Score')
    plt.xticks(runs_x)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "custom_mlp_runs_plot.png"))
    print(f"[✔] Gráfico Linear salvo.")
    
    # --- GRÁFICO 2: CONVERGÊNCIA (LOSS da Melhor Run) ---
    plt.figure(figsize=(8, 4))
    plt.plot(best_loss_history, label='Treino Loss')
    if best_val_loss_history:
        plt.plot(best_val_loss_history, label='Validação Loss', linestyle='--')
    plt.title('Curva de Aprendizagem (Melhor Run)')
    plt.xlabel('Épocas')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, "custom_training_loss.png"))
    print(f"[✔] Gráfico de Convergência salvo.")

    # --- SALVAR RESULTADOS ---
    evaluate_and_save_metrics(y_test, best_overall_pred, save_dir, fold_name="custom_mlp_best_run")
    
    pd.DataFrame({
        "Run": runs_x, 
        "F1_Score": custom_scores
    }).to_csv(os.path.join(save_dir, "custom_mlp_all_runs.csv"), index=False)