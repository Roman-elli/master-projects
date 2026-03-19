import torch
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import config as cfg
import os

def analyze_df(df, train_dataset):
    sample_img, _ = train_dataset[0]
    num_inputs = sample_img.shape[0] * sample_img.shape[1] * sample_img.shape[2]
    num_classes = len(train_dataset.classes)
    
    print("-"*30)
    print(f"Número de classes: {num_classes}")
    print(f"Número de inputs para o MLP: {num_inputs}")
    print("-"*30)

    class_counts = df['label'].value_counts()

    class_proportions = df['label'].value_counts(normalize=True) * 100

    distribution_summary = pd.DataFrame({
        'Quantidade de Imagens': class_counts,
        'Proporção (%)': class_proportions
    })
    print("-"*30)
    print(distribution_summary)
    print("-"*30)
    
    return num_inputs, num_classes

# utils/metrics.py

def evaluate_network(net, dataloader, device="cpu", save_path=None, split_name="Avaliacao", is_cnn=False):
    net.eval()
    net = net.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            
            # Se NÃO for CNN (ou seja, se for MLP), fazemos o flatten
            if not is_cnn:
                images = images.view(images.size(0), -1)
                
            # Agora enviamos para o device (sem forçar o nome images_flattened)
            images = images.to(device)
            labels = labels.to(device)
            
            # Passamos as imagens (flattened ou não, dependendo do modelo)
            outputs = net(images)
            
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    conf_mat = confusion_matrix(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted') 

    print(f'[{split_name}] F1 Score: {f1:.4f}')
    
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        
        file_name = os.path.join(save_path, f"{split_name}_metrics.txt")
        
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(f"=== Resultados da Avaliacao: {split_name} ===\n")
            f.write(f"F1 Score (Weighted): {f1:.4f}\n\n")
            f.write("Matriz de Confusao:\n")
            # Escrever a matriz de forma legível
            for row in conf_mat:
                f.write(" ".join([str(elem) for elem in row]) + "\n")
                
        print(f"Métricas de {split_name} salvas...")

    return f1, conf_mat

def save_acc_graph(train_losses, val_losses, file_name, folder_name, model_type=""):
    epochs_range = range(1, cfg.MLP_EPOCHS + 1)
        
    os.makedirs(file_name, exist_ok=True)
    
    plt.figure(figsize=(10, 6))

    plt.plot(epochs_range, train_losses, marker='o', linestyle='-', color='blue', label='Treino (Train Loss)')
    plt.plot(epochs_range, val_losses, marker='s', linestyle='--', color='red', label='Validação (Valid Loss)')

    plt.title(f"Evolução da Acc - {model_type}\n{folder_name}", fontsize=10)
    plt.xlabel('Épocas')
    plt.ylabel('ACCuracy')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()

    # CORREÇÃO AQUI: Criar o ficheiro dentro da pasta com a extensão .png
    file_path = file_name / "acc_curve.png"
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado com sucesso...")
    
    plt.close()