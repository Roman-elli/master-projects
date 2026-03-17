import torch
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

def analyze_df(df):
    class_counts = df['label'].value_counts()

    class_proportions = df['label'].value_counts(normalize=True) * 100

    distribution_summary = pd.DataFrame({
        'Quantidade de Imagens': class_counts,
        'Proporção (%)': class_proportions
    })

    print(distribution_summary)

def evaluate_network(net, dataloader, device="cpu"):
    # 1. Colocar o modelo em modo de avaliação
    # Isto desliga comportamentos de treino como o Dropout (se usarmos no futuro)
    net.eval()
    net = net.to(device)

    # Listas para guardar todas as previsões e os rótulos reais de todos os lotes
    all_preds = []
    all_labels = []

    # 2. Desligar o cálculo de gradientes (poupa imensa memória e fica mais rápido)
    with torch.no_grad():
        for images, labels in dataloader:
            
            # 3. Flatten (Achatar a imagem para 1D, tal como no treino)
            images_flattened = images.view(images.size(0), -1)
            
            # Enviar para o dispositivo correto
            images_flattened = images_flattened.to(device)
            labels = labels.to(device)

            # 4. Forward pass
            outputs = net(images_flattened)
            
            # 5. Obter a classe com maior probabilidade
            _, predicted = torch.max(outputs, 1)

            # 6. Guardar resultados deste lote (trazendo de volta para o CPU)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 7. Calcular as métricas finais com todas as imagens
    conf_mat = confusion_matrix(all_labels, all_preds)
    
    # O average='weighted' é ótimo se tivermos um dataset desbalanceado!
    f1 = f1_score(all_labels, all_preds, average='weighted') 

    print('F1 Score: {:.4f}'.format(f1))
    #print('Confusion Matrix:\n', conf_mat)
    
    return f1, conf_mat