import torch
import torch.nn as nn
from torch.functional import F
import copy
import config as cfg
from utils.io import save_best_model

class CNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(CNN, self).__init__()
        
        # 1ª Camada
        # Entrada: [Batch, 3, 64, 64]
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        # Após conv1 + MaxPool: [Batch, 32, 32, 32]

        # 2ª Camada
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Após conv2 + MaxPool: [Batch, 64, 16, 16]
        
        # 3ª Camada
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # Após conv3 + MaxPool: [Batch, 128, 8, 8]
        
        # 128 canais * 8 de altura * 8 de largura = 8192
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        
        # Dropout desliga 50% dos neurónios aleatoriamente no treino para evitar Overfitting
        self.dropout = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        
        x = x.view(-1, self.fc1.in_features) # Flatten
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def fit(train_dataloader, val_dataloader, nn, criterion, optimizer, n_epochs, to_device=True, device="cpu"):
    if to_device:
        nn = nn.to(device)

    # Listas para guardar a accuracy
    train_acc_values = []
    val_acc_values = []

    # Variáveis para rastrear o melhor modelo
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(nn.state_dict())

    for epoch in range(n_epochs):
        # --- TREINO ---
        nn.train() 
        correct_train = 0
        total_train = 0
        
        for X_batch, y_batch in train_dataloader:
            if to_device:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = nn(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Atualizar os pesos
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calcular Accuracy do batch
            # torch.max devolve o valor máximo e o índice (que é a classe prevista)
            _, predicted = torch.max(outputs.data, 1)
            total_train += y_batch.size(0) # Adiciona o número de imagens no batch
            correct_train += (predicted == y_batch).sum().item() # Conta quantas acertou
            
        # Calcular Accuracy da época de treino (Acertos / Total)
        epoch_train_acc = correct_train / total_train
        train_acc_values.append(epoch_train_acc)

        # --- VALIDAÇÃO ---
        nn.eval()
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for X_val, y_val in val_dataloader:
                if to_device:
                    X_val, y_val = X_val.to(device), y_val.to(device)
                    
                val_outputs = nn(X_val)
                
                # Calcular Accuracy do batch de validação
                _, predicted_val = torch.max(val_outputs.data, 1)
                total_val += y_val.size(0)
                correct_val += (predicted_val == y_val).sum().item()
                
        # Calcular Accuracy da época de validação
        epoch_val_acc = correct_val / total_val
        val_acc_values.append(epoch_val_acc)

        # Imprimir os resultados em percentagem
        print(f'Epoch [{epoch+1}/{n_epochs}], Train Acc: {epoch_train_acc*100:.4f}% | Val Acc: {epoch_val_acc*100:.4f}%')
    
        # Atualizar os melhores pesos se a validação melhorar
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(nn.state_dict())

    
     # Carregar os melhores pesos de volta para o modelo
    print(f"\nTreino concluído. A restaurar os pesos do melhor modelo (Val Acc: {best_val_acc*100:.2f}%)...")
    nn.load_state_dict(best_model_wts)

    if cfg.AUGMENT_DATA:
        save_dir = cfg.cnn_results_path_augmented
    else:
        save_dir = cfg.cnn_results_path

    save_best_model(best_model_wts, save_dir=save_dir)

    return train_acc_values, val_acc_values, nn.to("cpu")