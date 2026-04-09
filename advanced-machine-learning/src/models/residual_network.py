import torch.nn as nn
import torch.nn.functional as F
import torch
import copy
import config as cfg
from utils.io import save_best_model
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self, num_classes=75):
        super(ResNet, self).__init__()
        
        # 1. Carregar a arquitetura ResNet-18
        self.resnet = models.resnet18(weights='DEFAULT')
        
        # 2. Modificar a última camada
        num_features = self.resnet.fc.in_features
        
        # Substituímos a camada final original por uma nova focada nas tuas 75 classes
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x) 
    
def fit(train_dataloader, val_dataloader, nn, criterion, optimizer, n_epochs, to_device=True, device="cpu"):
    if to_device:
        nn = nn.to(device)

    # Listas para guardar a accuracy em vez da loss
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
        save_dir = cfg.rnet_results_path_augmented
    else:
        save_dir = cfg.rnet_results_path

    save_best_model(best_model_wts, save_dir=save_dir)

    return train_acc_values, val_acc_values, nn.to("cpu")