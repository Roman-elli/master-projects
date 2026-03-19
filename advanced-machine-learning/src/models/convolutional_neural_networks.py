'''
import torch.nn as nn
import torch
from torch.functional import F

class CNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding = 1)
        
        #8x8x8=512 is the dimension of the tensor after the last convolution. 
        #It consists of 8 filters of size 8x8 (due to pooling)
        self.fc1 = nn.Linear(8*16*16, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x =  F.max_pool2d(F.relu(self.conv2(x)), 2)
        
        x = x.view(-1, self.fc1.in_features)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

def fit(train_dataloader, val_dataloader, nn, criterion, optimizer, n_epochs, to_device=True, device="cpu"):
    if to_device:
        nn = nn.to(device)

    train_loss_values = []
    val_loss_values = []

    for epoch in range(n_epochs):
        # --- TREINO ---
        nn.train()
        accu_train_loss = 0
        
        for X_batch, y_batch in train_dataloader:
            if to_device:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Nota: Na CNN NÃO fazemos Flatten da imagem aqui no DataLoader (ao contrário do MLP)
            # A imagem entra com as dimensões [Batch, Canais, Altura, Largura]

            outputs = nn(X_batch)
            loss = criterion(outputs, y_batch)
            accu_train_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        avg_train_loss = accu_train_loss / len(train_dataloader)
        train_loss_values.append(avg_train_loss)

        # --- VALIDAÇÃO ---
        nn.eval()
        accu_val_loss = 0
        
        with torch.no_grad():
            for X_val, y_val in val_dataloader:
                if to_device:
                    X_val, y_val = X_val.to(device), y_val.to(device)
                    
                val_outputs = nn(X_val)
                val_loss = criterion(val_outputs, y_val)
                accu_val_loss += val_loss.item()
                
        avg_val_loss = accu_val_loss / len(val_dataloader)
        val_loss_values.append(avg_val_loss)

        print(f'Epoch [{epoch+1}/{n_epochs}], Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')
    
    return train_loss_values, val_loss_values, nn.to("cpu")

'''

import torch
import torch.nn as nn
from torch.functional import F

class CNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(CNN, self).__init__()
        
        # 1ª Camada: Captar formas simples (linhas, cantos)
        # Entrada: [Batch, 3, 64, 64]
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        # Após conv1 + MaxPool: [Batch, 32, 32, 32]

        # 2ª Camada: Captar texturas e padrões
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Após conv2 + MaxPool: [Batch, 64, 16, 16]
        
        # 3ª Camada: Captar partes complexas da borboleta (asas, antenas)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # Após conv3 + MaxPool: [Batch, 128, 8, 8]
        
        # O Flatten será: 128 canais * 8 de altura * 8 de largura = 8192
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
        x = self.dropout(x) # Aplicar dropout antes da camada final
        x = self.fc2(x)
        return x
    
'''
def fit(train_dataloader, val_dataloader, nn, criterion, optimizer, n_epochs, to_device=True, device="cpu"):
    if to_device:
        nn = nn.to(device)

    train_loss_values = []
    val_loss_values = []

    for epoch in range(n_epochs):
        # --- TREINO ---
        nn.train() 
        accu_train_loss = 0
        
        for X_batch, y_batch in train_dataloader:
            if to_device:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = nn(X_batch)
            loss = criterion(outputs, y_batch)
            accu_train_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        avg_train_loss = accu_train_loss / len(train_dataloader)
        train_loss_values.append(avg_train_loss)

        # --- VALIDAÇÃO ---
        nn.eval()
        accu_val_loss = 0
        
        with torch.no_grad():
            for X_val, y_val in val_dataloader:
                if to_device:
                    X_val, y_val = X_val.to(device), y_val.to(device)
                    
                val_outputs = nn(X_val)
                val_loss = criterion(val_outputs, y_val)
                accu_val_loss += val_loss.item()
                
        avg_val_loss = accu_val_loss / len(val_dataloader)
        val_loss_values.append(avg_val_loss)

        print(f'Epoch [{epoch+1}/{n_epochs}], Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')
    
    return train_loss_values, val_loss_values, nn.to("cpu")
'''

def fit(train_dataloader, val_dataloader, nn, criterion, optimizer, n_epochs, to_device=True, device="cpu"):
    if to_device:
        nn = nn.to(device)

    # Listas para guardar a accuracy em vez da loss
    train_acc_values = []
    val_acc_values = []

    for epoch in range(n_epochs):
        # --- TREINO ---
        nn.train() 
        correct_train = 0
        total_train = 0
        
        for X_batch, y_batch in train_dataloader:
            if to_device:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = nn(X_batch)
            loss = criterion(outputs, y_batch) # Continuamos a precisar disto para a rede aprender!
            
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

        # Imprimir os resultados em percentagem (opcionalmente podes multiplicar por 100)
        print(f'Epoch [{epoch+1}/{n_epochs}], Train Acc: {epoch_train_acc:.4f} | Val Acc: {epoch_val_acc:.4f}')
    
    return train_acc_values, val_acc_values, nn.to("cpu")