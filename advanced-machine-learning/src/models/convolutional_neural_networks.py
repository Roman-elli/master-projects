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
