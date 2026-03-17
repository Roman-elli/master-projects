import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        if len(hidden_sizes) == 0:
            self.layers.append(nn.Linear(input_size, num_classes))
        else:
            # 1ª Camada (Input -> 1ª Oculta)
            self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
            self.activations.append(nn.ReLU())        
            
            # Camadas Ocultas
            for i in range(len(hidden_sizes)-1):
                self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
                self.activations.append(nn.ReLU())

            # Última camada (Última Oculta -> Classes de Saída)
            self.layers.append(nn.Linear(hidden_sizes[-1], num_classes))
    
    def forward(self, x):
        for ix in range (len(self.layers) - 1):
            x = self.layers[ix](x)
            x = self.activations[ix](x)
        return self.layers[-1](x)
    
def fit(train_dataloader, val_dataloader, nn, criterion, optimizer, n_epochs, to_device=True, device="cpu"):
    if to_device:
        nn = nn.to(device)

    train_loss_values = []
    val_loss_values = []

    for epoch in range(n_epochs):
        nn.train()
        accu_train_loss = 0
        
        for X_batch, y_batch in train_dataloader:
            if to_device:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
            X_batch = X_batch.view(X_batch.size(0), -1) # Flatten

            outputs = nn(X_batch)
            loss = criterion(outputs, y_batch)
            accu_train_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Calcular média para não depender do tamanho do dataset
        avg_train_loss = accu_train_loss / len(train_dataloader)
        train_loss_values.append(avg_train_loss)

        # Validação
        nn.eval()
        accu_val_loss = 0
        
        with torch.no_grad():
            for X_val, y_val in val_dataloader:
                if to_device:
                    X_val, y_val = X_val.to(device), y_val.to(device)
                    
                X_val = X_val.view(X_val.size(0), -1) # Flatten

                val_outputs = nn(X_val)
                val_loss = criterion(val_outputs, y_val)
                accu_val_loss += val_loss.item()
                
        # Calcular média
        avg_val_loss = accu_val_loss / len(val_dataloader)
        val_loss_values.append(avg_val_loss)

        print(f'Epoch [{epoch+1}/{n_epochs}], Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')
    
    return train_loss_values, val_loss_values, nn.to("cpu")