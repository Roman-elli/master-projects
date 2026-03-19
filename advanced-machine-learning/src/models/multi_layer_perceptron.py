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

        x = x.view(x.size(0), -1)

        for ix in range (len(self.layers) - 1):
            x = self.layers[ix](x)
            x = self.activations[ix](x)
        return self.layers[-1](x)
    
import torch

def fit(train_dataloader, val_dataloader, nn_model, criterion, optimizer, n_epochs, to_device=True, device="cpu"):
    if to_device:
        nn_model = nn_model.to(device)

    train_acc_values = []
    val_acc_values = []

    for epoch in range(n_epochs):
        
        # FASE DE TREINO
        nn_model.train()
        correct_train = 0  # Imagens que acertámos
        total_train = 0    # Total de imagens vistas
        
        for X_batch, y_batch in train_dataloader:
            if to_device:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
            # Forward pass
            outputs = nn_model(X_batch)
            loss = criterion(outputs, y_batch) # A Loss continua a ser calculada porque o otimizador precisa dela!
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 2. Calcular Accuracy do Batch
            _, predicted = torch.max(outputs.data, 1) # Pega na classe com maior pontuação
            total_train += y_batch.size(0)            # Soma o número de imagens no batch (ex: 32)
            correct_train += (predicted == y_batch).sum().item() # Soma quantas acertou
            
        # Calcular média da Accuracy da Época (0.0 a 1.0)
        epoch_train_acc = correct_train / total_train
        train_acc_values.append(epoch_train_acc)

        # FASE DE VALIDAÇÃO
        nn_model.eval()
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for X_val, y_val in val_dataloader:
                if to_device:
                    X_val, y_val = X_val.to(device), y_val.to(device)
                    
                X_val = X_val.view(X_val.size(0), -1) # Flatten

                val_outputs = nn_model(X_val)
                
                # 3. Calcular Accuracy do Batch (Validação)
                _, predicted = torch.max(val_outputs.data, 1)
                total_val += y_val.size(0)
                correct_val += (predicted == y_val).sum().item()
                
        epoch_val_acc = correct_val / total_val
        val_acc_values.append(epoch_val_acc)

        print(f'Epoch [{epoch+1}/{n_epochs}], Train Acc: {epoch_train_acc*100:.2f}% | Val Acc: {epoch_val_acc*100:.2f}%')
    
    # 4. Devolver as listas de Accuracy
    return train_acc_values, val_acc_values, nn_model.to("cpu")