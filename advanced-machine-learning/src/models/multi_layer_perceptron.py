import torch.nn as nn
import torch
import copy
import config as cfg
from utils.io import save_best_model

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

def fit(train_dataloader, val_dataloader, nn, criterion, optimizer, n_epochs, to_device=True, device="cpu"):
    if to_device:
        nn = nn.to(device)

    train_acc_values = []
    val_acc_values = []
    
    # Variáveis para rastrear o melhor modelo
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(nn.state_dict())

    for epoch in range(n_epochs):
        
        # FASE DE TREINO
        nn.train()
        correct_train = 0  
        total_train = 0    
        
        for X_batch, y_batch in train_dataloader:
            if to_device:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
            outputs = nn(X_batch)
            loss = criterion(outputs, y_batch) 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1) 
            total_train += y_batch.size(0)            
            correct_train += (predicted == y_batch).sum().item() 
            
        epoch_train_acc = correct_train / total_train
        train_acc_values.append(epoch_train_acc)

        # FASE DE VALIDAÇÃO
        nn.eval()
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for X_val, y_val in val_dataloader:
                if to_device:
                    X_val, y_val = X_val.to(device), y_val.to(device)

                val_outputs = nn(X_val)
                
                _, predicted = torch.max(val_outputs.data, 1)
                total_val += y_val.size(0)
                correct_val += (predicted == y_val).sum().item()
                
        epoch_val_acc = correct_val / total_val
        val_acc_values.append(epoch_val_acc)

        print(f'Epoch [{epoch+1}/{n_epochs}], Train Acc: {epoch_train_acc*100:.2f}% | Val Acc: {epoch_val_acc*100:.2f}%')

        # Atualizar os melhores pesos se a validação melhorar
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(nn.state_dict())

    # Carregar os melhores pesos de volta para o modelo
    print(f"\nTreino concluído. A restaurar os pesos do melhor modelo (Val Acc: {best_val_acc*100:.2f}%)...")
    nn.load_state_dict(best_model_wts)

    if cfg.AUGMENT_DATA:
        save_dir = cfg.mlp_results_path_augmented
    else:
        save_dir = cfg.mlp_results_path

    save_best_model(best_model_wts, save_dir=save_dir)

    return train_acc_values, val_acc_values, nn.to("cpu")