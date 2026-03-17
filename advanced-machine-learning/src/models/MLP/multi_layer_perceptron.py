import torch.nn as nn

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
            
            # Camadas Ocultas (Índices Corrigidos Aqui!)
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
    
def fit(dataloader, nn, criterion, optimizer, n_epochs, to_device=True, device="cpu"):
    # send everything to the device (ideally a GPU)
    if to_device:
        nn = nn.to(device)

    # Train the network
    loss_values = []
    for epoch in range(n_epochs):
        accu_loss = 0
        
        # O dataloader entrega os lotes automaticamente!
        for X_batch, y_batch in dataloader:
            
            # 1. Enviar batch para o dispositivo
            if to_device:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
            X_batch = X_batch.view(X_batch.size(0), -1)

            # Forward pass
            outputs = nn(X_batch)
            loss = criterion(outputs, y_batch)
            accu_loss += loss.item()
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, n_epochs, accu_loss))
        loss_values.append(accu_loss)

    return loss_values, nn.to("cpu")