import torch
import torch.nn as nn
from tqdm import tqdm
import copy
import os
import lpips

class ConditionalVAE(nn.Module):
    def __init__(self, num_classes, latent_dim=128, img_channels=3, img_size=64):
        super().__init__()
        
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.img_size = img_size
        
        # Embedding da classe para o Encoder (transformado num canal extra 64x64)
        self.class_embed_enc = nn.Embedding(num_classes, img_size * img_size)
        
        # ENCODER: Recebe 3 canais RGB + 1 canal da label
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels + 1, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten()
        )
        
        self.flatten_size = 256 * 4 * 4
        
        # Espaço Latente (Média e Variância)
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
        # Embedding da classe para o Decoder (vetor 1D concatenado com Z)
        self.class_embed_dec = nn.Embedding(num_classes, latent_dim)
        
        self.fc_decode = nn.Sequential(
            nn.Linear(latent_dim * 2, self.flatten_size),
            nn.ReLU(inplace=True)
        )
        
        # DECODER
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() # Normaliza os pixels gerados para [0, 1]
        )

    def encode(self, x, labels):
        label_embed = self.class_embed_enc(labels).view(-1, 1, self.img_size, self.img_size)
        x = torch.cat([x, label_embed], dim=1)
        hidden = self.encoder(x)
        return self.fc_mu(hidden), self.fc_logvar(hidden)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, labels):
        label_embed = self.class_embed_dec(labels)
        z = torch.cat([z, label_embed], dim=1)
        hidden = self.fc_decode(z)
        hidden = hidden.view(-1, 256, 4, 4)
        return self.decoder(hidden)

    def forward(self, x, labels):
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        reconstructed_img = self.decode(z, labels)
        return reconstructed_img, mu, logvar

# --- Funções de Treino e Loss ---
def vae_loss_function(recon_x, x, mu, logvar, loss_fn_vgg, beta=0.01, perceptual_weight=1.0):
    """Calcula a perda do VAE combinando MSE, Perceptual (LPIPS) e KLD na mesma escala."""
    
    # 1. MSE: Média sobre todos os píxeis e batch (valor tipicamente entre 0.01 e 0.1)
    MSE = nn.functional.mse_loss(recon_x, x, reduction='mean')
    
    # 2. Perda Perceptual: LPIPS já calcula uma média normalizada (valor tipicamente entre 0.1 e 0.5)
    recon_x_scaled = recon_x * 2.0 - 1.0
    x_scaled = x * 2.0 - 1.0
    P_LOSS = loss_fn_vgg(recon_x_scaled, x_scaled).mean()
    
    # Ao tirar a média, o KLD fica na mesma casa decimal do MSE
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    total_loss = MSE + (perceptual_weight * P_LOSS) + (beta * KLD)
    
    return total_loss, MSE, P_LOSS, KLD

def train_cvae(model, train_loader, val_loader, optimizer, device, num_epochs=50, save_dir=None):
    """Ciclo de treino completo para o cVAE com LPIPS e Model Checkpointing."""
    history = {'train_loss': [], 'val_loss': [], 'train_mse': [], 'train_perceptual': [], 'train_kld': []}
    
    # Inicializa a VGG pré-treinada para a Perda Perceptual (congela os pesos automaticamente)
    print("Carregando modelo VGG para Perda Perceptual...")
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_mse_loss, train_p_loss, train_kld_loss = 0, 0, 0, 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(inputs, labels)
            
            # Passamos a loss_fn_vgg e os hiperparâmetros (beta e peso perceptual)
            loss, mse, p_loss, kld = vae_loss_function(
                recon_batch, inputs, mu, logvar, loss_fn_vgg, beta=0.05, perceptual_weight=1.5
            )
            
            loss.backward()
            optimizer.step()
            
            # reduction='mean', multiplicamos pelo tamanho do batch atual para a média final
            batch_size = inputs.size(0)
            train_loss += loss.item() * batch_size
            train_mse_loss += mse.item() * batch_size
            train_p_loss += p_loss.item() * batch_size
            train_kld_loss += kld.item() * batch_size
            
        n_train = len(train_loader.dataset)
        avg_train_loss = train_loss / n_train
        history['train_loss'].append(avg_train_loss)
        history['train_mse'].append(train_mse_loss / n_train)
        history['train_perceptual'].append(train_p_loss / n_train)
        history['train_kld'].append(train_kld_loss / n_train)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                recon_batch, mu, logvar = model(inputs, labels)
                loss, _, _, _ = vae_loss_function(
                    recon_batch, inputs, mu, logvar, loss_fn_vgg, beta=0.05, perceptual_weight=1.5
                )
                val_loss += loss.item() * inputs.size(0)
                
        avg_val_loss = val_loss / len(val_loader.dataset)
        history['val_loss'].append(avg_val_loss)
        
        print(f'Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} (P_Loss: {(train_p_loss/n_train):.4f}) | Val Loss: {avg_val_loss:.4f}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

    print(f'\nTreinamento concluído. Melhor Validation Loss: {best_val_loss:.4f}')

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, 'cvae_last_model.pth'))
        torch.save(best_model_wts, os.path.join(save_dir, 'cvae_best_model.pth'))

    model.load_state_dict(best_model_wts)
    return model, history