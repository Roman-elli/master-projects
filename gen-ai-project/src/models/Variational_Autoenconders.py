import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        # --- ENCODER MELHORADO ---
        self.encoder = nn.Sequential(
            # 32x32 -> 16x16
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64), # NOVO: Estabiliza o treino
            nn.LeakyReLU(0.2, inplace=True), # NOVO: Previne "dead neurons"
            
            # 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Flatten()
        )

        # Atualizado para refletir os 256 canais no nível mais profundo
        flattened_dim = 256 * 4 * 4
        
        # Espaço Latente
        self.fc_mu = nn.Linear(flattened_dim, latent_dim)
        self.fc_logvar = nn.Linear(flattened_dim, latent_dim)

        # Entrada do Decoder
        self.decoder_input = nn.Linear(latent_dim, flattened_dim)
        
        # --- DECODER MELHORADO ---
        self.decoder = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            # NOTA IMPORTANTE: Sigmoid assume que as imagens originais estão entre [0, 1]
            nn.Sigmoid(), 
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        h = self.decoder_input(z).view(-1, 256, 4, 4) 
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        xhat = self.decode(z)
        return xhat, mu, logvar
    
    def sample(self, num_samples, device):
        z = torch.randn(num_samples, self.latent_dim).to(device)
        z = self.decoder_input(z).view(-1, 256, 4, 4) # Alterado para 256
        samples = self.decoder(z)
        return samples
    
    
def vae_loss(xhat, x, mu, logvar, beta=0.7):
    recon_loss = F.binary_cross_entropy(xhat, x, reduction='sum') / x.shape[0]
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
    loss = recon_loss + beta * kl_loss
    return loss, recon_loss, kl_loss

def train_vae(model, loader, optimizer, device, epochs=20, beta=0.7):
    model.train()
    hist = []
    for ep in range(epochs):
        tl, tr, tk = 0.0, 0.0, 0.0
        for x, _, _ in tqdm(loader, leave=False):
            x = x.to(device)

            optimizer.zero_grad()
            xhat, mu, logvar = model(x)
            loss, recon, kl = vae_loss(xhat, x, mu, logvar, beta=beta)
            
            loss.backward()
            optimizer.step()

            tl += loss.item() * x.size(0)
            tr += recon.item() * x.size(0)
            tk += kl.item() * x.size(0)
            
        n = len(loader.dataset)
        hist.append({'train_loss': tl/n, 'train_recon_bce': tr/n, 'train_kl': tk/n})
        print(f'Epoch {ep+1}/{epochs} | train_loss={tl/n:.4f} train_recon={tr/n:.4f} train_kl={tk/n:.4f}')
    return hist