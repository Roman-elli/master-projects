import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path

class cDCGenerator(nn.Module):
    def __init__(self, num_classes, latent_dim=100, image_channels=3, ngf=64, embed_size=100):
        super().__init__()
        self.latent_dim = latent_dim
        self.label_emb = nn.Embedding(num_classes, embed_size)

        self.net = nn.Sequential(
            # Camada 1: Z+Embed (1x1) -> 4x4
            nn.ConvTranspose2d(latent_dim + embed_size, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # MÁGICA SOTA: Força a diversidade quebrando a memorização do Mode Collapse
            nn.Dropout2d(0.3),

            # Camada 2: 4x4 -> 8x8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # Camada 3: 8x8 -> 16x16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # Camada 4: 16x16 -> 32x32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # Camada 5 (NOVA): 32x32 -> 64x64
            nn.ConvTranspose2d(ngf, image_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, labels):
        c = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        z = z.view(z.size(0), self.latent_dim, 1, 1)
        x = torch.cat([z, c], 1)
        return self.net(x)

class cDCDiscriminator(nn.Module):
    def __init__(self, num_classes, image_channels=3, ndf=64, img_size=64):
        super().__init__()
        self.img_size = img_size
        self.label_emb = nn.Embedding(num_classes, img_size * img_size)

        self.net = nn.Sequential(
            # Camada 1: 64x64 -> 32x32
            nn.Conv2d(image_channels + 1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # Camada 2: 32x32 -> 16x16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Camada 3: 16x16 -> 8x8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Camada 4 (NOVA): 8x8 -> 4x4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Camada 5: 4x4 -> 1x1
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),

            # Camada de Segurança (força sempre a ser 1x1, independentemente de erros de arredondamento)
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid(),
        )

    def forward(self, img, labels):
        c = self.label_emb(labels).view(-1, 1, self.img_size, self.img_size)
        x = torch.cat([img, c], 1)
        return self.net(x).view(-1, 1)

def init_dcgan_weights(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def train_cgan(generator, discriminator, loader, latent_dim, epochs=20, lr=2e-4, device='cpu', save_dir=None, verbose=True):
    """Ciclo de treino para a GAN Condicional"""
    criterion = nn.BCELoss()

    lr_g = 0.0004
    lr_d = 0.0001

    opt_g = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

    history = {'g_loss': [], 'd_loss': []}
    generator.train()
    discriminator.train()

    if verbose:
        print(f"A iniciar treino da cDCGAN por {epochs} épocas...")

    for epoch in range(epochs):
        g_running = 0.0
        d_running = 0.0
        n_batches = 0

        # Interruptor Silencioso para o Optuna não encher o ecrã
        iterator = tqdm(loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False) if verbose else loader

        for real, labels in iterator:
            real = real.to(device)
            labels = labels.to(device)

            # Lemos o tamanho real exato deste lote específico
            bs = real.size(0)

            # Labels Reais = 0.8 (Smoothing), Labels Falsas = 0.0
            real_targets = (torch.ones(bs, 1, device=device) * 0.8)
            fake_targets = torch.zeros(bs, 1, device=device)

            # -------------------------------------------------
            # 1. TREINAR O DISCRIMINADOR
            # -------------------------------------------------
            opt_d.zero_grad()

            # 1.1 Avaliar imagens REAIS
            pred_real = discriminator(real, labels)
            d_loss_real = criterion(pred_real, real_targets)

            # 1.2 Avaliar imagens FALSAS
            z = torch.randn(bs, latent_dim, device=device)
            fake = generator(z, labels)
            pred_fake = discriminator(fake.detach(), labels) # .detach() bloqueia o gradiente de passar pro gerador
            d_loss_fake = criterion(pred_fake, fake_targets)

            # 1.3 Atualizar Pesos do Discriminador (Dividido por 2 para estabilidade SOTA)
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            opt_d.step()

            # -------------------------------------------------
            # 2. TREINAR O GERADOR
            # -------------------------------------------------
            opt_g.zero_grad()

            # Avaliamos as imagens falsas novamente
            preds = discriminator(fake, labels)

            # O Gerador quer que as previsões sejam 1 perfeitas (Sem smoothing para o G!)
            g_targets = torch.ones(bs, 1, device=device)
            g_loss = criterion(preds, g_targets)

            g_loss.backward()
            opt_g.step()

            # --- Bookkeeping ---
            g_running += g_loss.item()
            d_running += d_loss.item()
            n_batches += 1

        avg_g_loss = g_running / n_batches
        avg_d_loss = d_running / n_batches
        history['g_loss'].append(avg_g_loss)
        history['d_loss'].append(avg_d_loss)

        if verbose:
            print(f"Epoch {epoch + 1:02d} | D loss: {avg_d_loss:.4f} | G loss: {avg_g_loss:.4f}")

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(generator.state_dict(), save_dir / 'cgan_generator.pth')
        torch.save(discriminator.state_dict(), save_dir / 'cgan_discriminator.pth')
        if verbose:
            print(f"Modelos guardados em {save_dir}")

    return generator, history