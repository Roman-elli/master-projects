import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

class DCGenerator(nn.Module):
    def __init__(self, latent_dim=100, image_channels=3, ngf=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, image_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        z = z.view(z.size(0), self.latent_dim, 1, 1)
        return self.net(z)

class DCDiscriminator(nn.Module):
    def __init__(self, image_channels=3, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(image_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).view(-1, 1)

def init_dcgan_weights(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def save_checkpoint(generator, discriminator, history, checkpoint_path, latent_dim, channels, image_size):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'history': history,
            'config': {
                'latent_dim': latent_dim,
                'channels': channels,
                'image_size': image_size,
            },
        },
        checkpoint_path,
    )
    print('Saved checkpoint to', checkpoint_path)

@torch.no_grad()
def load_dcgan_generator_for_inference(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt['config']
    generator = DCGenerator(latent_dim=cfg['latent_dim'], image_channels=cfg['channels']).to(device)
    generator.load_state_dict(ckpt['generator'])
    generator.eval()
    return generator, cfg, ckpt.get('history', None)

def train_gan(generator, discriminator, loader, latent_dim, epochs=20, lr=2e-4, device='cpu'):
    criterion = nn.BCELoss()

    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    history = {'g_loss': [], 'd_loss': []}
    generator.train()
    discriminator.train()

    for epoch in range(epochs):
        g_running = 0.0
        d_running = 0.0
        n_batches = 0

        for real, _, _ in tqdm(loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False):
            real = real.to(device)
            bs = real.size(0)

            real_targets = torch.ones(bs, 1, device=device)
            fake_targets = torch.zeros(bs, 1, device=device)

            # TODO START - Discriminator update
            # 1) Reset discriminator gradients for the new mini-batch.
            opt_d.zero_grad()
            
            # 2) Measure how well it recognizes real images as real.
            d_loss_real = criterion(discriminator(real), real_targets)
            
            # 3) Sample latent noise and generate a fake batch.
            z = torch.randn(bs, latent_dim, device=device)
            fake = generator(z)
            
            # 4) Measure how well it recognizes fake images as fake.
            d_loss_fake = criterion(discriminator(fake.detach()), fake_targets)
            
            # 5) Combine both real/fake discriminator losses.
            d_loss = d_loss_real + d_loss_fake
            
            # 6) Backpropagate discriminator loss and update discriminator weights.
            d_loss.backward()
            opt_d.step()
            # TODO END

            if d_loss is None:
                raise NotImplementedError('Implement Discriminator update in train_gan.')

            # TODO START - Generator update
            # 1) Reset generator gradients for the new mini-batch.
            opt_g.zero_grad()
            
            # 2) Sample fresh latent noise and generate a fake batch.
            z = torch.randn(bs, latent_dim, device=device)
            fake = generator(z)
            
            # 3) Evaluate how convincing those fake images look to the discriminator.
            preds = discriminator(fake)
            
            # 4) Compute generator loss so fake images are pushed toward "real" predictions.
            g_loss = criterion(preds, real_targets)
            
            # 5) Backpropagate generator loss and update generator weights.
            g_loss.backward()
            opt_g.step()
            # TODO END

            if g_loss is None:
                raise NotImplementedError('Implement Generator update in train_gan.')

            # TODO START - Bookkeeping
            g_running += g_loss.item()
            d_running += d_loss.item()
            n_batches += 1
            # TODO END

        history['g_loss'].append(g_running / max(n_batches, 1))
        history['d_loss'].append(d_running / max(n_batches, 1))

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"D loss: {history['d_loss'][-1]:.4f} | "
            f"G loss: {history['g_loss'][-1]:.4f}"
        )

    return history