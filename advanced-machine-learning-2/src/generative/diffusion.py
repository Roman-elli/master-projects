import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
from pathlib import Path

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
class ResnetBlock(nn.Module):
    def __init__(self, dim, time_emb_dim, out_dim=None, dropout=0.1):
        super().__init__()
        self.out_dim = out_dim or dim
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, self.out_dim)
        )
        self.conv1 = nn.Conv2d(dim, self.out_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(self.out_dim, self.out_dim, 3, padding=1)
        
        self.norm1 = nn.GroupNorm(32, dim)
        self.norm2 = nn.GroupNorm(32, self.out_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.shortcut = nn.Conv2d(dim, self.out_dim, 1) if dim != self.out_dim else nn.Identity()

    def forward(self, x, time_emb):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        
        time_emb = self.mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return self.shortcut(x) + h
   
class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.GroupNorm(32, dim)
        self.q = nn.Conv2d(dim, dim, 1)
        self.k = nn.Conv2d(dim, dim, 1)
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj_out = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        
        q = self.q(h).view(B, C, H * W)
        k = self.k(h).view(B, C, H * W)
        v = self.v(h).view(B, C, H * W)
        
        attn = torch.bmm(q.transpose(1, 2), k) * (int(C) ** (-0.5))
        attn = F.softmax(attn, dim=2)
        
        out = torch.bmm(v, attn.transpose(1, 2))
        out = out.view(B, C, H, W)
        out = self.proj_out(out)
        return x + out 
    
class PixelUNet(nn.Module):
    def __init__(self, in_channels=3, model_channels=128, num_classes=None): 
        super().__init__()
        self.num_classes = num_classes
        
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(model_channels),
            nn.Linear(model_channels, model_channels * 4),
            nn.SiLU(),
            nn.Linear(model_channels * 4, model_channels * 4),
        )
        time_dim = model_channels * 4
        
        # Embedding Condicional de Classes
        if num_classes is not None:
            self.class_emb = nn.Embedding(num_classes, time_dim)

        self.init_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Down
        self.down1_res = ResnetBlock(model_channels, time_dim, out_dim=model_channels)
        self.down1_pool = nn.Conv2d(model_channels, model_channels, 3, stride=2, padding=1)
        self.down2_res = ResnetBlock(model_channels, time_dim, out_dim=model_channels * 2)
        self.down2_pool = nn.Conv2d(model_channels * 2, model_channels * 2, 3, stride=2, padding=1)
        self.down3_res = ResnetBlock(model_channels * 2, time_dim, out_dim=model_channels * 2)
        self.down3_pool = nn.Conv2d(model_channels * 2, model_channels * 2, 3, stride=2, padding=1)
        
        # Middle
        self.mid_res1 = ResnetBlock(model_channels * 2, time_dim)
        self.mid_attn = AttentionBlock(model_channels * 2)
        self.mid_res2 = ResnetBlock(model_channels * 2, time_dim)
        
        # Up
        self.up3_conv = nn.ConvTranspose2d(model_channels * 2, model_channels * 2, 4, stride=2, padding=1)
        self.up3_res = ResnetBlock(model_channels * 4, time_dim, out_dim=model_channels * 2)
        self.up2_conv = nn.ConvTranspose2d(model_channels * 2, model_channels * 2, 4, stride=2, padding=1) 
        self.up2_res = ResnetBlock(model_channels * 4, time_dim, out_dim=model_channels)
        self.up1_conv = nn.ConvTranspose2d(model_channels, model_channels, 4, stride=2, padding=1) 
        self.up1_res = ResnetBlock(model_channels * 2, time_dim, out_dim=model_channels)
        
        self.out_conv = nn.Conv2d(model_channels, in_channels, 3, padding=1)
        
    def forward(self, x, t, labels=None):
        t_emb = self.time_embed(t)
        
        # Funde a label com o tempo para guiar a difusão
        if self.num_classes is not None and labels is not None:
            t_emb = t_emb + self.class_emb(labels)
            
        h_init = self.init_conv(x) 
        h1 = self.down1_res(h_init, t_emb) 
        h1_pool = self.down1_pool(h1)      
        h2 = self.down2_res(h1_pool, t_emb) 
        h2_pool = self.down2_pool(h2)       
        h3 = self.down3_res(h2_pool, t_emb)
        h3_pool = self.down3_pool(h3)
        
        h_mid = self.mid_res1(h3_pool, t_emb) 
        h_mid = self.mid_attn(h_mid)
        h_mid = self.mid_res2(h_mid, t_emb)   
        
        h_up3 = self.up3_conv(h_mid) 
        h_up3 = torch.cat([h_up3, h3], dim=1) 
        h_up3 = self.up3_res(h_up3, t_emb)
        
        h_up2 = self.up2_conv(h_up3) 
        h_up2 = torch.cat([h_up2, h2], dim=1) 
        h_up2 = self.up2_res(h_up2, t_emb)   
        
        h_up1 = self.up1_conv(h_up2) 
        h_up1 = torch.cat([h_up1, h1], dim=1) 
        h_up1 = self.up1_res(h_up1, t_emb)   
        
        return self.out_conv(h_up1)

class GaussianDiffusion:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cpu'):
        self.num_timesteps = num_timesteps
        self.device = device
        
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.]).to(device), self.alphas_cumprod[:-1]])
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_prod = self._get_index(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha_prod = self._get_index(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        return sqrt_alpha_prod * x_0 + sqrt_one_minus_alpha_prod * noise

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index, labels=None):
        betas_t = self._get_index(self.betas, t, x.shape)
        sqrt_one_minus_alpha_cumprod_t = self._get_index(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = 1. / torch.sqrt(self._get_index(self.alphas, t, x.shape))
        
        # Passamos as labels ao modelo para orientar a remoção do ruído!
        predicted_noise = model(x, t, labels)
        
        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alpha_cumprod_t)
        
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self._get_index(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape, labels=None, initial_noise=None):
        model.eval()
        x = initial_noise.to(self.device) if initial_noise else torch.randn(shape).to(self.device)
            
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc="Denoising", total=self.num_timesteps, leave=False):
            t = torch.full((shape[0],), i, dtype=torch.long).to(self.device)
            x = self.p_sample(model, x, t, i, labels)
        return x

    def _get_index(self, tensor, t, x_shape):
        out = tensor.gather(-1, t)
        return out.view(t.shape[0], *((1,) * (len(x_shape) - 1)))


def train_diffusion(model, loader, schedule, epochs=20, lr=2e-4, device=torch.device('cpu'), save_dir=None, verbose=True):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = {'loss': []}
    model.train()

    if verbose:
        print(f"A iniciar treino Condicional do Diffusion Model por {epochs} épocas...")

    for epoch in range(epochs):
        running = 0.0
        n_batches = 0

        iterator = tqdm(loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False) if verbose else loader

        for x, labels in iterator:
            x = x.to(device)
            labels = labels.to(device)
            batch_size = x.size(0)
            
            # Sorteia tempos aleatórios para cada imagem no batch
            t = torch.randint(0, schedule.num_timesteps, (batch_size,), device=device).long()
            
            # Adiciona ruído
            noise = torch.randn_like(x)
            x_t = schedule.q_sample(x_0=x, t=t, noise=noise)
            
            # Tenta prever o ruído DADO o tempo e a LABEL
            pred_noise = model(x_t, t, labels)
            
            loss = F.mse_loss(pred_noise, noise)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            running += loss.item()
            n_batches += 1

        avg = running / max(n_batches, 1)
        history['loss'].append(avg)
        
        if verbose:
            print(f'Epoch {epoch + 1:02d} | MSE Loss: {avg:.4f}')

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / 'diffusion_unet.pth')
        if verbose:
            print(f"Modelo guardado em {save_dir}")

    return model, history