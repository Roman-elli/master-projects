import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm


class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal Position Embedding for time steps.
    """
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
        # Handle odd dimension by padding if necessary, but dim should be even
        return emb

class ResnetBlock(nn.Module):
    """
    Residual Block with Time Embedding projection.
    Supports channel dimension changes with short-cut projection.
    """
    def __init__(self, dim, time_emb_dim, out_dim=None):
        super().__init__()
        self.out_dim = out_dim or dim
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, self.out_dim)
        )
        self.conv1 = nn.Conv2d(dim, self.out_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(self.out_dim, self.out_dim, 3, padding=1)
        # GroupNorm tends to work better for diffusion than BatchNorm
        self.norm1 = nn.GroupNorm(4, dim)
        self.norm2 = nn.GroupNorm(4, self.out_dim)
        self.act = nn.SiLU()
        
        # Shortcut for residual if dims don't match
        self.shortcut = nn.Conv2d(dim, self.out_dim, 1) if dim != self.out_dim else nn.Identity()

    def forward(self, x, time_emb):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        # Add time embedding
        time_emb = self.mlp(time_emb)
        # Expand time_emb to match spatial dimensions [B, C, 1, 1]
        h = h + time_emb[:, :, None, None]
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        return self.shortcut(x) + h
    
class PixelUNet(nn.Module):

    # Alteração: in_channels muda para 3 (RGB) por padrão
    def __init__(self, in_channels=3, model_channels=64): 
        super().__init__()
        # Time Embedding
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(model_channels),
            nn.Linear(model_channels, model_channels * 4),
            nn.SiLU(),
            nn.Linear(model_channels * 4, model_channels * 4),
        )
        
        time_dim = model_channels * 4
        
        # Initial Conv
        self.init_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Down 1: 32x32 -> 16x16
        self.down1_res = ResnetBlock(model_channels, time_dim)
        self.down1_pool = nn.Conv2d(model_channels, model_channels, 3, stride=2, padding=1)
        
        # Down 2: 16x16 -> 8x8
        self.down2_res = ResnetBlock(model_channels, time_dim, out_dim=model_channels * 2)
        self.down2_pool = nn.Conv2d(model_channels * 2, model_channels * 2, 3, stride=2, padding=1)
        
        # Middle (8x8)
        self.mid_res1 = ResnetBlock(model_channels * 2, time_dim)
        self.mid_res2 = ResnetBlock(model_channels * 2, time_dim)
        
        # Up 2: 8x8 -> 16x16
        self.up2_conv = nn.ConvTranspose2d(model_channels * 2, model_channels, 4, stride=2, padding=1) 
        self.up2_res = ResnetBlock(model_channels * 3, time_dim, out_dim=model_channels)
        
        # Up 1: 16x16 -> 32x32
        self.up1_conv = nn.ConvTranspose2d(model_channels, model_channels, 4, stride=2, padding=1) 
        self.up1_res = ResnetBlock(model_channels * 2, time_dim, out_dim=model_channels)
        
        # Out: Retorna aos 3 canais (RGB)
        self.out_conv = nn.Conv2d(model_channels, in_channels, 3, padding=1)
        
    def forward(self, x, t):
        # A lógica do forward mantém-se EXATAMENTE igual ao guião!
        t_emb = self.time_embed(t)
        
        h_init = self.init_conv(x) 
        
        h1 = self.down1_res(h_init, t_emb) 
        h1_pool = self.down1_pool(h1)      
        
        h2 = self.down2_res(h1_pool, t_emb) 
        h2_pool = self.down2_pool(h2)       
        
        h_mid = self.mid_res1(h2_pool, t_emb) 
        h_mid = self.mid_res2(h_mid, t_emb)   
        
        h_up2 = self.up2_conv(h_mid) 
        h_up2 = torch.cat([h_up2, h2], dim=1) 
        h_up2 = self.up2_res(h_up2, t_emb)   
        
        h_up1 = self.up1_conv(h_up2) 
        h_up1 = torch.cat([h_up1, h1], dim=1) 
        h_up1 = self.up1_res(h_up1, t_emb)   
        
        return self.out_conv(h_up1)
    
def train_diffusion(model, loader, schedule, epochs=20, lr=2e-4, encode_fn=None, device=torch.device('cpu')):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    model.train()

    for epoch in range(epochs):
        running = 0.0
        n_batches = 0

        for x, _, _ in tqdm(loader, desc=f'Diff epoch {epoch + 1}/{epochs}', leave=False):
            x = x.to(device)
            if encode_fn is not None:
                with torch.no_grad():
                    x = encode_fn(x)

            # TODO START - Diffusion training step
            # 1) Determine batch size and sample random diffusion steps.
            batch_size = x.size(0)
            t = torch.randint(0, schedule.num_timesteps, (batch_size,), device=device).long()
            
            # 2) Use q_sample to obtain x_t and the target noise.
            noise = torch.randn_like(x)
            x_t = schedule.q_sample(x_0=x, t=t, noise=noise)
            
            # 3) Predict the noise with model(x_t, t).
            pred_noise = model(x_t, t)
            
            # 4) Compute the MSE loss against the sampled noise.
            loss = F.mse_loss(pred_noise, noise)
            
            # 5) Zero gradients, backpropagate, and step the optimizer.
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            # 6) Update the running loss and batch counter.
            running += loss.item()
            n_batches += 1
            # TODO END

            if loss is None:
                raise NotImplementedError('Implement diffusion training step in train_diffusion().')

        avg = running / max(n_batches, 1)
        history.append(avg)
        print(f'Diff epoch {epoch + 1:02d}/{epochs} | loss: {avg:.4f}')

    return history

class GaussianDiffusion:
    """
    DDPM (Denoising Diffusion Probabilistic Models) Scheduler.
    """
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cpu'):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Linear beta scheduler (can be swapped for Cosine for better efficiency)
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # alphas_cumprod_prev starts with 1.0 (no noise)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.]).to(device), self.alphas_cumprod[:-1]])
        
        # Calculations for diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        # posterior_mean = posterior_mean_coef1 * x_0 + posterior_mean_coef2 * x_t
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        
        # posterior_variance
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion process: Add noise to x_0 at step t.
        q(x_t | x_0) = N(x_t; sqrt(alpha_prod)*x_0, (1-alpha_prod)*I)
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_prod = self._get_index(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha_prod = self._get_index(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        return sqrt_alpha_prod * x_0 + sqrt_one_minus_alpha_prod * noise

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        """
        Reverse diffusion step: Sample x_{t-1} given x_t and the model.
        """
        betas_t = self._get_index(self.betas, t, x.shape)
        sqrt_one_minus_alpha_cumprod_t = self._get_index(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = 1. / torch.sqrt(self._get_index(self.alphas, t, x.shape))
        
        # Predict noise
        predicted_noise = model(x, t)
        
        # Compute mean
        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alpha_cumprod_t)
        
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self._get_index(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Clip step to be safe, or just add variance
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape, initial_noise=None):
        """
        Sample all steps from pure noise to reconstruct an image in pixel/latent space.
        Se `initial_noise` for fornecido, a geração começa a partir dele (útil para interpolação).
        """
        model.eval()
        
        # Se enviarmos o Slerp de fora, usamos
        if initial_noise is not None:
            x = initial_noise.to(self.device)
        else:
            x = torch.randn(shape).to(self.device)
            
        # Reverse loop from T-1 back to 0
        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((shape[0],), i, dtype=torch.long).to(self.device)
            x = self.p_sample(model, x, t, i)
        return x

    def _get_index(self, tensor, t, x_shape):
        """Get value at index t and expand to match x_shape."""
        out = tensor.gather(-1, t)
        return out.view(t.shape[0], *((1,) * (len(x_shape) - 1)))

# Versao stor

# --- PIXEL UNET ---

# class PixelUNet(nn.Module):
#     """
#     Standard UNet for Diffusion on image space.
#     Fits 28x28 MNIST images.
#     """
#     def __init__(self, in_channels=1, model_channels=64):
#         super().__init__()
#         # Time Embedding
#         self.time_embed = nn.Sequential(
#             SinusoidalPosEmb(model_channels),
#             nn.Linear(model_channels, model_channels * 4),
#             nn.SiLU(),
#             nn.Linear(model_channels * 4, model_channels * 4),
#         )
        
#         time_dim = model_channels * 4
        
#         # Initial Conv
#         self.init_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
#         # Down 1: 28 -> 14
#         self.down1_res = ResnetBlock(model_channels, time_dim)
#         self.down1_pool = nn.Conv2d(model_channels, model_channels, 3, stride=2, padding=1)
        
#         # Down 2: 14 -> 7
#         self.down2_res = ResnetBlock(model_channels, time_dim, out_dim=model_channels * 2)
#         self.down2_pool = nn.Conv2d(model_channels * 2, model_channels * 2, 3, stride=2, padding=1)
        
#         # Middle
#         self.mid_res1 = ResnetBlock(model_channels * 2, time_dim)
#         self.mid_res2 = ResnetBlock(model_channels * 2, time_dim)
        
#         # Up 2: 7 -> 14
#         self.up2_conv = nn.ConvTranspose2d(model_channels * 2, model_channels, 4, stride=2, padding=1) # 7 -> 14
#         # Skip connection from down2_res is model_channels * 2
#         # After concat: model_channels (up) + model_channels*2 (skip) = model_channels * 3
#         self.up2_res = ResnetBlock(model_channels * 3, time_dim, out_dim=model_channels)
        
#         # Up 1: 14 -> 28
#         self.up1_conv = nn.ConvTranspose2d(model_channels, model_channels, 4, stride=2, padding=1) # 14 -> 28
#         # Skip connection from down1_res is model_channels
#         # After concat: model_channels (up) + model_channels (skip) = model_channels * 2
#         self.up1_res = ResnetBlock(model_channels * 2, time_dim, out_dim=model_channels)
        
#         # Out
#         self.out_conv = nn.Conv2d(model_channels, in_channels, 3, padding=1)
        
#     def forward(self, x, t):
#         t_emb = self.time_embed(t)
        
#         # Initial
#         h_init = self.init_conv(x) # [B, C, 28, 28]
        
#         # Down 1
#         h1 = self.down1_res(h_init, t_emb) # [B, C, 28, 28]
#         h1_pool = self.down1_pool(h1)      # [B, C, 14, 14]
        
#         # Down 2
#         h2 = self.down2_res(h1_pool, t_emb) # [B, 2C, 14, 14]
#         h2_pool = self.down2_pool(h2)       # [B, 2C, 7, 7]
        
#         # Middle
#         h_mid = self.mid_res1(h2_pool, t_emb) # [B, 2C, 7, 7]
#         h_mid = self.mid_res2(h_mid, t_emb)   # [B, 2C, 7, 7]
        
#         # Up 2
#         h_up2 = self.up2_conv(h_mid) # [B, C, 14, 14]
#         h_up2 = torch.cat([h_up2, h2], dim=1) # [B, 3C, 14, 14]
#         h_up2 = self.up2_res(h_up2, t_emb)   # [B, C, 14, 14]
        
#         # Up 1
#         h_up1 = self.up1_conv(h_up2) # [B, C, 28, 28]
#         h_up1 = torch.cat([h_up1, h1], dim=1) # [B, 2C, 28, 28]
#         h_up1 = self.up1_res(h_up1, t_emb)   # [B, C, 28, 28]
        
#         # Out
#         return self.out_conv(h_up1)
