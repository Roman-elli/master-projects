import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm
from torchvision.utils import save_image
from pathlib import Path
from tqdm import tqdm
from models.Variational_Autoenconders  import vae_loss

###
# VAE
def evaluate_vae(model, loader, device, beta=0.7):
    model.eval()
    tl, tr, tk, tm, ta, n = 0.0, 0.0, 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for x, _, _ in loader:
            x = x.to(device)
            xhat, mu, logvar = model(x)
            b = x.size(0)
            loss, recon, kl = vae_loss(xhat, x, mu, logvar, beta=beta)
            
            tl += loss.item() * b
            tr += recon.item() * b
            tk += kl.item() * b
            tm += F.mse_loss(xhat, x, reduction='sum').item()
            ta += F.l1_loss(xhat, x, reduction='sum').item()
            n += b

    numel = x[0].numel()
    return {'loss': tl/n, 'recon_bce': tr/n, 'kl': tk/n, 'mse': tm/(n*numel), 'mae': ta/(n*numel)}

def interpolacao_latente_vae(model, img_a, img_b, device, steps=10, save_path=None):
    """Cria transição suave entre a img_a e a img_b no espaço latente."""
    model.eval()
    with torch.no_grad():
        img_a = img_a.unsqueeze(0).to(device)
        img_b = img_b.unsqueeze(0).to(device)
        
        mu_a, logvar_a = model.encode(img_a)
        z_a = model.reparameterize(mu_a, logvar_a)
        
        mu_b, logvar_b = model.encode(img_b)
        z_b = model.reparameterize(mu_b, logvar_b)
        
        alphas = np.linspace(0, 1, steps)
        imagens_geradas = []
        
        for alpha in alphas:
            z_interp = (1.0 - alpha) * z_a + alpha * z_b
            img_gerada = model.decode(z_interp)
            imagens_geradas.append(img_gerada.squeeze(0).cpu().permute(1, 2, 0).numpy())
            
    fig, axes = plt.subplots(1, steps, figsize=(15, 3))
    fig.suptitle("Interpolação no Espaço Latente (A -> B)", fontsize=16)
    for i, ax in enumerate(axes):
        ax.imshow(imagens_geradas[i])
        ax.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def gerar_amostras_fid_vae(model, num_samples, batch_size, device, output_dir):
    """Gera amostras e guarda numa pasta para cálculo do FID/KID."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    amostras_geradas = 0
    print(f"\nA gerar {num_samples} imagens para FID em: {output_dir}")
    
    with torch.no_grad(), tqdm(total=num_samples) as pbar:
        while amostras_geradas < num_samples:
            n_gerar = min(batch_size, num_samples - amostras_geradas)
            batch_imagens = model.sample(n_gerar, device)
            
            for i in range(n_gerar):
                nome_ficheiro = output_dir / f"fake_{amostras_geradas:05d}.png"
                save_image(batch_imagens[i], nome_ficheiro)
                amostras_geradas += 1
                pbar.update(1)
###

###
# GANs
@torch.no_grad()
def interpolacao_latente_gan(generator, latent_dim, steps=8, device='cpu', save_path=None):
    """Faz uma interpolação linear entre dois pontos no espaço latente."""
    generator.eval()
    
    # TODO START SOLVED
    # 1) sample z0 and z1
    z0 = torch.randn(1, latent_dim, device=device)
    z1 = torch.randn(1, latent_dim, device=device)
    
    # 2) interpolate with alpha in [0, 1]
    alphas = np.linspace(0, 1, steps)
    
    # 3) generate fake images (Iterando pelos alphas)
    imagens_geradas = []
    for alpha in alphas:
        z_interp = (1.0 - alpha) * z0 + alpha * z1
        fake = generator(z_interp)
        
        # Desnormalizar
        fake = (fake + 1) / 2.0
        imagens_geradas.append(fake.squeeze(0).cpu().permute(1, 2, 0).numpy())
    # TODO END

    fig, axes = plt.subplots(1, steps, figsize=(15, 3))
    fig.suptitle("Interpolação no Espaço Latente DCGAN (A -> B)", fontsize=16)
    for i, ax in enumerate(axes):
        ax.imshow(imagens_geradas[i])
        ax.axis('off')
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Interpolação guardada em: {save_path}")
    plt.show()

def gerar_amostras_fid_gan(generator, num_samples, batch_size, latent_dim, device, output_dir):
    """Gera amostras e guarda numa pasta para cálculo do FID/KID."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generator.eval()
    amostras_geradas = 0
    print(f"\n🚀 A preparar {num_samples} amostras FID para a GAN em: {output_dir}")
    
    with torch.no_grad(), tqdm(total=num_samples) as pbar:
        while amostras_geradas < num_samples:
            n_gerar = min(batch_size, num_samples - amostras_geradas)
            z = torch.randn(n_gerar, latent_dim, device=device)
            fake_images = generator(z)
            
            # Desnormalizar para guardar corretamente
            fake_images = (fake_images + 1) / 2.0 
            
            for i in range(n_gerar):
                nome_ficheiro = output_dir / f"fake_gan_{amostras_geradas:05d}.png"
                save_image(fake_images[i], nome_ficheiro)
                amostras_geradas += 1
                pbar.update(1)

@torch.no_grad()
def run_inference(generator, latent_dim, num_samples=16, seed=123, device='cpu', save_path=None):
    """Gera uma grelha de imagens a partir de ruído aleatório."""
    generator.eval()

    # 1) set torch seed
    torch.manual_seed(seed)
    
    # 2) sample z
    z = torch.randn(num_samples, latent_dim, device=device)
    
    # 3) generate fake images
    fake = generator(z)

    # Desnormalizar para plotar cores reais
    fake = (fake + 1) / 2.0 
    
    # Plot e Save
    fake_np = fake.cpu().permute(0, 2, 3, 1).numpy()
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle("Artes Geradas do Zero (DCGAN)", fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            ax.imshow(fake_np[i])
        ax.axis('off')
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Grelha guardada em: {save_path}")
    plt.show()
###

###
# Diffusion PixelUnit
def slerp(val, low, high):
    """Interpolação Linear Esférica para tensores."""
    low_norm = low / torch.norm(low)
    high_norm = high / torch.norm(high)
    omega = torch.acos(torch.clamp(torch.sum(low_norm * high_norm), -1, 1))
    so = torch.sin(omega)
    if so == 0:
        return (1.0 - val) * low + val * high # Fallback se os pontos forem iguais
    return (torch.sin((1.0 - val) * omega) / so * low) + (torch.sin(val * omega) / so * high)

def interpolacao_latente_diffusion(diffusion_schedule, model, device, steps=8, save_path=None):
    """Viaja de forma suave entre dois pontos de ruído no espaço da Difusão."""
    model.eval()
    
    # 1. Gerar os nossos dois pontos de partida e chegada (Ruído Puro)
    noise_A = torch.randn(1, 3, 32, 32).to(device)
    noise_B = torch.randn(1, 3, 32, 32).to(device)

    # 2. Calcular o Slerp para todos os passos intermédios
    alphas = torch.linspace(0, 1, steps).to(device)
    interpolated_noises = [slerp(alpha, noise_A, noise_B) for alpha in alphas]
    
    # 3. Juntar todos os ruídos num único batch para a rede processar de uma vez: [steps, 3, 32, 32]
    batch_noise = torch.cat(interpolated_noises, dim=0)

    # 4. Passar o lote de ruído pela U-Net (O passo mágico)
    sampled_images = diffusion_schedule.p_sample_loop(model, batch_noise.shape, initial_noise=batch_noise)

    # 5. Desnormalizar [-1, 1] -> [0, 1] e mostrar
    # Nota: Podes importar e usar a tua função denorm do guião aqui
    sampled_images = ((sampled_images + 1.0) / 2.0).clamp(0, 1)

    # 6. Criar o plot
    fig, axes = plt.subplots(1, steps, figsize=(steps * 2, 2))
    for i in range(steps):
        img = sampled_images[i].permute(1, 2, 0).cpu().numpy()
        axes[i].imshow(img)
        axes[i].axis('off')
        if i == 0: axes[i].set_title("Ruído A")
        if i == steps - 1: axes[i].set_title("Ruído B")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def gerar_amostras_fid_diffusion(diffusion_schedule, model, num_samples, batch_size, device, output_dir):
    """Gera N imagens a partir do modelo de Difusão e guarda na pasta."""
    output_dir.mkdir(parents=True, exist_ok=True)
    imagens_geradas = 0
    
    print(f"\nA extrair {num_samples} imagens para o FID...")
    
    with tqdm(total=num_samples) as pbar:
        while imagens_geradas < num_samples:
            # Calcular o tamanho deste batch para não gerar imagens a mais no último loop
            current_batch_size = min(batch_size, num_samples - imagens_geradas)
            shape = (current_batch_size, 3, 32, 32)
            
            # 1. Gerar imagens da U-Net
            samples = diffusion_schedule.p_sample_loop(model, shape)
            
            # 2. Desnormalizar para imagem visível
            samples = ((samples + 1.0) / 2.0).clamp(0, 1)
            
            # 3. Guardar como PNG
            for i in range(current_batch_size):
                img_tensor = samples[i]
                img_pil = transforms.ToPILImage()(img_tensor.cpu())
                nome_ficheiro = output_dir / f"diff_{imagens_geradas:05d}.png"
                img_pil.save(nome_ficheiro)
                
                imagens_geradas += 1
                pbar.update(1)
                
    print("Concluído! As imagens para o FID estão guardadas.")
###

# Funções gerais
def mostrar_reconstrucoes(modelo, test_loader, device, num_imagens=8, save_path=None):
    """
    Compara imagens reais do test_loader com as reconstruções geradas pelo VAE.
    """
    modelo.eval()
    
    # Extrair o primeiro batch (Imagens estão no índice 0)
    batch = next(iter(test_loader))
    imagens_reais = batch[0][:num_imagens].to(device)
    
    with torch.no_grad():
        imagens_geradas, _, _ = modelo(imagens_reais)
    
    # Mover para CPU e ajustar os eixos para o Matplotlib
    imagens_reais = imagens_reais.cpu().permute(0, 2, 3, 1).numpy()
    imagens_geradas = imagens_geradas.cpu().permute(0, 2, 3, 1).numpy()
    
    fig, axes = plt.subplots(2, num_imagens, figsize=(15, 4))
    fig.suptitle("Original (Cima) vs Reconstrução VAE (Baixo)", fontsize=16)
    
    for i in range(num_imagens):
        axes[0, i].imshow(imagens_reais[i])
        axes[0, i].axis('off')
        
        axes[1, i].imshow(imagens_geradas[i])
        axes[1, i].axis('off')
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Reconstruções guardadas em: {save_path}")
        
    plt.show()

def gerar_grelha_amostras(modelo, device, num_samples=16, save_path=None):
    """
    Gera novas imagens a partir de ruído aleatório no espaço latente e mostra-as numa grelha.
    """
    modelo.eval()
    with torch.no_grad():
        amostras = modelo.sample(num_samples, device)
        
        # Converter para visualização no Matplotlib
        amostras = amostras.cpu().permute(0, 2, 3, 1).numpy()
        
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle("Artes Geradas do Zero (Random Latent Sampling)", fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        if i < num_samples: # Proteção caso peçam um número diferente de 16
            ax.imshow(amostras[i])
        ax.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Grelha de amostras guardada em: {save_path}")
        
    plt.show()

def extrair_amostras_reais_fid(loader, num_samples, output_dir):
    """Extrai imagens reais do DataLoader para uma pasta (para o cálculo do FID)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    amostras_salvas = 0
    print(f"\nA extrair {num_samples} imagens REAIS para FID em: {output_dir}")
    
    with torch.no_grad(), tqdm(total=num_samples) as pbar:
        for x, _, _ in loader:
            batch_size = x.size(0)
            for i in range(batch_size):
                if amostras_salvas >= num_samples:
                    print("Extração de imagens reais concluída!")
                    return
                
                nome_ficheiro = output_dir / f"real_{amostras_salvas:05d}.png"
                save_image(x[i], nome_ficheiro)
                amostras_salvas += 1
                pbar.update(1)