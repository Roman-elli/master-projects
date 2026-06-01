import optuna
import torch
import torch.optim as optim
import lpips
from torchmetrics.image.kid import KernelInceptionDistance
import generative.c_vae as cvae
import generative.dcgan as gan
import generative.diffusion as diff

class CVAEObjective:
    def __init__(self, train_loader, val_loader, device, n_classes, img_size, epochs_trial=10):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.n_classes = n_classes
        self.img_size = img_size
        self.epochs_trial = epochs_trial
        
        # Carregamos a VGG apenas uma vez no __init__.
        print("Carregando modelo VGG (LPIPS) para o Optuna...")
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').to(self.device)
        self.loss_fn_vgg.eval() # Garante que os pesos da VGG estão congelados

    def __call__(self, trial):
        # 1. Sugestões de hiperparâmetros
        latent_dim = trial.suggest_categorical('latent_dim', [64, 128, 256, 512])
        lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
        
        # Novos hiperparâmetros para baixar o FID (Perceptual vs Diversidade)
        beta = trial.suggest_float('beta', 0.01, 0.2, log=True)
        perceptual_weight = trial.suggest_float('perceptual_weight', 0.5, 3.0)

        # 2. Instanciar o modelo
        model = cvae.ConditionalVAE(
            num_classes=self.n_classes,
            latent_dim=latent_dim,
            img_channels=3,
            img_size=self.img_size
        ).to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=lr)

        # 3. Treino simplificado para a tentativa (Trial)
        for epoch in range(self.epochs_trial):
            model.train()
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                recon, mu, logvar = model(inputs, labels)
                
                # Chamamos a nova função de loss passando os pesos e a VGG
                loss, _, _, _ = cvae.vae_loss_function(
                    recon, inputs, mu, logvar, 
                    loss_fn_vgg=self.loss_fn_vgg, 
                    beta=beta, 
                    perceptual_weight=perceptual_weight
                )
                
                loss.backward()
                optimizer.step()

            # 4. Validação
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    recon, mu, logvar = model(inputs, labels)
                    
                    loss, _, _, _ = cvae.vae_loss_function(
                        recon, inputs, mu, logvar, 
                        loss_fn_vgg=self.loss_fn_vgg, 
                        beta=beta, 
                        perceptual_weight=perceptual_weight
                    )
                    # Multiplicamos pelo tamanho real do batch para evitar viés na média final
                    val_loss += loss.item() * inputs.size(0)

            avg_val_loss = val_loss / len(self.val_loader.dataset)

            # 5. Reportar ao Optuna e verificar poda (Pruning)
            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return avg_val_loss
    
class CGANObjective:
    def __init__(self, train_loader, val_loader, device, n_classes, img_size, epochs_trial=10):
        self.train_loader = train_loader
        self.val_loader = val_loader # Usado para extrair imagens reais para o KID
        self.device = device
        self.n_classes = n_classes
        self.img_size = img_size
        self.epochs_trial = epochs_trial

    def __call__(self, trial):
        # 1. Hiperparâmetros a testar 
        latent_dim = trial.suggest_categorical('latent_dim', [64, 100, 128])
        lr = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
        embed_size = trial.suggest_categorical('embed_size', [50, 100, 128])

        # 2. Instanciar Modelos
        generator = gan.cDCGenerator(self.n_classes, latent_dim, 3, embed_size=embed_size).to(self.device)
        discriminator = gan.cDCDiscriminator(self.n_classes, 3, img_size=self.img_size).to(self.device)
        generator.apply(gan.init_dcgan_weights)
        discriminator.apply(gan.init_dcgan_weights)

        # 3. Treinar
        gan.train_cgan(
            generator, discriminator, self.train_loader, 
            latent_dim, epochs=self.epochs_trial, lr=lr, device=self.device
        )

       # 4. Avaliação (KID Score) em vez de Loss
        kid_metric = KernelInceptionDistance(feature=2048, subset_size=50, normalize=True).to(self.device)
        generator.eval()
        
        with torch.no_grad():
            n_amostras = 0
            for real_imgs, labels in self.val_loader:
                real_imgs, labels = real_imgs.to(self.device), labels.to(self.device)
                
                z = torch.randn(real_imgs.size(0), latent_dim).to(self.device)
                fake_imgs = generator(z, labels)
                
                kid_metric.update(real_imgs, real=True)
                kid_metric.update(fake_imgs, real=False)
                
                n_amostras += real_imgs.size(0)
                if n_amostras >= 64: 
                    break
            
            kid_mean, _ = kid_metric.compute()

        return kid_mean.item()
    

class DDPMObjective:
    def __init__(self, train_loader, device, n_classes):
        """
        Guarda as variáveis de ambiente para que o Optuna as possa usar
        durante os trials.
        """
        self.train_loader = train_loader
        self.device = device
        self.n_classes = n_classes

    def __call__(self, trial):
        # 1. Hiperparâmetros para testar para DDPM
        lr = trial.suggest_float('lr', 1e-4, 5e-4, log=True)
        model_channels = trial.suggest_categorical('model_channels', [32, 64]) 

        # 2. Instanciar a Rede (usando self.device e self.n_classes)
        unet_trial = diff.PixelUNet(
            in_channels=3, 
            model_channels=model_channels, 
            num_classes=self.n_classes
        ).to(self.device)
        
        schedule_trial = diff.GaussianDiffusion(num_timesteps=1000, device=self.device)

        # 3. Treino rápido de 10 épocas 
        _, history = diff.train_diffusion(
            model=unet_trial, 
            loader=self.train_loader, 
            schedule=schedule_trial, 
            epochs=10, 
            lr=lr, 
            device=self.device, 
            verbose=False # Silenciamos os prints
        )

        # 4. O Optuna vai escolher os parâmetros que tiverem a menor MSE Loss na última época
        return history['loss'][-1]
