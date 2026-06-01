import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd

def generate_cvae_samples_grid(model, target_classes, class_to_idx, latent_dim, device, save_path):
    """
    Gera uma grelha 1x4 com amostras sintéticas do cVAE para as 4 classes alvo.
    """
    model.eval()
    
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15, 4))
    
    axes = axes.flatten()

    print("A gerar amostras representativas para as 4 classes alvo...")
    with torch.no_grad():
        for i, class_name in enumerate(target_classes[:4]): # Focado nas 4 classes
            # Preparar label e ruído (z)
            class_idx = class_to_idx[class_name]
            label = torch.tensor([class_idx], dtype=torch.long).to(device)
            z = torch.randn(1, latent_dim).to(device)

            # Gerar a imagem
            fake_img = model.decode(z, label).cpu().squeeze(0)
            
            # Converter de (C, H, W) para (H, W, C)
            img_np = fake_img.numpy().transpose(1, 2, 0)

            # Plotar na grelha
            axes[i].imshow(img_np)
            axes[i].set_title(class_name, fontsize=12, fontweight='bold', pad=10)
            axes[i].axis('off')

    plt.suptitle("Amostras Sintéticas do cVAE (Top 4 Classes Críticas)", fontsize=16, fontweight='bold', y=1.1)
    
    # bbox_inches='tight' remove as margens brancas desnecessárias
    plt.tight_layout()

    # Guardar a imagem garantindo que a pasta existe
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Grelha visual guardada com sucesso em: {save_path}")

def generate_cgan_samples_grid(model, target_classes, class_to_idx, latent_dim, device, save_path):
    """
    Gera uma grelha 1x4 com amostras sintéticas da cDCGAN para as 4 classes alvo.
    """
    model.eval()
    
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15, 4))
    axes = axes.flatten()

    print("A gerar amostras representativas para as 4 classes alvo (cDCGAN)...")
    with torch.no_grad():
        for i, class_name in enumerate(target_classes[:4]):
            class_idx = class_to_idx[class_name]
            label = torch.tensor([class_idx], dtype=torch.long).to(device)
            z = torch.randn(1, latent_dim).to(device)

            # forward em vez de decode()
            fake_img = model(z, label).cpu().squeeze(0)
            
            # Converter de (C, H, W) para (H, W, C)
            # Desnormaliza de [-1, 1] de volta para [0, 1] para o ecrã
            fake_img_norm = (fake_img + 1) / 2.0 
            img_np = fake_img_norm.numpy().transpose(1, 2, 0)

            axes[i].imshow(img_np)
            axes[i].set_title(class_name, fontsize=12, fontweight='bold', pad=10)
            axes[i].axis('off')

    plt.suptitle("Amostras Sintéticas da cDCGAN (Top 4 Classes Críticas)", fontsize=16, fontweight='bold', y=1.1)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Grelha visual guardada com sucesso em: {save_path}")

def generate_ddpm_samples_grid(model, schedule, target_classes, class_to_idx, image_size, device, save_path):
    """
    Gera uma grelha 1x4 com amostras sintéticas do DDPM para as 4 classes alvo.
    """
    import os
    import matplotlib.pyplot as plt
    import torch

    print("A iniciar o processo de Denoising Reverso SOTA (1000 passos)...")
    model.eval()

    # Garantir que só processamos no máximo 4 classes para a grelha
    classes_to_plot = target_classes[:4]
    
    fig, axes = plt.subplots(nrows=1, ncols=len(classes_to_plot), figsize=(15, 4))
    if len(classes_to_plot) == 1:
        axes = [axes] # Proteção caso seja só 1 classe
    else:
        axes = axes.flatten()

    with torch.no_grad():
        for i, class_name in enumerate(classes_to_plot):
            class_idx = class_to_idx[class_name]
            
            # 1. Prepara a Label Condicional
            label = torch.tensor([class_idx], dtype=torch.long).to(device)
            
            # 2. Passamos a Label para a UNet saber o que esculpir a partir do ruído
            fake_img = schedule.p_sample_loop(
                model=model, 
                shape=(1, 3, image_size, image_size), 
                labels=label
            ).cpu().squeeze(0)
            
            # 3. Desnormaliza de [-1, 1] de volta para [0, 1] para visualização
            fake_img_norm = (fake_img + 1) / 2.0 
            fake_img_norm = torch.clamp(fake_img_norm, 0, 1) # Previne pequenos artefactos visuais
            
            img_np = fake_img_norm.numpy().transpose(1, 2, 0)
            
            axes[i].imshow(img_np)
            axes[i].set_title(class_name, fontsize=12, fontweight='bold', pad=10)
            axes[i].axis('off')

    plt.suptitle("Amostras Condicionais do DDPM (Top 4 Classes Críticas)", fontsize=16, fontweight='bold', y=1.1)
    plt.tight_layout()

    # Guardar a grelha visual
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Grelha visual guardada com sucesso em: {save_path}")

def plot_augmentation_comparison(baseline_csv_path, new_metrics_json, target_classes, save_dir, aug_type="cVAE"):
    """
    Gera gráficos de barras separados e esteticamente otimizados comparando a Baseline com o Modelo Aumentado.
    """
    # 1. Carregar Baseline
    if not os.path.exists(baseline_csv_path):
        print(f"Aviso: CSV da baseline não encontrado em {baseline_csv_path}. Gráfico abortado.")
        return
        
    df_base = pd.read_csv(baseline_csv_path)
    df_base = df_base.set_index('Classe')
    
    # 2. Preparar as listas para os gráficos
    classes_encontradas = []
    base_prec, base_rec, base_f1 = [], [], []
    aug_prec, aug_rec, aug_f1 = [], [], []
    
    for cls in target_classes:
        if cls in df_base.index and cls in new_metrics_json:
            classes_encontradas.append(cls)
            
            # Dados Antigos (Baseline)
            base_prec.append(df_base.loc[cls, 'Precision'])
            base_rec.append(df_base.loc[cls, 'Recall'])
            base_f1.append(df_base.loc[cls, 'F1-Score'])
            
            # Dados Novos (Aumentados)
            aug_prec.append(new_metrics_json[cls]['precision'])
            aug_rec.append(new_metrics_json[cls]['recall'])
            aug_f1.append(new_metrics_json[cls]['f1-score'])

    if not classes_encontradas:
        print("Aviso: Nenhuma classe alvo encontrada para comparação.")
        return

    # 3. Definições de Estilo Comuns
    x = np.arange(len(classes_encontradas))
    width = 0.35
    
    metricas = [
        ('Precision', base_prec, aug_prec),
        ('Recall', base_rec, aug_rec),
        ('F1-Score', base_f1, aug_f1)
    ]
    
    # 4. Gerar um gráfico independente para cada métrica
    for titulo, base_vals, aug_vals in metricas:
        fig, ax = plt.subplots(figsize=(8, 6)) # Tamanho individual
        
        # Adicionar grelha subtil no eixo Y e enviá-la para o fundo (zorder=0)
        ax.yaxis.grid(True, linestyle='--', alpha=0.6, zorder=0)
        
        # Desenhar as barras por cima da grelha (zorder=3)
        rects1 = ax.bar(x - width/2, base_vals, width, label='Baseline (Apenas Reais)', color='#95a5a6', zorder=3)
        rects2 = ax.bar(x + width/2, aug_vals, width, label=f'Augmented (+{aug_type})', color='#2980b9', zorder=3)
        
        # Estilização de Texto e Eixos
        ax.set_ylabel('Score', fontweight='bold', fontsize=12)
        ax.set_title(f'Impacto nas Classes Críticas: {titulo}', fontweight='bold', fontsize=14, pad=15)
        ax.set_xticks(x)
        
        # Rotação das labels baseada no tamanho das palavras
        ax.set_xticklabels(classes_encontradas, rotation=0, fontsize=11, fontweight='bold')
        
        # Dar espaço extra no topo para os números não cortarem
        ax.set_ylim(0, 1.2) 
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]) # Forçar escala limpa até 1.0
        
        # Remover bordas desnecessárias
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Adicionar os valores numéricos exatamente em cima de cada barra
        for rect in rects1 + rects2:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5),  # 5 pontos de margem vertical
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2c3e50')
        
        # Legenda fora da área do gráfico (em baixo, centrada)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=11)
        
        plt.tight_layout()
        
        # Salvar o gráfico individual
        save_path = os.path.join(save_dir, f'{aug_type.lower()}_comparison_{titulo.lower()}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Gráfico '{titulo}' salvo em: {save_path}")