import matplotlib.pyplot as plt
import os
import torch
import config as cfg

def print_images(dataset, dataloader):
    images, labels = next(iter(dataloader))

    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    axes = axes.flatten()

    for i in range(len(images)):
        img = images[i].permute(1, 2, 0).numpy()
        label_idx = labels[i].item()
        label_name = dataset.classes[label_idx]

        axes[i].imshow(img)
        axes[i].set_title(label_name.capitalize(), fontsize=8)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
    
def save_best_model(model_state_dict, save_dir, file_name="best.pt"):
    """Cria os diretórios necessários e guarda os pesos do modelo."""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file_name)
    torch.save(model_state_dict, save_path)
    print(f"\nMelhor modelo guardado com sucesso em: {save_path}")
    
def save_acc_graph(train_losses, val_losses, file_name, folder_name, model_type=""):
    epochs_range = range(1, len(train_losses) + 1)
        
    os.makedirs(file_name, exist_ok=True)
    
    plt.figure(figsize=(10, 6))

    plt.plot(epochs_range, train_losses, marker='o', linestyle='-', color='blue', label='Treino (Train Loss)')
    plt.plot(epochs_range, val_losses, marker='s', linestyle='--', color='red', label='Validação (Valid Loss)')

    plt.title(f"Evolução da Acc - {model_type}\n{folder_name}", fontsize=10)
    plt.xlabel('Épocas')
    plt.ylabel('ACCuracy')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()

    file_path = file_name / "acc_curve.png"
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado com sucesso...")
    
    plt.close()