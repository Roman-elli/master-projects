import torch
import torchvision.transforms as transforms
from torch.utils.data import WeightedRandomSampler
import config as cfg

def apply_data_augmentation(df_train):
    # 1. Transformações de Treino SOTA (com Random Erasing para robustez)
    train_transform = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=45),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0) 
    ])

    # 2. Transformações de Validação (Limpo, sem ruído)
    val_transform = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    # 3. Weighted Sampler para o desbalanceamento
    
    # Contamos as classes, mas deixamos como uma "Series" do Pandas 
    # (assim ele guarda a ligação: 'NomeDaClasse' -> Quantidade)
    class_counts = df_train['label'].value_counts()
    
    # Calculamos o peso
    class_weights = 1.0 / class_counts
    
    # O .map() olha para a coluna 'label' e vai buscar o peso exato à tabela class_weights, independentemente de ser texto ou número!
    sample_weights = df_train['label'].map(class_weights).values
    
    # Converte para Tensor do PyTorch
    sample_weights = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=cfg.DATA_AUG_SIZE, 
        replacement=True 
    )
    return train_transform, val_transform, sampler