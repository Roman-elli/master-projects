import torch.utils.data as data
import torch
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class ButterflyDataset(data.Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.img_labels = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

        self.classes = sorted(self.img_labels['label'].unique())
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx]['filename']
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        label_name = self.img_labels.iloc[idx]['label']
        label_idx = self.class_to_idx[label_name]
        label = torch.tensor(label_idx, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label
    
def get_stratified_splits(df, test_size, val_size, random_state=42):
    """
    Divide o dataframe em Train, Val e Test de forma estratificada.
    """
    # Primeiro separamos o conjunto de Teste
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df['label'], 
        random_state=random_state
    )
    
    # Calculamos a proporção correta do Validation em relação ao que sobrou (Train+Val)
    val_ratio = val_size / (1.0 - test_size)
    
    # Agora separamos Train e Val
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=val_ratio, 
        stratify=train_val_df['label'], 
        random_state=random_state
    )
    
    return train_df, val_df, test_df

def analyze_dataset(df, dataset_name="Train", save_dir=None):
    """
    Analisa o balanceamento de classes num DataFrame, imprime estatísticas 
    e guarda o gráfico de distribuição.
    """
    class_counts = df['label'].value_counts()
    
    print(f"--- Análise do Dataset: {dataset_name} ---")
    print(f"Total de classes: {len(class_counts)}")
    print(f"Total de amostras: {len(df)}")
    print(f"Máximo de imagens numa classe: {class_counts.max()}")
    print(f"Mínimo de imagens numa classe: {class_counts.min()}")
    print(f"Média de imagens por classe: {class_counts.mean():.2f}\n")
    
    # Plot da distribuição
    plt.figure(figsize=(18, 6))
    
    sns.barplot(
        x=class_counts.index, 
        y=class_counts.values, 
        hue=class_counts.index, 
        palette="viridis",
        legend=False
    )
        
    plt.title(f"Distribuição das Classes - {dataset_name}", fontsize=14)
    plt.xlabel("Classes (Borboletas)", fontsize=12)
    plt.ylabel("Número de Amostras", fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()
    
    if save_dir is not None:
        # Formata o nome do ficheiro para não ter espaços
        safe_name = dataset_name.lower().replace(" ", "_")
        dist_plot_path = save_dir / f'class_distribution_{safe_name}.png'
        
        plt.savefig(dist_plot_path, dpi=300)
        print(f"Gráfico de distribuição salvo em: {dist_plot_path}")
        
    plt.show()
    
    return class_counts