import os
import re
import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import config as cfg
from models.residual_network import ResNet

# Dataset Customizado para a pasta de Teste
class TestButterflyDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        # 1. Listar todas as imagens na pasta
        self.image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # 2. Função para extrair o número do nome do ficheiro (ex: "Image_12.jpg" -> 12)
        def extract_number(filename):
            match = re.search(r'\d+', filename)
            return int(match.group()) if match else 0
            
        # 3. Ordenar a lista pela Ordem Numérica Natural
        self.image_files.sort(key=extract_number)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Carregar imagem e converter para RGB
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, img_name


# Pipeline de Previsão
def generate_submission():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"A preparar para fazer previsões usando: {device}")

    df_train = pd.read_csv(cfg.TRAIN_LABELS_PATH)
    class_names = sorted(df_train['label'].unique().tolist())
    num_classes = len(class_names)

    test_transform = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    # Criar Dataset e DataLoader
    test_dataset = TestButterflyDataset(img_dir=cfg.TEST_IMG_DIR, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2)

    print("A carregar o modelo...")
    model = ResNet(num_classes=num_classes) 
    
    # Carregar os pesos
    model.load_state_dict(torch.load(cfg.MODEL_TO_LOAD_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Loop de Previsão
    print(f"A prever {len(test_dataset)} imagens...")
    predictions = []

    with torch.no_grad():
        for images, filenames in test_loader:
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Obter a classe com maior probabilidade
            _, predicted_indices = torch.max(outputs, 1)
            
            # Transformar os índices matemáticos nos nomes em texto
            for i in range(len(filenames)):
                pred_idx = predicted_indices[i].item()
                pred_label_name = class_names[pred_idx]
                
                predictions.append({
                    "filename": filenames[i],
                    "label": pred_label_name
                })

    # Guardar no formato exigido pelo Kaggle
    df_submission = pd.DataFrame(predictions)
    df_submission.to_csv(cfg.SUBMISSION_CSV_PATH, index=False)
    
    print(f"Submissão gerada com sucesso em: {cfg.SUBMISSION_CSV_PATH}")
    print("\nConfirmação das primeiras 5 previsões:")
    print(df_submission.head()) 

if __name__ == "__main__":
    generate_submission()