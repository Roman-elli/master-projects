import sys
import json
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.utils import save_image

def safe_num_workers(requested: int) -> int:
    # Avoid notebook multiprocessing pickling issues on macOS/ipykernel.
    if "ipykernel" in sys.modules and int(requested) > 0:
        print("Notebook kernel detected: forcing num_workers=0 for DataLoader stability.")
        return 0
    return int(requested)

def make_subset_indices(n_total: int, fraction: float, seed: int = 42):
    n_keep = max(1, int(round(n_total * fraction)))
    g = np.random.RandomState(seed)
    idx = np.arange(n_total)
    g.shuffle(idx)
    return idx[:n_keep].tolist()

def load_ids_from_training_csv(csv_path: Path, index_column: str = "train_id_original") -> list[int]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"training.csv not found: {csv_path}\n"
            "Generate it first with scripts/generate_training_csv.py"
        )

    ids = []
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        r = csv.DictReader(f)
        if index_column not in (r.fieldnames or []):
            raise ValueError(
                f"Column {index_column!r} not present in {csv_path}. "
                f"Available: {r.fieldnames}"
            )
        for row in r:
            v = str(row.get(index_column, "")).strip()
            if v == "":
                continue
            ids.append(int(v))

    if len(ids) == 0:
        raise ValueError(f"No ids found in {csv_path} column {index_column!r}")
    return ids

def export_split_to_folder(
    loader: DataLoader,
    class_names: list[str],
    out_dir: Path,
    max_images: int | None = 500,
):
    out_dir = Path(out_dir)
    img_dir = out_dir / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    saved = 0

    for x, y, idx in loader:
        b = x.shape[0]
        for i in range(b):
            if max_images is not None and saved >= max_images:
                break

            label_id = int(y[i].item())
            label_name = class_names[label_id]
            src_idx = int(idx[i].item())

            file_name = f"img_{saved:06d}_label{label_id:02d}_idx{src_idx:06d}.png"
            path = img_dir / file_name
            save_image(x[i], path)

            rows.append({
                'file_name': file_name,
                'label_id': label_id,
                'label_name': label_name,
                'source_index': src_idx,
            })
            saved += 1

        if max_images is not None and saved >= max_images:
            break

    csv_path = out_dir / 'metadata.csv'
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['file_name', 'label_id', 'label_name', 'source_index'])
        w.writeheader()
        w.writerows(rows)

    print(f'Exported {saved} images to: {img_dir}')
    print(f'Metadata CSV: {csv_path}')

def save_experiment_results(run_dir: Path, model, history: list, config_dict: dict, test_metrics: dict = None):
    """
    Guarda o modelo, o gráfico de loss, o histórico em CSV, configs e métricas finais.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nA guardar resultados da experiência em:\n -> {run_dir}")
    
    # 1. Guardar Configurações (JSON)
    with open(run_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=4)
        
    # 2. Guardar Métricas Finais de Teste (JSON)
    if test_metrics:
        with open(run_dir / "test_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=4)
        
    # 3. Guardar Pesos do Modelo (.pth)
    torch.save(model.state_dict(), run_dir / "model.pth")
    
    # 4. Guardar Histórico (CSV)
    df_history = pd.DataFrame(history)
    df_history.to_csv(run_dir / "history.csv", index=False)
    
    # 5. Desenhar e Guardar Gráfico de Loss
    plt.figure(figsize=(10, 5))
    plt.plot(df_history['train_loss'], label='Loss Total', color='black', linewidth=2)
    plt.plot(df_history['train_recon_bce'], label='Reconstrução (BCE)', color='blue', linestyle='--')
    plt.plot(df_history['train_kl'], label='KL Divergence', color='red', linestyle='--')
    
    plt.title("Curvas de Treino - VAE", fontsize=14)
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(run_dir / "loss_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Guardado com sucesso!")