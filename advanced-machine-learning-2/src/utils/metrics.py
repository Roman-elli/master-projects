import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report
import json
from pathlib import Path
from sklearn.metrics import classification_report

# Baseline CNN
def save_cnn_train(history, save_dir):
    # Gráfico de Loss
    plt.figure(figsize=(8, 6))
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'], label='Val Loss', color='orange')
    plt.title('Baseline CNN - Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    loss_path = save_dir / 'baseline_loss.png'
    plt.savefig(loss_path)
    plt.show()
    print(f"Gráfico de Loss salvo em: {loss_path}")

    # Gráfico de Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(history['train_acc'], label='Train Accuracy', color='blue')
    plt.plot(history['val_acc'], label='Val Accuracy', color='orange')
    plt.title('Baseline CNN - Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    acc_path = save_dir / 'baseline_accuracy.png'
    plt.savefig(acc_path)
    plt.show()
    print(f"Gráfico de Acurácia salvo em: {acc_path}")

def evaluate_cnn(best_model, test_dataset, test_loader, device='cpu', results_dir=None):
    best_model.eval()

    all_preds = []
    all_labels = []

    print("Avaliando o modelo no conjunto de teste...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = best_model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 1. Relatório de Classificação em Texto (Para o console)
    print("\n--- Relatório de Classificação ---")
    report_text = classification_report(
        all_labels, all_preds, target_names=test_dataset.classes, zero_division=0
    )
    print(report_text)

    # 2. Relatório de Classificação em Dicionário (Para extrair as métricas)
    report_dict = classification_report(
        all_labels, all_preds, target_names=test_dataset.classes, zero_division=0, output_dict=True
    )
    
    # 3. Extrair e Imprimir as Métricas Globais SOTA
    accuracy = report_dict['accuracy']
    macro_f1 = report_dict['macro avg']['f1-score']
    weighted_f1 = report_dict['weighted avg']['f1-score']

    print("\n" + "="*40)
    print(" RESUMO DE PERFORMANCE GLOBAL (BASELINE)")
    print("="*40)
    print(f"Accuracy Geral:    {accuracy:.4f}")
    print(f"Macro F1-Score:    {macro_f1:.4f}")
    print(f"Weighted F1-Score: {weighted_f1:.4f}")
    print("="*40 + "\n")

    # 4. Lógica para extrair e salvar os resultados
    if results_dir is not None:        
        summary = {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1
        }
        with open(results_dir / 'baseline_summary.json', 'w') as f:
            json.dump(summary, f, indent=4)
        # ----------------------------------------
        
        class_metrics = {k: v for k, v in report_dict.items() if k in test_dataset.classes}
        sorted_classes = sorted(class_metrics.items(), key=lambda item: item[1]['f1-score'])
        
        worst_15 = {k: v for k, v in sorted_classes[:15]}
        worst_classes_path = results_dir / 'worst_classes.json'

        with open(worst_classes_path, 'w') as f:
            json.dump(worst_15, f, indent=4)
            
        print(f"Resumo global salvo em: {results_dir / 'baseline_summary.json'}")
        print(f"Lista das 15 piores classes salva em: {worst_classes_path}")

    return report_dict