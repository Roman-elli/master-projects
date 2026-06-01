import matplotlib.pyplot as plt
import torch
import json
import pandas as pd
import seaborn as sns
import os
import numpy as np

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

    # 1. Relatório de Classificação em Texto
    report_text = classification_report(
        all_labels, all_preds, target_names=test_dataset.classes, zero_division=0
    )

    # 2. Relatório de Classificação em Dicionário (Para extrair as métricas)
    report_dict = classification_report(
        all_labels, all_preds, target_names=test_dataset.classes, zero_division=0, output_dict=True
    )
    
    # 3. Extrair e Imprimir as Métricas Globais
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

    # 4. Salvar os relatórios e resumos
    if results_dir is not None:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Resumo Global (JSON)
        summary = {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1
        }
        with open(results_dir / 'baseline_summary.json', 'w') as f:
            json.dump(summary, f, indent=4)

        # Relatório Completo (TXT)
        full_report_path = results_dir / 'full_classification_report.txt'
        with open(full_report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(f"Relatório de texto completo salvo em: {full_report_path}")
        print(f"Resumo global salvo em: {results_dir / 'baseline_summary.json'}")

    return report_dict

def analyze_augmentation_candidates(report_dict, save_dir=None):
    """
    Analisa o dicionário de resultados, cria o gráfico de quadrantes
    e devolve as classes ideais para Data Augmentation.
    """
    # 1. Extrair os dados das classes (ignorando as médias globais)
    classes_data = []
    for key, metrics in report_dict.items():
        if key not in ['accuracy', 'macro avg', 'weighted avg']:
            classes_data.append({
                'Classe': key,
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1-score']
            })

    df = pd.DataFrame(classes_data)

    # 2. Lógica de Categorização (Os 4 Quadrantes)
    def categorize(row):
        p = row['Precision']
        r = row['Recall']

        if p < 0.5 and r < 0.5:
            return 'Crítica (Gerar Dados)'
        elif p >= 0.5 and r < 0.5:
            return 'Tímida (Gerar Dados)'
        elif p < 0.5 and r >= 0.5:
            return 'Gulosa (Não Gerar)'
        else:
            return 'Boa (Não Gerar)'

    df['Estrategia'] = df.apply(categorize, axis=1)

    # 3. Gerar o Gráfico de Dispersão (Scatter Plot)
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df,
        x='Recall',
        y='Precision',
        hue='Estrategia',
        palette={
            'Crítica (Gerar Dados)': '#d62728',
            'Tímida (Gerar Dados)': '#ff7f0e',
            'Gulosa (Não Gerar)': '#9467bd',
            'Boa (Não Gerar)': '#2ca02c'
        },
        s=100, alpha=0.8, edgecolor='k'
    )

    # Adicionar as linhas divisórias dos quadrantes (0.5)
    plt.axvline(0.5, color='gray', linestyle='--', alpha=0.6)
    plt.axhline(0.5, color='gray', linestyle='--', alpha=0.6)

    plt.title('Estratégia de Data Augmentation: Precision vs Recall', fontsize=14)
    plt.xlabel('Recall (Capacidade de encontrar a classe)', fontsize=12)
    plt.ylabel('Precision (Frequência de acerto quando prevê a classe)', fontsize=12)

    # Mover a legenda para fora do gráfico
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()

    # Salvar o gráfico na Drive
    if save_dir is not None:
        plot_path = save_dir / 'augmentation_strategy_plot.png'
        plt.savefig(plot_path, dpi=300)
        print(f"Gráfico de estratégia salvo em: {plot_path}\n")

    plt.show()

    # 4. Filtrar e listar as recomendadas para geração
    recomendadas = df[df['Estrategia'].str.contains('Gerar Dados')].copy()

    # Ordenar pelas que têm o pior F1-score primeiro
    recomendadas = recomendadas.sort_values(by='F1-Score')

    print("="*65)
    print(f"  CLASSES RECOMENDADAS PARA GERAÇÃO ({len(recomendadas)} classes)")
    print("="*65)
    print(recomendadas[['Classe', 'Precision', 'Recall', 'F1-Score']].to_string(index=False))

    if save_dir is not None:
        csv_path = save_dir / 'classes_para_augmentar.csv'
        recomendadas.to_csv(csv_path, index=False)
        print(f"\nLista detalhada guardada em: {csv_path}")

    return recomendadas

def plot_f1_tradeoff(baseline_metrics_dict, new_metrics_dict, save_dir, aug_type="cVAE", top_n=5):
    """
    Gera um Gráfico de Barras Divergentes mostrando as classes que mais ganharam 
    e as que mais perderam F1-Score.
    """
    # 1. Calcular as diferenças (Deltas) para todas as classes
    deltas = []
    for cls, metrics in baseline_metrics_dict.items():
        # Ignorar as chaves globais do classification_report (accuracy, macro avg, etc.)
        if isinstance(metrics, dict) and 'f1-score' in metrics and cls in new_metrics_dict:
            f1_base = metrics['f1-score']
            f1_aug = new_metrics_dict[cls]['f1-score']
            delta = f1_aug - f1_base
            deltas.append({'Classe': cls, 'F1_Base': f1_base, 'F1_Aug': f1_aug, 'Delta': delta})
    
    df_deltas = pd.DataFrame(deltas)
    if df_deltas.empty:
        print("Aviso: Nenhum dado válido encontrado para comparar trade-offs.")
        return

    # 2. Ordenar pelos deltas (do pior para o melhor)
    df_deltas = df_deltas.sort_values(by='Delta', ascending=True)

    # 3. Remover classes que não sofreram alteração (Delta == 0) para limpar o gráfico
    df_deltas = df_deltas[df_deltas['Delta'] != 0]
    
    # 4. Selecionar as "Top N" Perdedoras e as "Top N" Ganhadoras
    piores = df_deltas.head(top_n)
    melhores = df_deltas.tail(top_n)
    
    # Unir num só DataFrame para o gráfico
    plot_df = pd.concat([piores, melhores])
    
    # ====================
    # DESENHAR O GRÁFICO 
    # ====================
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Vermelho para perdas, Verde para ganhos
    cores = ['#e74c3c' if val < 0 else '#2ecc71' for val in plot_df['Delta']]
    
    y_pos = np.arange(len(plot_df))
    bars = ax.barh(y_pos, plot_df['Delta'], color=cores, edgecolor='white', linewidth=1.5, height=0.6)
    
    # Linha zero central grossa para marcar a divisão
    ax.axvline(0, color='#2c3e50', linewidth=1.5, zorder=0)
    
    # Formatação de Eixos
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df['Classe'], fontweight='bold', fontsize=10)
    
    # Remover as bordas (Spines) 
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(False)
    
    # Grelha suave no fundo
    ax.xaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    
    # Adicionar os valores exatos nas pontas das barras
    for i, val in enumerate(plot_df['Delta']):
        # Se for positivo, coloca o texto à direita da barra. Se negativo, à esquerda.
        ha = 'left' if val > 0 else 'right'
        offset = 0.02 if val > 0 else -0.02
        sinal = "+" if val > 0 else ""
        
        ax.text(val + offset, i, f"{sinal}{val:.2f}", va='center', ha=ha, 
                fontweight='bold', fontsize=11, 
                color='#27ae60' if val > 0 else '#c0392b')
        
    ax.set_xlabel('Variação Absoluta do F1-Score ($\Delta$)', fontweight='bold', fontsize=12)
    ax.set_title(f'Efeito Trade-off: Maiores Ganhos e Perdas com {aug_type}', fontweight='bold', fontsize=14, pad=20)
    
    # Estender o eixo X para que o texto não saia da imagem
    max_abs = plot_df['Delta'].abs().max()
    ax.set_xlim(-max_abs - 0.15, max_abs + 0.15)
    
    plt.tight_layout()
    
    # Salvar
    save_path = os.path.join(save_dir, f'{aug_type.lower()}_f1_tradeoff_diverging.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Gráfico Divergente (Ganhos vs Perdas) salvo em: {save_path}")

def parse_classification_report_txt(filepath):
    """Lê o .txt do sklearn e converte de volta para o formato de dicionário."""
    metrics_dict = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Ignorar linhas vazias, cabeçalho e resumos globais
            if not line or line.startswith('precision') or line.startswith('accuracy') or line.startswith('macro avg') or line.startswith('weighted avg'):
                continue
            
            # Dividir a linha por espaços. Os últimos 4 elementos são sempre os números
            parts = line.split()
            if len(parts) >= 4:
                try:
                    f1_score = float(parts[-2]) # O F1-Score é o penúltimo valor
                    
                    # O nome da classe é tudo o que vem antes dos 4 números
                    class_name = " ".join(parts[:-4]).strip()
                    
                    metrics_dict[class_name] = {'f1-score': f1_score}
                except ValueError:
                    continue # Se falhar a conversão para float, ignora a linha
    return metrics_dict

# cVAE
def evaluate_vae(history, save_dir):
    """Gera gráficos de convergência detalhados para o VAE."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    
    # Gráfico 1: Acompanhamento da Loss Total (Verificação de Overfitting)
    ax1.plot(epochs, history['train_loss'], label='Treino (Loss Total)', color='royalblue', linewidth=2)
    ax1.plot(epochs, history['val_loss'], label='Validação (Loss Total)', color='darkorange', linewidth=2)
    ax1.set_title('Convergência da Loss Total', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Épocas')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Gráfico 2: Decomposição
    ax2.plot(epochs, history['train_mse'], label='Erro de Pixels (MSE)', color='forestgreen', linewidth=2)
    ax2.plot(epochs, history['train_perceptual'], label='Semântica / Textura (LPIPS)', color='crimson', linewidth=2)
    ax2.plot(epochs, history['train_kld'], label='Organização Latente (KLD)', color='purple', linewidth=2)
    ax2.set_title('Decomposição Dinâmica das Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Épocas')
    ax2.set_ylabel('Loss (Escala Logarítmica)')
    ax2.set_yscale('log') # Escala Logarítmica para vermos as três linhas juntas claramente
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'cvae_loss_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Gráficos de métricas SOTA guardados em: {save_path}")

# cDCGAN
def evaluate_gan(history, save_dir):
    """
    Gera o gráfico da dinâmica do treino adversário (Min-Max Game) para a GAN.
    """
    # A history da GAN tem 'g_loss' e 'd_loss'
    epochs = range(1, len(history['g_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot das Losses
    plt.plot(epochs, history['g_loss'], label='Gerador (G_Loss)', color='royalblue', linewidth=2)
    plt.plot(epochs, history['d_loss'], label='Discriminador (D_Loss)', color='crimson', linewidth=2)
    
    # Estilização
    plt.title('Dinâmica de Treino Adversário (Equilíbrio de Nash)', fontsize=14, fontweight='bold')
    plt.xlabel('Épocas')
    plt.ylabel('BCE Loss')
    
    # Adicionar uma linha de referência em 0.69 (o valor ideal teórico onde o D está 50% confuso)
    plt.axhline(y=0.693, color='forestgreen', linestyle=':', label='Equilíbrio Ideal (ln(2))', linewidth=2)
    
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    # Guardar ficheiro
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'cgan_loss_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Gráfico de métricas adversárias guardado em: {save_path}")

# diff PixelUnit
def evaluate_diff(history, save_dir):
    """
    Gera o gráfico da dinâmica do treino para o Modelo de Difusão (DDPM).
    """
    import os
    import matplotlib.pyplot as plt

    # A history do DDPM tem apenas a chave 'loss' (MSE)
    epochs = range(1, len(history['loss']) + 1)
    
    # Criamos um gráfico com as mesmas dimensões da GAN para manter o padrão 
    plt.figure(figsize=(12, 5))
    
    # Plot da Loss (Cor diferente para distinguir facilmente das GANs)
    plt.plot(epochs, history['loss'], label='UNet (MSE Noise Loss)', color='darkorange', linewidth=2.5)
    
    # Estilização SOTA
    plt.title('Dinâmica de Treino do Modelo de Difusão (DDPM)', fontsize=14, fontweight='bold')
    plt.xlabel('Épocas')
    plt.ylabel('MSE Loss')
    
    # Adicionar uma linha de referência no 0 (O modelo nunca chega a 0 perfeito, mas é o limite teórico)
    plt.axhline(y=0.0, color='forestgreen', linestyle=':', label='Objetivo Teórico (Erro Zero)', linewidth=2)
    
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    # Guardar ficheiro
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'ddpm_loss_curve.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Gráfico da MSE Loss guardado em: {save_path}")