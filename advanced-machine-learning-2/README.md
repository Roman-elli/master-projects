## Estrutura de Código

Este repositório contém a implementação modular de uma pipeline de Deep Learning focada em Data Augmentation e classificação de imagens. O código foi desenhado para separar responsabilidades entre configuração, manipulação de dados, modelação generativa, classificação e avaliação.

### Estrutura de Diretórios
O projeto está organizado na seguinte árvore, separando scripts de módulos e notebooks de experimentação:

```text
./src/  
├── classifier/                    # Módulo do modelo de classificação  
├── dataset/                       # Módulo de manipulação de dados  
├── generative/                    # Arquiteturas generativas  
├── utils/                         # Utilitários de métricas e gráficos  
├── classifier_baseline.ipynb      # Pipeline do modelo base de classificação  
├── DF_note.ipynb                  # Experimentação com Modelo de Difusão  
├── GAN_note.ipynb                 # Experimentação com Rede Adversária  
└── VAE_note.ipynb                 # Experimentação com Autoencoder Variacional  
```

### Descrição dos Módulos
#### 1. config.py
- config.py: Centraliza os hiperparâmetros (Learning Rates, Batch Sizes, épocas), seeds para reprodutibilidade e todos os caminhos (paths) usados no projeto.

#### 2. classifier/
- cnn.py: Define a arquitetura da rede neural convolucional (baseline), incluindo a estrutura das camadas, funções de ativação e a lógica de inferência.

#### 3. dataset/
- dataloader.py: Responsável pelo pipeline ETL. Lida com o carregamento de imagens, transformações, separação (treino/validação/teste) e balanceamento de classes.

#### 4. generative/
- cvae.py: Implementação do Conditional Variational Autoencoder.

- dcgan.py: Implementação da Conditional Deep Convolutional Generative Adversarial Network.

- ddpm.py: Implementação do Denoising Diffusion Probabilistic Model.

- tuning.py: Configura a baseline utilizada em cada modelo generativo para a utilização do Optuna e parameter tuning.

#### 5. utils/
- metrics.py: Funções matemáticas para cálculo de métricas de avaliação (ex: F1-Score, Precision, Recall, FID, MSE).

- visualization.py: Scripts para desenhar curvas de convergência (Loss), matrizes de confusão e gerar as grelhas visuais de amostras artificiais.

### Cadernos de Experimentação (.ipynb)
Os ficheiros na raiz orquestram o uso dos módulos acima, servindo como ambiente principal para a execução:

- classifier_baseline.ipynb: Estabelece a pipeline elaborada para o modelo classificador (CNN).

- VAE_note.ipynb: Focado no ciclo iterativo do cVAE.

- GAN_note.ipynb: Executa e avalia a dinâmica adversarial da cDCGAN.

- DF_note.ipynb: Destinado ao treino do modelo de difusão (DDPM).