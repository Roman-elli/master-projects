# Assignment 1: Keypoint Detectors, Feature Descriptors and Applications

Este repositório contém a implementação em Python para avaliação de detetores de *keypoints* e descritores (clássicos e baseados em *Deep Learning*) e uma pipeline completa de *Image Stitching*.

## 1. Requisitos do Sistema
* **Python:** Versão 3.10 ou superior.
* **Hardware (CPU/GPU):** O código deteta e adapta-se automaticamente a ambientes CPU ou GPU. Recomenda-se o uso de uma GPU NVIDIA para acelerar a execução dos métodos de *Deep Learning* (SuperPoint e LightGlue).

## 2. Configuração do Ambiente

- **Passo 1: Criar e ativar o ambiente Conda**
```bash
conda create --name cvc python=3.10 -y
conda activate cvc
```

- **Passo 2: Instalar o motor PyTorch (Escolha a sua opção)**  
  - **Opção A (Com GPU NVIDIA):** Bashpip install torch torchvision --index-url [https://download.pytorch.org/whl/cu132](https://download.pytorch.org/whl/cu132)
(Nota: Se a sua placa for mais antiga, substitua cu132 pela versão CUDA apropriada, ex: cu118 ou cu126).  
  - **Opção B (Apenas CPU):** Abra o ficheiro requirements.txt e apague as linhas referentes ao torch e torchvision. Execute:
  ```Bash
  pip install torch torchvision --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
  ```

- **Passo 3: Instalar as dependências do projeto**  
```bash
pip install -r requirements.txt
```

## 3. Preparação dos Dados
Coloque os ficheiros de imagem na pasta data/ na raiz do projeto, respeitando a seguinte estrutura:   
- ```data/graf/```: Contém as imagens de teste e a homografia de ground-truth (H1to4p.txt).   
- ```data/Panorama/```: Contém as imagens (keble_a.jpg, etc.) usadas para o Image Stitching.   
- ```data/vehicle-seqs/```: Contém as sequências de imagens para as métricas de repetibilidade.

## 4. Execução e Reprodução
A implementação central está desenvolvida de forma modular através do pacote ```src/``` e interativa através de um Jupyter Notebook. Para reproduzir as experiências principais:   
1. Abra o VSCode na raiz do projeto.

2. Abra o ficheiro note.ipynb.Configure o Kernel do notebook para utilizar o ambiente Conda cvc.
3. Execute as células sequencialmente para observar a extração de características, cálculo de métricas e construção dos panoramas.

## 5. Referências e Implementações Externas
- Os métodos clássicos (SIFT, FAST, ORB, KAZE) utilizam a implementação do ```OpenCV 4.13.0.92```.   
- Os métodos de Deep Learning (SuperPoint, SuperGlue e LightGlue) utilizam as implementações pré-treinadas baseadas em PyTorch, conforme recomendado na documentação do projeto.