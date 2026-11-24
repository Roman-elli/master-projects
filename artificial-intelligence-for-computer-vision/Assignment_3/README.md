# cvc-a3: Real-Time Facial Analysis and Identity Recognition

Este projeto implementa uma aplicação modular em Python para análise facial em tempo real. Os requisitos incluem deteção facial, estimativa de idade e género, análise de expressões e reconhecimento de identidade em *closed-set* (<20 indivíduos), processados localmente.

## Estrutura do Projeto

```text
cvc-a3/
│
├── config.yaml                # Ficheiro central de configurações (caminhos, thresholds)
├── note.ipynb                 # Notebook principal para orquestração da pipeline
├── requirements.txt           # Dependências do projeto
├── README.md                  # Este ficheiro
│
├── src/                       # Módulos independentes do código-fonte
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py          # Loader de configurações
│   ├── io/
│   │   ├── __init__.py
│   │   ├── acquisition.py     # Captura de webcam/vídeo
│   │   └── ui.py              # Interface gráfica e anotações
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── detection.py       # Deteção facial e landmarks
│   │   ├── alignment.py       # Alinhamento e recorte facial
│   │   ├── attributes.py      # Estimativa de idade, género e expressão
│   │   └── recognition.py     # Extração de embeddings e matching
│   └── utils/
│       ├── __init__.py
│       └── monitoring.py      # Métricas de tempo real (FPS, latência, recursos)
│       └── evaluation.py
│
├── models/                    # Pesos de modelos pré-treinados
│   ├── detection/             
│   ├── attributes/            
│   └── recognition/           
│
└── data/                      
    ├── gallery/               # Imagens/templates de identidades inscritas (<20)
    └── test_videos/           # Vídeos para testes iterativos
```

## Configuração do Ambiente (Miniconda)

Para garantir compatibilidade e estabilidade com as bibliotecas de visão computacional, o projeto utiliza Python 3.10.

**1. Criar o ambiente virtual:**
```bash
conda create --name cvc-a3 python=3.10 -y
```

**2. Ativar o ambiente:**
```bash
conda activate cvc-a3
```

**3. Instalar as dependências:**
```bash
pip install -r requirements.txt
```

**4. Registar o kernel para o Jupyter Notebook:**
```bash
python -m ipykernel install --user --name=cvc-a3 --display-name "Python (cvc-a3)"
```

## Execução

Inicie o Jupyter Lab ou Notebook na raiz do projeto e abra o ficheiro `note.ipynb`. Certifique-se de selecionar o kernel **Python (cvc-a3)**.

```bash
jupyter lab
```