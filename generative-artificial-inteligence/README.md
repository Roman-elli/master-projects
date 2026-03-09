# Instalação e Configuração

## Versão recomendada de Python

👉 **Python 3.10 (recomendado)**

**Justificação:**
- Totalmente compatível com `scikit-learn`, `numpy`, `matplotlib` e `jupyter`
- Estável e amplamente suportado em ambientes académicos
- Evita incompatibilidades ainda frequentes em Python 3.11+ e 3.12

Python 3.9 também funciona, mas **Python 3.10 é a versão aconselhada para esta ficha**.

---

## Instalação das dependências

As instruções abaixo assumem que estás na pasta do projeto,
onde se encontra o ficheiro `requirements.txt`.

---

## Opção A — Ambiente virtual com `venv` + `pip` (recomendado)

Esta é a opção **mais simples e universal**, adequada para a maioria dos alunos.

### 1️⃣ Criar ambiente virtual

```bash
python3.10 -m venv genai-env
```

### 2️⃣ Ativar o ambiente

- **Linux / macOS**
```bash
source genai-env/bin/activate
```

- **Windows**
```bash
genai-env\Scripts\activate
```

Após ativação, o terminal deverá indicar algo semelhante a:
```
(genai-env)
```

---

### 3️⃣ Instalar dependências

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Abrir Jupyter

```bash
jupyter notebook
```

ou

```bash
jupyter lab
```

Depois, abrir o ficheiro da ficha (`.ipynb`).

---

## Opção B — Ambiente com Conda / Miniconda

Esta opção é recomendada para alunos que **já utilizam conda**
ou para laboratórios institucionais.

### 1️⃣ Criar ambiente conda

```bash
conda create -n genai python=3.10
```

### 2️⃣ Ativar o ambiente

```bash
conda activate genai
```

---

### 3️⃣ Instalar dependências

```bash
python -m pip install -r requirements.txt
```

> Nota: usar `pip` dentro do ambiente conda é intencional,
> garantindo que as versões seguem exatamente o `requirements.txt`.

---

### 4️⃣ Abrir Jupyter

```bash
jupyter notebook
```

ou

```bash
jupyter lab
```
ou 

Utilizar VSCode ou outro Editor de código que suporte IronPython Notebooks.
---


## Resolução de problemas comuns

### ❗ Erro `ModuleNotFoundError`
- Verifica se o ambiente virtual está ativo
- Verifica se as dependências foram instaladas com  
  `pip install -r requirements.txt` **no ambiente correto**

### ❗ Gráficos não aparecem no notebook
Adicionar no topo do notebook:
```python
%matplotlib inline
```

