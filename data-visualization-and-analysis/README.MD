# EduVision USA
**Análise Interativa do Retorno sobre o Investimento no Ensino Superior dos Estados Unidos**

O **EduVision USA** é um *dashboard* interativo desenvolvido em Python (utilizando a framework Plotly Dash e Bootstrap). A aplicação tem como objetivo apoiar a tomada de decisão de gestores universitários e decisores políticos, analisando o Retorno sobre o Investimento (ROI), a evolução salarial, a distribuição demográfica e o risco de incumprimento (*Default Risk*) de milhares de cursos nos EUA.

---

## Estrutura do Projeto

O projeto foi construído com uma arquitetura modular para facilitar a manutenção e escalabilidade do código:

*   **`/assets/`**: Pasta destinada a armazenar os ficheiros CSV originais extraídos do *College Scorecard* do governo dos EUA (dados brutos ao nível da instituição e dos cursos).
*   **`/data/`**: Pasta gerada pelo *script* de extração, onde são guardados os ficheiros limpos e fundidos (`merged_data.csv`), prontos a serem consumidos pelo *dashboard*.
*   **`/src/`**: Diretório principal do código fonte:
    *   `config.py`: Dicionários de mapeamento (ex: estados americanos), caminhos de ficheiros e listas de variáveis numéricas/categóricas.
    *   **`/core/`**:
        *   `engine.py`: Lógica pesada de *backend* (processamento de dados Pandas, preenchimento de nulos e cálculos dos KPIs principais).
        *   `layout.py`: Estrutura visual da aplicação (Sidebar e Content), grelhas do Bootstrap e posicionamento dos botões.
        *   `components.py`: Funções auxiliares para gerar componentes visuais reutilizáveis (como os cartões de KPI e o Resumo Dinâmico).
*   **`app.py`**: O ficheiro principal de entrada da aplicação. Gere a inicialização do servidor Dash e contém os *Callbacks* responsáveis pela interatividade (filtragem cruzada e atualização em tempo real).
*   **`run-extract.cmd`**: *Script* executável de automação para extrair, limpar e fundir os dados brutos.
*   **`run-app.cmd`**: *Script* executável de automação para iniciar o servidor local do *dashboard*.

---

## Como Executar o Projeto

Para testar e rodar o projeto localmente na tua máquina, segue os passos abaixo.

### Pré-requisitos
Certifica-te de que tens o **Python 3.8+** instalado. Recomenda-se a criação de um ambiente virtual (*virtual environment*) e a instalação das dependências (Pandas, Dash, Plotly, Dash Bootstrap Components).

```bash
pip install dash dash-bootstrap-components pandas plotly