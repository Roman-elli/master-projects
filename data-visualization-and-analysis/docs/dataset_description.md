# Dicionário de Dados: Análise de ROI (Dívida vs. Ganhos)

Este documento descreve as colunas selecionadas do dataset *College Scorecard (Field of Study)* para a criação da visualização de **Retorno sobre Investimento**. O objetivo é cruzar a dívida mediana acumulada com o salário mediano após a graduação.

## 1. Identificação e Categorização

Estas colunas definem "Quem" (a instituição) e "O Quê" (o curso).

### `INSTNM` (Institution Name)
* **Descrição:** O nome da instituição de ensino (ex: *University of Florida*).
* **Uso na Visualização:** Usado para *Tooltips* (ao passar o mouse sobre um ponto no gráfico) ou para permitir que o usuário pesquise uma faculdade específica.

### `CIPCODE` (Classification of Instructional Programs Code)
* **Descrição:** Um código numérico que identifica a área de estudo específica. O formato é `XX.XXXX`.
    * *Exemplo:* `14.0101` (Engenharia Geral).
* **Uso na Visualização:** **Essencial para Agrupamento**.
    * *Dica:* Utilize apenas os **primeiros 2 dígitos** deste código para criar "Famílias de Cursos" (Cores do gráfico).
    * `14` = Engenharia
    * `52` = Negócios, Gestão e Marketing
    * `50` = Artes Visuais e Cênicas

### `CIPDESC` (CIP Description)
* **Descrição:** O nome por extenso da área de estudo associada ao código CIP.
* **Uso na Visualização:** Rótulos (Labels) legíveis para o usuário final.

### `CREDLEV` (Credential Level)
* **Descrição:** O nível da credencial conferida pelo curso.
* **Uso na Visualização:** **Filtro Obrigatório**.
    * Para uma comparação justa, filtre apenas um nível.
    * `3` = Bacharelato (Bachelor’s Degree) - *Recomendado para esta análise*.
    * `5` = Mestrado.
    * `6` = Doutoramento.

---

## 2. Eixo X: O Custo (Dívida)

### `DEBT_ALL_STGP_ANY_MDN`
* **Descrição:** A mediana da dívida acumulada de empréstimos federais (Stafford e Grad PLUS) pelos estudantes que **concluíram** o curso.
* **Por que esta coluna?**
    * `ALL`: Inclui todos os estudantes (independentemente do género).
    * `STGP`: Foca na dívida do aluno, excluindo empréstimos feitos pelos pais (Parent PLUS), o que reflete melhor o fardo do recém-graduado.
    * `MDN` (Mediana)

---

## 3. Eixo Y: O Retorno (Ganhos)

### `EARN_MDN_4YR` (Earnings Median - 4 Years After Completion)
* **Descrição:** O rendimento mediano dos graduados 4 anos após a conclusão do curso.
* **Por que esta coluna?**
    * `4YR`: Oferece uma visão mais estável da carreira do que `1YR` (1 ano após a formatura), onde muitos ainda estão em estágios, empregos temporários ou a transição para o mercado.
    * *Nota:* Se houver muitos dados faltantes para 4 anos, a alternativa é `EARN_MDN_1YR`.

---

## 4. Notas de Processamento de Dados (Data Cleaning)

Para garantir a eficiência do projeto, aplique as seguintes regras ao carregar os dados:

1.  **Tratamento de "PrivacySuppressed":** O dataset contém a string `PrivacySuppressed` ou `NULL` quando o número de alunos é muito baixo (para proteger a identidade).
    * *Ação:* Converta estas strings para `NaN` (Not a Number) e remova essas linhas antes da plotagem.
2.  **Conversão Numérica:** Certifique-se de que `DEBT_ALL_STGP_ANY_MDN` e `EARN_MDN_4YR` sejam lidos como `float` ou `integer`, não como texto.

## Fonte: https://collegescorecard.ed.gov/data