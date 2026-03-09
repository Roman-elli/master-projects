# Questões Analíticas e Correlações: Retorno sobre o Investimento (ROI) no Ensino Superior

Este documento detalha as principais relações exploradas no dataset *College Scorecard* para o desenvolvimento da aplicação de visualização interativa, cumprindo os requisitos de análise profunda estipulados no projeto.

## 1. A Correlação Principal: O Verdadeiro Retorno (ROI)
* **Variáveis Cruzadas:** Dívida Mediana (`DEBT_ALL_STGP_ANY_MDN`) vs. Ganhos Medianos 4 Anos Após a Conclusão (`EARN_MDN_4YR`).
* **Questão Analítica:** Um maior investimento inicial (dívida) traduz-se obrigatoriamente num maior salário a médio prazo? 
* **Objetivo Visual:** Identificar os cursos "Unicórnio" (baixa dívida e alto ganho) e as "Armadilhas Financeiras" (alta dívida e baixo ganho).

## 2. A Correlação de Risco: A Realidade do Pagamento
* **Variáveis Cruzadas:** Rácio Dívida/Ganhos vs. Taxas de Reembolso a 2 anos (`BBRR2_FED_COMP_MAKEPROG` e `BBRR2_FED_COMP_DFLT`).
* **Questão Analítica:** Os cursos identificados como tendo um fraco ROI são os mesmos que levam os alunos ao incumprimento (default)? 
* **Objetivo Visual:** Validar se uma dívida alta é justificável caso o curso garanta uma boa taxa de progresso no pagamento (alunos que conseguem efetivamente abater a dívida).

## 3. A Correlação Demográfica: O Peso Socioeconómico
* **Variáveis Cruzadas:** Dívida de alunos com bolsa Pell (`DEBT_PELL_STGP_ANY_MDN`) vs. Dívida de alunos sem bolsa Pell (`DEBT_NOPELL_STGP_ANY_MDN`).
* **Questão Analítica:** Para o mesmo curso e instituição, a dívida acumulada por um aluno com grandes necessidades financeiras (bolseiro Pell) é desproporcional à de um aluno não-bolseiro?
* **Objetivo Visual:** Avaliar se o sistema atua como "elevador social" ou se agrava as desigualdades financeiras à partida.

## 4. A Correlação de Género: Desigualdade no Financiamento
* **Variáveis Cruzadas:** Dívida acumulada por homens (`DEBT_MALE_STGP_ANY_MDN`) vs. Dívida acumulada por mulheres/não-homens (`DEBT_NOTMALE_STGP_ANY_MDN`).
* **Questão Analítica:** Existe uma diferença estrutural na forma como homens e mulheres financiam o mesmo curso?
* **Objetivo Visual:** Mapear o fosso de género no custo do ensino superior.

## 5. A Correlação por Áreas de Estudo (A Grande Divisão)
* **Variáveis Cruzadas:** Famílias de cursos (primeiros 2 dígitos do `CIPCODE`) vs. Métricas financeiras (Dívida e Ganhos).
* **Questão Analítica:** Quais são as famílias de cursos (ex: Engenharia, Gestão, Artes, Saúde) que oferecem o retorno financeiro mais seguro e estável?
* **Objetivo Visual:** Utilizar esta dimensão categórica para agrupar e colorir as marcas visuais (ex: bolhas num *Scatter Plot*), facilitando a identificação de padrões por área do saber.

## 6. A Correlação de Formato: Ensino Online vs. Presencial
* **Variáveis Cruzadas:** Modalidade de ensino à distância (`DISTANCE`) vs. Dívida e Ganhos.
* **Questão Analítica:** Os cursos que podem ser concluídos de forma 100% remota oferecem um ROI competitivo em relação aos programas tradicionais presenciais?
* **Objetivo Visual:** Integrar um filtro interativo na *dashboard* para permitir a comparação direta entre as duas modalidades.