1. Análise Exploratória e Estrutura dos Dados
Divisão de Dados (TVT Split): Implementámos uma divisão rigorosa de Treino/Validação internamente, com random_state fixo, garantindo que o modelo é avaliado de forma justa e reprodutível, sem fugas de dados (data leakage). O conjunto de Teste oficial é mantido oculto pelo Kaggle.

Balanço de Classes: (Ponto a avaliar). Dado que existem 75 espécies de borboletas, é expectável que algumas classes tenham menos imagens. Se se confirmar o desequilíbrio, será necessário aplicar Data Augmentation (rotações, cortes) nas próximas fases (CNN) para gerar novos dados artificiais e equilibrar a aprendizagem.

2. Implementação da Baseline: Multi-Layer Perceptron (MLP)
Arquitetura: Testámos o conceito de Bottleneck (ex: [1024, 512] neurónios), reduzindo progressivamente o tamanho das camadas ocultas para forçar a rede a extrair características essenciais e ignorar ruído.

Experimentação Dinâmica: Desenvolvemos um pipeline modular (config.py) que permite alternar e comparar rapidamente os requisitos do guião:

Loss Functions: CrossEntropy e MultiMarginLoss.

Otimizadores: ADAM e RMSprop.

Monitorização: Implementámos a validação em tempo real e a gravação automática das curvas de Loss para identificar o ponto exato de paragem da aprendizagem.

3. Resultados e Evolução do MLP (O Problema do Espaço 2D)
Observação: O treino do MLP resultou num F1-Score de treino elevado (~86.5%) mas um F1-Score de validação catastrófico (~0.9%). Aumentar a profundidade da rede (mais camadas) apenas acelerou a memorização dos dados de treino, sem ganhos na validação.

Diagnóstico: Detetámos um cenário de Overfitting extremo. A operação de Flatten (achatar imagens 2D num vetor 1D) destrói as correlações espaciais. O modelo decorou os píxeis exatos do treino em vez de aprender os padrões geométricos das asas das borboletas.

Decisão Estratégica: Vamos submeter este MLP básico no Kaggle exatamente como está. Ele servirá como a nossa baseline (linha de base) provando empiricamente no relatório final porquê que arquiteturas lineares falham em visão computacional e justificando a evolução para CNNs.

4. Próximos Passos (Rumo às CNNs e ResNets)
Submeter as previsões do MLP no Kaggle para obter a nota base.

Desenvolver a arquitetura CNN para manter a integridade 2D das imagens através de filtros espaciais.

Introduzir Model Checkpointing (guardar os pesos da melhor época validada) e Data Augmentation para robustecer a CNN e a futura ResNet.