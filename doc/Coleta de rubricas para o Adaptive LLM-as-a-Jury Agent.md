# Coleta de rubricas para o Adaptive LLM-as-a-Jury Agent

Oct 11, 2026 · @Thais

## Objetivo da coleta

A coleta montou um catálogo de 181 rubricas de avaliação, extraídas de 30 avaliadores LLM-as-a-judge consolidados. O catálogo alimenta o Estágio 2 (Gerador de Regras) do Adaptive LLM-as-a-Jury Agent. O Gerador de Regras combina critérios base estáticos (Relevância, Corretude, Clareza e Completude) com até três critérios contextuais gerados por LLM. Sem uma referência externa, esses critérios dependem apenas do julgamento do modelo gerador, o que reintroduz a instabilidade que a arquitetura busca mitigar.

A RSL apontou que a avaliação automática ainda é fragmentada por domínio e que frameworks multimétricos são a dimensão mais madura da literatura. A coleta aproveita essa maturidade: em vez de redigir critérios do zero, o sistema parte de definições, escalas e procedimentos já usados em ferramentas de mercado e em benchmarks acadêmicos. O resultado é um repositório versionável de critérios, do qual o Gerador de Regras seleciona e adapta itens conforme o tipo de tarefa e a complexidade da conversa.

## Etapa 1: levantamento dos 30 avaliadores

A primeira etapa identificou os 30 avaliadores LLM-as-a-judge mais relevantes, ordenados de forma aproximada por adoção, uso em produção e citações. A ordem não é um ranking oficial. A busca cobriu documentação oficial, repositórios e artigos de 2023 a 2026, com verificação na documentação vigente para as plataformas que mudam com frequência (Amazon Bedrock, Azure AI Foundry, MLflow 3, Vertex AI, Galileo, Llama Guard 4 e Granite Guardian).

Para garantir diversidade de abordagens, a seleção incluiu cinco grupos:

- **Frameworks e bibliotecas abertas:** DeepEval, Ragas, LlamaIndex, TruLens, Arize Phoenix, MLflow, Braintrust autoevals, Promptfoo, Opik, Langfuse, Evidently e W&B Weave.
- **Modelos juízes especializados:** Llama Guard 4 (Meta), Granite Guardian (IBM), Prometheus 2 e os modelos Lynx e Glider da Patronus.
- **Serviços de nuvem:** Vertex AI, Amazon Bedrock e Azure AI Foundry.
- **Ferramentas de critérios customizados:** IBM EvalAssist/Unitxt, Llama Stack Scoring, OpenAI Evals, LangChain e Galileo.
- **Métodos e benchmarks acadêmicos:** G-Eval, MT-Bench, AlpacaEval, Arena-Hard-Auto, WildBench e Inspect AI.

Para cada avaliador foram registrados organização, tipo, licença, abordagem de julgamento e foco principal. Essa ficha permite rastrear de onde vem cada critério usado pelo júri e qual a sua licença de reuso.

## Etapa 2: coleta e estruturação das rubricas

A segunda etapa decompôs cada avaliador em suas métricas individuais e produziu 181 registros. A unidade de coleta é a métrica, não a ferramenta: um mesmo avaliador contribui com várias linhas (o DeepEval, por exemplo, com 16). Cada registro segue um esquema comum, pensado para ser consumido diretamente pelo Gerador de Regras.

| Campo | Conteúdo | Uso no framework |
| --- | --- | --- |
| métrica e categoria | Nome original e grupo (RAG, segurança, agentes etc.) | Filtrar critérios pela classificação da tarefa |
| o que avalia | Definição operacional do critério | Texto-base do critério na rubrica |
| entradas necessárias | Pergunta, contexto, resposta, referência, trace | Verificar se a conversa tem os dados exigidos |
| resumo do prompt | Procedimento do juiz, parafraseado | Passos do prompt dos juízes |
| escala de saída | Binária, Likert, 0–1, pairwise, severidade | Indicar como converter para True/False |
| exige referência | Sim, não ou opcional | Separar critérios do Score do Júri e do Score de Acurácia |
| tarefas indicadas | Domínios de uso recomendados | Mapear para Código, Resumo ou Geral |

Os prompts originais não foram copiados. A coluna de resumo descreve a estrutura de cada julgamento (passos, formato de saída, regras de agregação) e aponta, na coluna de fonte, para o template oficial. Isso respeita as licenças das ferramentas e mantém o catálogo auditável.

A distribuição confirma o diagnóstico da RSL. RAG concentra 48 métricas (34 de geração e 14 de recuperação), seguido de segurança e ética (23) e qualidade e estilo (23). Agentes (14) e diálogo multi-turno (6) aparecem bem menos, justamente as dimensões centrais para conversas com assistentes virtuais. Quanto à escala, 76 métricas (42%) já são binárias; as outras 105 usam Likert, escalas contínuas, comparação pairwise ou severidade.

## Etapa 3: geração dos prompts de rubrica

A terceira etapa transformou cada registro em um prompt de rubrica executável, a partir da definição ("o que avalia") e do resumo do procedimento. A geração foi programática e determinística: um script classifica cada métrica por modo de julgamento e por tipo de escala e monta o prompt a partir de blocos fixos. Assim, os 181 prompts compartilham a mesma estrutura, o que reduz a variância introduzida pela redação e facilita comparar juízes no Estágio 3.

Cada prompt contém seis blocos:

1. **Papel e definição:** o juiz recebe o nome da métrica, a categoria e a definição operacional.
2. **Entradas delimitadas:** placeholders entre tags (`{pergunta}`, `{contexto}`, `{resposta}`, `{referencia}`, `{conversa}`, `{trace}`), deduzidos das entradas exigidas pela métrica.
3. **Procedimento:** passos específicos do modo de julgamento. Métricas de fidelidade decompõem a resposta em afirmações atômicas classificadas como SUPORTADA, CONTRADITA ou NÃO\_ENCONTRADA. Métricas de recuperação avaliam cada trecho na ordem do ranking. A sumarização combina cobertura por perguntas fechadas e alinhamento factual.
4. **Rubrica ancorada:** níveis descritos na escala original da métrica, com âncoras textuais por modo (fidelidade, relevância, corretude, agentes, conversa, segurança, entre outros).
5. **Regras anti-viés:** neutralidade quanto à posição e ao tamanho da resposta, proibição de conhecimento externo em métricas de fidelidade e raciocínio antes do veredito.
6. **Saída estruturada:** JSON parseável, compatível com o tratamento de abstenções do júri, que considera abstinente o juiz que não formata a resposta corretamente.

Duas situações receberam tratamento próprio. As 40 métricas que agrupam vários critérios (como os 14 critérios legados do LangChain) geram uma saída por critério, com recomendação de uma chamada por critério quando há mais de quatro. Os métodos com formato consagrado mantêm esse formato para permitir reprodução: `[[nota]]` no MT-Bench, `[RESULT] n` no Prometheus, `GRADE: C/I` no Inspect, `safe`/`unsafe` com as categorias S1–S14 no Llama Guard e os cinco níveis de preferência do Arena-Hard.

## Validação, limitações e integração com o júri

A consistência do catálogo foi verificada automaticamente. Os 181 registros têm todos os campos preenchidos, todo prompt contém ao menos um placeholder de entrada e toda métrica composta foi decomposta em critérios com definição própria. Prompts de cada família (Llama Guard, Arena-Hard, G-Eval, Inspect, segurança do Azure, Phoenix) também foram revisados manualmente. Durante essa revisão foram corrigidos rótulos mal formados e a precedência de procedimentos nas rubricas adaptativas do Vertex AI.

O catálogo se conecta ao Gerador de Regras em três pontos:

- **Critérios base:** cada critério estático passa a ter equivalentes de prateleira. Relevância corresponde a Answer Relevancy e RelevanceToQuery; Corretude, a Correctness e Factual Correctness; Clareza, a Coherence e Fluency; Completude, a Completeness e Conversation Completeness.
- **Critérios contextuais:** a classificação da tarefa (Código, Resumo ou Geral) filtra o catálogo pelas colunas de categoria e de tarefas indicadas. O LLM do Estágio 2 passa a escolher e adaptar critérios existentes em vez de inventá-los. A coluna de exigência de referência separa os critérios do Score do Júri dos que só valem quando há anotação humana (Score de Acurácia).
- **Booleanização:** as 76 métricas binárias entram diretamente na votação True/False. Escalas Likert são convertidas por limiar (por exemplo, nota ≥ 4 em 1–5 equivale a True). Métricas de proporção podem usar limiar ou ter cada item decomposto, como cada afirmação suportada, virando um critério booleano próprio. Métricas pairwise ficam reservadas para comparar versões do assistente, não para o veredito por critério.

Há três limitações. Primeiro, os prompts foram escritos a partir das definições e não reproduzem os templates oficiais; precisam ser calibrados contra anotação humana, como prevê a Seção 4.2. Segundo, algumas métricas não usam LLM na ferramenta original (Tool Correctness do DeepEval, avaliadores de segurança do Azure e guardrails locais do Weave); nesses casos o prompt é uma versão LLM-as-judge equivalente, sinalizada no próprio texto. Terceiro, nomes e escalas de plataformas comerciais mudam com frequência, então o catálogo deve ser revisado a cada nova versão do framework.
