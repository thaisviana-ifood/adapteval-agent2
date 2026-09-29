# `src/rule_generator/` — Estágio 2: Gerador de Regras

Transforma a análise de contexto em: um tipo de tarefa, objetivos de
sucesso, uma rubrica de critérios booleanos, e um conjunto de checks
heurísticos rápidos (sem LLM). O nó `rule_generation_node` chama, nessa
ordem: `TaskClassifier.classify` -> `ObjectiveDefinition.get_objectives` ->
`CriteriaGenerator.generate` -> `HeuristicChecker.check_all`.

## `classifier.py` — `TaskClassifier`

`classify(context)` é **rule-based, não usa LLM** (apesar de instanciar um
`LLMClient`/`PromptManager` no `__init__` — não são usados no método
`classify` atual). Casa `user_intents` + `topics` (de `context_analysis`)
contra um dicionário fixo de keywords por tipo de tarefa
(`question_answering`, `code_generation`, `text_summarization`,
`instruction_following`, `creative_writing`); sem match, retorna `"general"`.
Confiança é sempre fixa em `0.7`. Também devolve `templates` e
`recommended_evaluators` (listas fixas por tipo de tarefa) — hoje não são
consumidos por mais nenhum lugar do pipeline atual.

## `objectives.py` — `ObjectiveDefinition`

Tabela estática (`objectives_map`) por tipo de tarefa
(`question_answering`, `code_generation`, `text_summarization`, `general` —
qualquer outro tipo cai em `general`), cada entrada com `primary`,
`secondary` e `success_criteria`.

- `get_objectives(task_type)`: devolve a entrada da tabela.
- `define_success(task_type, additional_criteria=None)`: enriquece com
  `must_haves` (= `success_criteria`), `nice_to_haves` (=
  `additional_criteria`) e `failure_conditions` (outra tabela estática,
  `_get_failure_conditions`). É essa saída que `CriteriaGenerator` consome
  como `objectives`.
- `define_subgoals(task_type)`: checkpoints intermediários
  (`code_generation`/`general`) — não usado pelo pipeline atual.

## `criteria.py` — `CriteriaGenerator`

Gera a **rubrica**: uma lista de critérios, todos com
`criterion_type: "boolean"` e `description` fraseada como pergunta sim/não
(isso é o que faz o painel de juízes poder responder True/False critério a
critério em vez de dar uma nota holística — ver [jury.md](jury.md)).

`generate(context, objectives)` combina 3 fontes:

1. **`_extract_base_criteria`**: 4 critérios sempre presentes — `Relevance`,
   `Correctness`, `Clarity`, `Completeness` (pesos 1.0/1.0/0.8/0.9).
2. **`_generate_contextual_criteria`**: condicional ao contexto —
   `Technical Depth` se `complexity_score > 7`; `Problem Solving` se
   `"fix"` estiver em `user_intents`; e um critério por item em
   `objectives["success_criteria"]`/`must_haves` (ex: os 3
   `success_criteria` de `general` viram 3 critérios extras).
3. **`_generate_dynamic_criteria`**: **essa é a única parte do módulo que
   chama LLM** — usa o prompt `dynamic_criteria_generation` (gerenciado via
   `PromptManager`/Langfuse) para o modelo propor até 3 critérios extras
   específicos do contexto, sem duplicar os já existentes. Parseia um array
   JSON da resposta (`_parse_dynamic_criteria`); qualquer erro de parse ou
   falha de chamada resulta em lista vazia (fail-open, não quebra o
   pipeline).

`_assign_weights` normaliza os pesos de todos os critérios combinados
(`normalized_weight = weight / soma_dos_pesos`) — hoje esse peso não é usado
na votação do painel (que é maioria simples por critério, não ponderada),
mas fica disponível em `weighted_criteria` para uso futuro.

Saída: `{ base_criteria, contextual_criteria, dynamic_criteria,
weighted_criteria, total_criteria_count }`. `weighted_criteria` é a lista
completa (base + contextual + dynamic) que o resto do pipeline usa.

## `heuristics.py` — `HeuristicChecker`

Checks puramente sintáticos, sem LLM, sobre a resposta (`response`) e a
pergunta original (`query`):

| Check | Regra |
|---|---|
| `minimum_length` | `len(response) >= max(len(query)//2, 20)` |
| `no_empty_response` | resposta não é vazia após strip |
| `coherence` | tem ≥2 sentenças e não contém marcadores tipo `"[error]"`, `"sorry"`, `"cannot"` |
| `relevance` | ≥30% das palavras da query aparecem na resposta |

`check_all(response, query)` roda os 4 e devolve `passed_checks`,
`failed_checks`, `overall_pass` (AND de todos) e `pass_rate` (fração que
passou). É esse `pass_rate` que vira o componente `final.heuristic` em
`metrics/final_score.py` (`pass_rate * 10`).
