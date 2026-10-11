# `src/rule_generator/` — Estágio 2: Gerador de Regras

Transforma a análise de contexto em: um tipo de tarefa, objetivos de
sucesso, uma rubrica de critérios booleanos, e um conjunto de checks
heurísticos rápidos (sem LLM). O nó `rule_generation_node` chama, nessa
ordem: `TaskClassifier.classify` -> `ObjectiveDefinition.get_objectives` ->
`CriteriaGenerator.generate` -> `HeuristicChecker.check_all`.

A rubrica em si não vive no código: ela é lida do catálogo em banco
(`criteria_store.py` / tabela `evaluation_criteria`), que diz quais
critérios são obrigatórios, quais se aplicam por tarefa ou por contexto, e
que recebe os critérios criados dinamicamente quando ainda não existem.

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

## `criteria_store.py` — `CriteriaRepository`

**O catálogo de rubricas, e a fonte de verdade da rubrica.** Nenhuma
rubrica é hard-coded no gerador: toda rubrica que o pipeline pode aplicar é
uma linha da tabela `evaluation_criteria` (DDL em
`docker/postgres/init.sql`), com as colunas:

| Coluna | Para quê |
|---|---|
| `name` | Identidade do critério no pipeline inteiro (o júri reporta um veredito por `name`) |
| `description` | A pergunta sim/não que o júri responde (ou a descrição do check, quando heurística) |
| `criterion_type` | `boolean` = pergunta respondida pelo LLM; `heuristic` = check de código em `heuristics.py`, achado por `name` |
| `mandatory` | `true` = entra em **toda** avaliação, qualquer que seja a tarefa |
| `avg_weight` | Média móvel dos pesos com que o critério já foi aplicado; é esse valor que a próxima rubrica usa como `weight` |
| `most_common_tasks` | `{task_type: nº de rubricas daquela tarefa que incluíram este critério}` — é por aqui que a seleção por tarefa funciona |
| `weight_samples` | Quantos pesos a média cobre (sem isso, uma nova amostra só poderia sobrescrever a média, não atualizá-la) |

A unicidade é `(lower(name), criterion_type)`: case-insensitive porque o LLM
propõe nomes com a capitalização dele, e por tipo porque a heurística
`relevance` (check de string) não é a pergunta `Relevance` do júri.

Seleção:

- `get_mandatory(criterion_type)` — os obrigatórios. Se o catálogo não
  tiver nenhum, cai nos `DEFAULT_CRITERIA` com warning: um catálogo vazio
  produziria rubrica vazia e avaliação sem sentido.
- `get_for_task(task_type)` — os **opcionais** que já foram usados nessa
  tarefa ou em uma **tarefa similar** (`SIMILAR_TASKS`, ex:
  `translation` ↔ `text_summarization`; `reasoning` →
  `question_answering`/`classification`/`general`). Ordenados por quantas
  vezes foram usados na própria tarefa, depois nas similares — então
  critério já estabelecido vem primeiro. Nunca devolve `mandatory`
  (já entram sempre; apareceriam duplicados).
- `get_by_name(name, criterion_type)` — um critério específico.

Escrita:

- `get_or_create(...)` — é assim que um critério inventado em runtime (uma
  proposta do LLM, ou um `success_criteria` de uma tarefa nova) **entra no
  catálogo na primeira vez que é usado**; da avaliação seguinte em diante
  ele é uma linha normal, que `get_for_task` oferece para a mesma tarefa e
  para as similares.
- `record_usage(criteria, task_type)` — aplicar uma rubrica é o que
  alimenta as estatísticas: incrementa `most_common_tasks[task_type]` (em
  SQL, com `jsonb_set`, para a contagem não se perder com duas avaliações
  simultâneas) e atualiza a média de `avg_weight`.
- `delete(name, criterion_type)` — aposenta um critério. Por isso o seed só
  roda em catálogo **vazio**: re-inserir os defaults a cada startup
  ressuscitaria silenciosamente um critério deletado de propósito, o que
  faria da tabela um cache em vez da fonte de verdade.
  `seed_defaults()` (usado por `scripts/dump_criteria.py --seed`) re-aplica
  os defaults faltantes sob demanda.

Armazenamento: Postgres (`USE_POSTGRES_CRITERIA`, ligado por default — ver
[config.md](config.md#banco-de-dados--persistência)). Quando o banco não
está acessível (sem psycopg2, sem servidor, flag desligada) o repositório
serve os mesmos `DEFAULT_CRITERIA` de um catálogo em memória e guarda as
adições até o fim do processo — nenhuma avaliação falha por causa do
catálogo, ela só deixa de acumular histórico. `is_persistent` diz qual dos
dois está em uso.

`DEFAULT_CRITERIA` é o seed: todas as rubricas que o código antes
hard-codava (os 4 critérios base, os 2 por gatilho de contexto, os
`success_criteria` dos 4 tipos de tarefa de `objectives.py` e as 4
heurísticas). É a **única** fonte do seed — `init.sql` cria a tabela e não
insere nada.

## `criteria.py` — `CriteriaGenerator`

Monta a **rubrica de uma avaliação** a partir do catálogo: decide *quais*
critérios catalogados se aplicam. Todos saem com
`criterion_type: "boolean"` e `description` fraseada como pergunta sim/não
(isso é o que faz o painel de juízes poder responder True/False critério a
critério em vez de dar uma nota holística — ver [jury.md](jury.md)).

`generate(context, objectives, task_type)` combina 3 fontes:

1. **`_mandatory_criteria`**: todo critério `mandatory` do catálogo —
   no seed, `Relevance`, `Correctness`, `Clarity`, `Completeness`
   (pesos 1.0/1.0/0.8/0.9). O peso vem do `avg_weight` da linha, não do
   código.
2. **`_contextual_criteria`**, nesta ordem (deduplicado por nome):
   - **gatilhos de contexto** (`_CONTEXT_TRIGGERS`): a *condição* fica no
     código porque lê a análise de contexto — `Technical Depth` se
     `complexity_score > 7`, `Problem Solving` se `"fix"` estiver em
     `user_intents` — mas o texto, o peso e o histórico vêm do catálogo.
     Gatilho que dispara para um nome ausente do catálogo não aplica nada
     (é assim que deletar a linha aposenta o critério de verdade);
   - os **must-haves da tarefa** (`objectives["success_criteria"]` /
     `must_haves`), via `get_or_create` — os novos entram no catálogo aqui;
   - o que o catálogo **aprendeu sobre a tarefa** e sobre as similares
     (`get_for_task`), limitado a `_MAX_TASK_CRITERIA = 6`, porque cada
     critério extra é mais um veredito no prompt de cada juiz.
3. **`_generate_dynamic_criteria`**: **essa é a única parte do módulo que
   chama LLM** — usa o prompt `dynamic_criteria_generation` (gerenciado via
   `PromptManager`/Langfuse), com o cliente construído com
   `max_tokens=CLASSIFIER_LLM_MAX_TOKENS` (não o `LLM_MAX_TOKENS` default:
   é um prompt que pede JSON a um modelo de raciocínio e com orçamento
   pequeno devolvia resposta vazia — ver
   [config.md](config.md#classificadores-do-pipeline)), para o modelo propor
   até 3 critérios extras
   específicos do contexto, sem duplicar os já existentes. Parseia um array
   JSON da resposta (`_parse_dynamic_criteria`); qualquer erro de parse ou
   falha de chamada resulta em lista vazia (fail-open, não quebra o
   pipeline). Cada proposta passa por `get_or_create`, então **entra no
   catálogo** e deixa de ser "dinâmica" nas próximas avaliações da mesma
   tarefa ou de tarefas similares — é esse loop que faz a rubrica crescer
   com o uso em vez de ser reinventada a cada chamada.

No fim, `record_usage` registra a rubrica aplicada contra o `task_type`
(contagens e média de peso).

`_assign_weights` normaliza os pesos de todos os critérios combinados
(`normalized_weight = weight / soma_dos_pesos`) — hoje esse peso não é usado
na votação do painel (que é maioria simples por critério, não ponderada),
mas fica disponível em `weighted_criteria` para uso futuro.

Saída: `{ base_criteria, contextual_criteria, dynamic_criteria,
weighted_criteria, total_criteria_count }`. `weighted_criteria` é a lista
completa (base + contextual + dynamic) que o resto do pipeline usa.

## `heuristics.py` — `HeuristicChecker`

Checks puramente sintáticos, sem LLM, sobre a resposta (`response`) e a
pergunta original (`query`). **Quais checks rodam vem do catálogo** (as
linhas `criterion_type = 'heuristic'` e `mandatory = true`); as
implementações são código aqui, ligadas à linha pelo `name`:

| Check | Regra |
|---|---|
| `minimum_length` | `len(response) >= max(len(query)//2, 20)` |
| `no_empty_response` | resposta não é vazia após strip |
| `coherence` | tem ≥2 sentenças e não contém marcadores tipo `"[error]"`, `"sorry"`, `"cannot"` |
| `relevance` | ≥30% das palavras da query aparecem na resposta |

Nome catalogado sem implementação aqui é ignorado com warning (contá-lo
seria reprovar um check que ninguém consegue rodar); catálogo sem nenhuma
linha de heurística cai em rodar todos os checks built-in, para o pipeline
nunca ficar sem o filtro barato pré-júri.

`check_all(response, query)` roda os checks ativos e devolve
`passed_checks`, `failed_checks`, `overall_pass` (AND de todos) e
`pass_rate` (fração que passou). É esse `pass_rate` que vira o componente
`final.heuristic` em `metrics/final_score.py` (`pass_rate * 10`).

## Dump do catálogo — `scripts/dump_criteria.py`

Lê o catálogo que o pipeline usa (Postgres quando acessível, senão os
defaults em memória — e avisa qual dos dois) e escreve em
`data/output/criteria_dump.{json,md,sql}`:

```bash
# JSON (default), tabela Markdown, ou os INSERTs para carregar em outro banco
python3 scripts/dump_criteria.py --format markdown
python3 scripts/dump_criteria.py --format sql --output - | less

# Primeiro insere os defaults faltantes: é como um banco novo ganha a rubrica
python3 scripts/dump_criteria.py --seed
```

`--criterion-type boolean|heuristic` filtra por tipo; `--output -` manda
para stdout. O dump SQL é idempotente (`ON CONFLICT ... DO NOTHING`), então
pode ser reaplicado sobre um catálogo existente sem sobrescrevê-lo.
