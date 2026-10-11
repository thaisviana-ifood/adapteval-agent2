# `src/shared/` — Infraestrutura Compartilhada

## `llm/classifier.py` — `JSONClassifier`

Todo classificador LLM do pipeline tem a mesma forma: formata um prompt do
`PromptManager`, chama o LLM, extrai um objeto JSON da resposta e **degrada
para um caminho sem LLM** se qualquer um desses passos falhar — registrando o
nome do prompt numa lista `classifier_failures` em vez de levantar exceção,
para que um campo indisponível nunca derrube a análise inteira.

`JSONClassifier.call(template_name, trace_name, failures, **variables)` é
esse laço, usado por `context_analysis/semantic.py` e
`context_analysis/intent.py`. As funções `clean_string_list` / `clean_score` /
`clean_confidence` normalizam o que o classificador devolve (é saída de
modelo: pode vir no tipo errado, fora de faixa, duplicada ou em branco).

⚠️ Usa `CLASSIFIER_LLM_TEMPERATURE` (default **0.0**), não o
`LLM_TEMPERATURE` global (0.7): classificar não é gerar — escolher um rótulo
tem uma resposta certa, e amostrar só adiciona variância. Com 0.7, duas runs
sobre as mesmas entradas discordam entre si, o que torna impossível atribuir
uma mudança de acurácia ao prompt em vez de ao sampler.

## `llm/client.py` — `LLMClient`

Wrapper sobre o SDK da OpenAI (via `langfuse.openai.OpenAI`/`AsyncOpenAI`,
que já traceia automaticamente no Langfuse) para qualquer provider
compatível com a API de chat completions (DeepSeek, OpenAI, etc — não serve
para o Typesafe, que tem API própria: ver `typesafe_client.py`).

Pontos importantes:

- **Detecção lazy de "manias" de modelo**: alguns modelos (ex: modelos de
  raciocínio da OpenAI) rejeitam `max_tokens` (querem
  `max_completion_tokens`) ou rejeitam `temperature` customizada. O cliente
  não sabe disso de antemão — ele tenta, pega o 400, reconhece a mensagem de
  erro (`_adapt_to_model_quirk`) e ajusta o `max_tokens_param`/
  `omit_temperature` **daquele modelo** (estado fica no `_Target`, não no
  cliente: o que o primário rejeita não diz nada sobre o fallback), daí
  refaz a chamada. Isso aparece nos logs como
  `"Model 'X' rejects 'max_tokens'; switching to 'max_completion_tokens'"` —
  é esperado, acontece uma vez por modelo, não é erro. A adaptação **não
  gasta tentativa**: duas manias esgotariam o orçamento de 3 antes da
  primeira tentativa real.
- **Retry por target**: até `LLM_MAX_RETRIES` (default 3) tentativas por
  modelo, com backoff exponencial + jitter entre
  `LLM_RETRY_BASE_DELAY` e `LLM_RETRY_MAX_DELAY`, respeitando o
  `Retry-After` do provider quando ele manda um. O jitter importa porque
  essas chamadas rodam em pool: sem ele, todos os workers que bateram no
  mesmo rate limit tentam de novo no mesmo instante.
- **O que é retriável** é a parte que importa — repetir o erro errado
  queima quota e esconde bug:
  - *retry no mesmo modelo*: 429, timeout, conexão caída, 5xx/408/409/425.
  - *pula para o próximo target sem retry*: 401, 403, 404 e o 400 que
    reclama do **nome do modelo** (a DeepSeek responde modelo inexistente
    com 400, não 404). A requisição está certa; esse provider não serve.
  - *levanta na hora, sem retry e sem fallback*: 400/422 de requisição
    malformada — toda provider vai rejeitar igual.
- **Fallback entre providers**: depois do primário vêm os providers de
  `LLM_FALLBACK_PROVIDERS` (default: `openai`; `LLM_FALLBACK_PROVIDERS=`
  vazio desliga), cada um com seu próprio orçamento de retry. Os handles do
  fallback são criados no primeiro uso, então configurar um não custa nada
  até alguém precisar. Esgotada a cadeia, o cliente levanta **o último erro
  real**, não um `RuntimeError` genérico.
  - Toda troca vira um `WARNING` no log e aparece em `get_stats()`:
    `model` é o que foi pedido, `calls_by_model` é quem respondeu de fato,
    mais `fallback_calls`, `retries` e `fallback_chain`.
  - Dois lugares passam `fallbacks=()` de propósito: os jurados (em
    `jury/evaluators.py` — um jurado respondendo com o modelo de outro
    vendor quebra a independência do painel, que é o ponto dele) e
    `scripts/eval_intent_accuracy.py` (o relatório nomeia um modelo; um
    rate limit tem que virar utterance falha na acurácia, não outro modelo
    respondendo metade da corrida em silêncio).
- `call(prompt, system_prompt=None, max_retries=None, **kwargs)`: síncrono,
  devolve `response.choices[0].message.content` (pode vir `""` se o modelo
  gastou todo o orçamento de tokens em raciocínio interno — ver notas sobre
  `JURY_LLM_MAX_TOKENS` e `CLASSIFIER_LLM_MAX_TOKENS` em
  [jury.md](jury.md) e [config.md](config.md)). Quando isso acontece
  (`finish_reason == "length"`), `_content_of` loga um aviso nomeando o
  modelo, o limite atingido e quantos tokens foram para raciocínio, para a
  causa não ficar invisível em quem só recebe a string vazia.
- `acall(...)`: mesma coisa, assíncrono.
- Cada instância de `LLMEvaluator` (em `jury/evaluators.py`) cria seu
  próprio `LLMClient`, então cada provider descobre suas próprias manias
  independentemente.

Os providers ficam num registro só em `src/config.py` (`PROVIDERS`), de onde
saem tanto `JURY_PROVIDERS` (um jurado por provider) quanto
`LLM_FALLBACK_PROVIDERS` (a cadeia do cliente) — um vendor novo se configura
uma vez. O Typesafe está no registro mas nunca entra na cadeia de fallback:
não fala a API de chat completions (`OPENAI_COMPATIBLE_PROVIDERS`). Um
fallback pedido e descartado (nome desconhecido, sem key, não compatível)
sai como aviso em `validate_config()`, para não aparecer só como indisponi-
bilidade na hora errada.

## `llm/typesafe_client.py` — `TypesafeClient`

Cliente HTTP puro (`httpx`, não é OpenAI-compatible) para a API de Q&A
estruturada do Typesafe (https://docs.typesafe.ai). `ask(state, questions,
max_retries=3)` manda `{model, state, questions}` e devolve o JSON de
resposta cru — não faz parsing de verdicts, isso é feito por
`TypesafeEvaluator` (ver [jury.md](jury.md)). Tipos de pergunta suportados
pela API: `"noul"` (yes/no, usado pelo painel hoje), `"choice"`
(múltipla escolha) e `"score"` (escala 0-N — usado pela versão antiga do
painel, antes da rubrica booleana; não é mais chamado pelo
`TypesafeEvaluator` atual). Retry com backoff exponencial em erros
5xx/429/rede; não é traceado automaticamente pelo Langfuse (é `httpx` puro),
por isso `TypesafeEvaluator` envolve a chamada manualmente em
`langfuse.start_as_current_observation` quando o cliente Langfuse está
configurado.

## `llm/prompts.py` — `PromptManager`

Gerencia todos os templates de prompt do pipeline (`context_analysis`,
`rule_generation`, `jury_evaluation` *(órfão — não é chamado por nenhum
evaluator hoje)*, `metrics_aggregation`, `dynamic_criteria_generation`,
`llm_evaluation`, os três classificadores semânticos
`semantic_key_phrase_extraction` / `semantic_summary_generation` /
`semantic_sentiment_classification`, e os quatro de intenção
`intent_user_classification` / `intent_user_classification_closed_set` /
`intent_assistant_objectives` / `intent_alignment_scoring` — ver
`doc/context_analysis.md`), com
**Langfuse Prompt Management como fonte de verdade** e o texto em
`_DEFAULT_PROMPTS` (neste arquivo) como fallback/seed.

⚠️ **Pegadinha importante**: `get_prompt_client(name)` sempre tenta o
Langfuse primeiro (`langfuse.get_prompt(name, fallback=...)`). Se o prompt
já existe lá (de uma run anterior, ou de rodar
`scripts/push_prompts_to_langfuse.py`), **a versão do Langfuse é usada,
mesmo que você tenha editado `_DEFAULT_PROMPTS` neste arquivo**. Editar o
texto aqui só afeta:
1. Ambientes sem `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` configurados
   (`get_langfuse_client()` devolve `None`, cai direto no fallback local).
2. Um prompt que **ainda não existe** no Langfuse (aí o fallback local é
   automaticamente publicado lá na primeira chamada — `is_fallback=True`).

**Depois de editar um template em `_DEFAULT_PROMPTS`, rode
`scripts/push_prompts_to_langfuse.py`** para publicar a nova versão com o
label `production` — senão o pipeline continua usando a versão antiga do
Langfuse silenciosamente.

- `format_prompt(template_name, **variables)`: pega o `PromptClient` do
  Langfuse e chama `.compile(**variables)` (sintaxe `{{var}}`); sem
  Langfuse, faz a substituição localmente (`_compile_locally`, regex
  `\{\{(.*?)\}\}`).
- `register_template(name, template)`: adiciona um template novo local e
  tenta publicar no Langfuse.

## `llm/langfuse_client.py` / `llm/langfuse_handler.py`

Singletons processo-wide, inicialização lazy e "degrade gracefully": sem
`LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` configuradas,
`get_langfuse_client()`/`get_langfuse_callback_handler()` devolvem `None` e
todo o resto do código (tracing, prompt management) simplesmente não
tracea/usa fallback local — nunca quebra o pipeline por falta de Langfuse.
O handler (`CallbackHandler` do `langfuse.langchain`) é o que se passa em
`config={"callbacks":[handler]}` nas chamadas `graph.invoke()`.

## `utils.py`

Funções puras usadas por vários módulos:

- `build_conversation_text(item)`: converte um item de dataset (formato
  Langfuse dataset export: `item["input"]["query"]["content"]` = JSON string
  de mensagens `{role, content}`) no formato texto `"User: ...\n\nAssistant:
  ..."` que todo o resto do pipeline espera. Pula mensagens `role=="system"`.
- `parse_conversation_turns(conversation, delimiter="\n\n")`: parseia de
  volta pra lista de `{role, content}` — usado por `nodes.py` pra extrair a
  última resposta do assistente e a pergunta do usuário anterior.
- `calculate_average`/`calculate_weighted_average`: usadas por
  `metrics/final_score.py`, `metrics/composition.py`,
  `memory_manager/calibration.py`.
- `safe_json_loads`/`safe_json_dumps`: parse/serialização que nunca lança —
  devolvem `default`/`"{}"` em erro, logando um warning.
- `truncate_text`, `timestamp_now`, `format_metrics`: utilitários menores de
  formatação/exibição.

## `logger.py`

`get_logger(name)`: um `logging.Logger` por nome de módulo, configurado uma
vez (console + arquivo rotativo em `LOG_FILE`, 10MB x 5 backups),
`LOG_LEVEL` vindo de `config.py`. Chamar de novo com o mesmo `name` não
duplica handlers (checa `logger.handlers` antes de configurar).

## `infra/database.py` e `infra/queue.py` — em grande parte não usados

- `PostgresConnector` (em `database.py`): **é** usado, por
  `memory_manager/postgres_store.py`. Wrapper simples sobre `psycopg2`
  (`connect`/`execute_query`/`execute_insert`/`execute_update`).
- `VectorStore` (em `database.py`): placeholder — `connect`/`store_vector`/
  `retrieve_similar` só logam e devolvem valores fixos, não fala com
  nenhum banco de vetores de verdade. Não é instanciado em lugar nenhum do
  pipeline.
- `queue.py` (`Queue`, `InMemoryQueue`, `TaskProcessor`): infraestrutura de
  fila assíncrona genérica, não referenciada por nenhum outro módulo do
  projeto — código de andaime para um processamento em lote assíncrono que
  não foi (ainda) plugado em nada.
