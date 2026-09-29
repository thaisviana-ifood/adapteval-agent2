# `src/shared/` — Infraestrutura Compartilhada

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
  erro (`_adapt_to_model_quirk`) e ajusta `self._max_tokens_param`/
  `self._omit_temperature` **permanentemente para aquela instância**, daí
  refaz a chamada. Isso aparece nos logs como
  `"Model 'X' rejects 'max_tokens'; switching to 'max_completion_tokens'"` —
  é esperado, acontece uma vez por instância de `LLMClient`, não é erro.
- **Retry**: até `max_retries` (default 3) com backoff exponencial em
  `RateLimitError` e `APIError` genérico.
- `call(prompt, system_prompt=None, max_retries=3, **kwargs)`: síncrono,
  devolve `response.choices[0].message.content` (pode vir `""` se o modelo
  gastou todo o orçamento de tokens em raciocínio interno — ver nota sobre
  `JURY_LLM_MAX_TOKENS` em [jury.md](jury.md) e [config.md](config.md)).
- `acall(...)`: mesma coisa, assíncrono.
- Cada instância de `LLMEvaluator` (em `jury/evaluators.py`) cria seu
  próprio `LLMClient`, então cada provider descobre suas próprias manias
  independentemente.

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
`llm_evaluation`), com **Langfuse Prompt Management como fonte de verdade**
e o texto em `_DEFAULT_PROMPTS` (neste arquivo) como fallback/seed.

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
