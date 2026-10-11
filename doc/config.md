# `src/config.py` — Configuração

Tudo lido de variáveis de ambiente (via `python-dotenv`, `.env` na raiz do
projeto) com defaults sensatos. `validate_config()` só checa se
`LLM_API_KEY`/`LLM_MODEL` estão setados (aviso, não bloqueia).

## Caminhos

| Variável | Default | Uso |
|---|---|---|
| `PROJECT_ROOT`, `DATA_DIR`, `RAW_DATA_DIR`, `ANNOTATED_DATA_DIR`, `OUTPUT_DATA_DIR` | relativos a `src/config.py` | Criados automaticamente no import (`mkdir(parents=True, exist_ok=True)`). `scripts/run_deep_agent.py` usa `RAW_DATA_DIR`/`OUTPUT_DATA_DIR` como defaults de `--dataset`/`--output`. |

## LLM do agente conversacional (não é o júri)

| Variável | Default | Uso |
|---|---|---|
| `LLM_MODEL` | `deepseek-v4-pro` | Modelo do `deep_agent.py` (front-end conversacional), lido de `LLM_MODEL` |
| `LLM_API_KEY` | via `DEEP_SEEK_API_KEY` | idem |
| `LLM_BASE_URL` | `https://api.deepseek.com` | idem |
| `LLM_TEMPERATURE` | `0.7` | idem |
| `LLM_MAX_TOKENS` | `2048` | idem — **não** é o orçamento usado pelos juízes nem pelos classificadores do pipeline, ver `JURY_LLM_MAX_TOKENS` e `CLASSIFIER_LLM_MAX_TOKENS` |
| `LLM_TIMEOUT` | `60` (s) | idem |

## Retry e fallback (todas as chamadas via `LLMClient`)

| Variável | Default | Uso |
|---|---|---|
| `LLM_MAX_RETRIES` | `3` | Tentativas **por modelo** antes da cadeia passar para o próximo; sobrescrevível por chamada (`call(..., max_retries=N)`) |
| `LLM_RETRY_BASE_DELAY` | `1.0` (s) | Base do backoff exponencial, com jitter (`uniform(d/2, d)`) para um pool não tentar de novo em uníssono |
| `LLM_RETRY_MAX_DELAY` | `30.0` (s) | Teto de cada espera, inclusive de um `Retry-After` exagerado do provider |
| `LLM_FALLBACK_PROVIDERS` | `openai` | Nomes de `PROVIDERS`, em ordem, tentados depois do primário. Vazio desliga o fallback |

`PROVIDERS`: registro único de vendors (`deepseek` via `LLM_MODEL_DEEP_SEEK`/
`DEEP_SEEK_API_KEY`/`DEEP_SEEK_BASE_URL`, `openai` via `LLM_MODEL_OPEN_AI`/
`OPEN_AI_API_KEY`/`OPEN_AI_BASE_URL`, `typesafe` via `LLM_MODEL_TYPESAFE`/
`TYPESAFE_API_KEY`/`TYPESAFE_BASE_URL`), de onde saem `JURY_PROVIDERS` e
`LLM_FALLBACK_PROVIDERS`. Só os de `OPENAI_COMPATIBLE_PROVIDERS`
(`deepseek`, `openai`) podem ser fallback — o Typesafe tem API própria.
Entradas sem `api_key` ou sem `model` são descartadas da cadeia (seriam só
uma forma mais lenta de falhar), e `validate_config()` avisa quais nomes
pedidos foram ignorados. Qual erro é retriável, qual pula de provider e qual
levanta na hora está em [shared.md](shared.md#llmclientpy--llmclient).

## Júri (`EvaluatorPanel`)

`JURY_PROVIDERS`: um dict (`name`, `model`, `api_key`, `base_url`) por juiz,
derivado de `PROVIDERS` na mesma ordem. `name == "typesafe"` é o que faz
`EvaluatorPanel._init_evaluators` escolher `TypesafeEvaluator` em vez de
`LLMEvaluator` para essa entrada. Cada `LLMEvaluator` constrói seu cliente
com `fallbacks=()`: um jurado respondendo com o modelo de outro vendor
quebraria a independência do painel.

`JURY_LLM_MAX_TOKENS` (default `8000`): orçamento de tokens passado só pros
`LLMEvaluator`s (não afeta `LLM_MAX_TOKENS`/o agente conversacional, nem o
Typesafe, que não usa esse parâmetro). Precisa ser bem maior que o padrão
porque modelos de raciocínio gastam parte do orçamento em tokens de
raciocínio ocultos antes de escrever o veredito visível — com uma rubrica de
muitos critérios, um orçamento pequeno pode esgotar em raciocínio e devolver
texto vazio/cortado (`finish_reason: "length"`). Ver [jury.md](jury.md).

## Classificadores do pipeline

`CLASSIFIER_LLM_MAX_TOKENS` (default `8000`): orçamento dos estágios que não
são o júri mas também pedem JSON estruturado a um modelo de raciocínio — os
três classificadores do `SemanticAnalyzer`
([context_analysis.md](context_analysis.md)) e a geração de critérios
dinâmicos do `CriteriaGenerator` ([rule_generator.md](rule_generator.md)).

Mesma causa do `JURY_LLM_MAX_TOKENS`, mas com uma falha **silenciosa**: cada
classificador degrada de forma independente, então um orçamento pequeno não
levanta erro — devolve o campo vazio e só registra o nome do prompt em
`classifier_failures`. Medido no `deepseek-v4-pro`, o prompt de key phrases
gasta ~7.4k tokens de raciocínio numa única conversa, ou seja: com o default
`LLM_MAX_TOKENS` (2048) ele nunca produzia saída alguma
(`finish_reason: "length"`, `completion_tokens == reasoning_tokens`, conteúdo
vazio). O `LLMClient` agora loga um aviso nomeando essa causa sempre que
`finish_reason == "length"`, em vez de devolver `""` sem explicação.

## Banco de dados / persistência

| Variável | Default | Uso |
|---|---|---|
| `DB_HOST`/`DB_PORT`/`DB_USER`/`DB_PASSWORD`/`DB_NAME` | `localhost`/`5432`/`admin`/``/`adapteval_agent` | Postgres, usado por `PostgresConnector` e no DSN do `PostgresSaver` (checkpointer do LangGraph) |
| `USE_POSTGRES_MEMORY` | `False` | Liga persistência do histórico de avaliações (`HistoryTracker`) e do checkpointer do LangGraph em Postgres; sem isso, tudo fica só em memória do processo |
| `VECTOR_DB_HOST`/`VECTOR_DB_PORT`/`VECTOR_DB_NAME` | `localhost`/`6379`/`adapteval_vectors` | Lidas mas só consumidas pelo `VectorStore` placeholder (não conectado a nada real) |

## Logging / processamento

| Variável | Default | Uso |
|---|---|---|
| `LOG_LEVEL` | `INFO` | Nível de log de todos os loggers (`shared/logger.py`) |
| `LOG_FILE` | `logs/adapteval.log` | Arquivo de log rotativo (10MB x 5 backups) |
| `BATCH_SIZE`, `NUM_WORKERS`, `ASYNC_PROCESSING` | `32`/`4`/`False` | Lidas mas não consumidas por nenhum módulo do pipeline atual |

## Avaliação

| Variável | Default | Uso |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | `0.7` | Default de `HumanInTheLoop.should_request_review` (mas o valor real passado por `jury_evaluation_node` usa o default do método, não lê esta env var diretamente — cuidado ao reconfigurar) |
| `MIN_EVALUATORS` | `3` | Documental — bate com o tamanho de `JURY_PROVIDERS`, mas `EvaluatorPanel` não valida esse mínimo em runtime |
| `USE_HUMAN_IN_THE_LOOP` | `True` | Lida mas não checada em nenhum `if` do pipeline atual — o HITL é sempre acionado quando a confiança é baixa, independente desta flag |
| `CACHE_SIZE_MB`, `CACHE_TTL_HOURS` | `1024`/`24` | `CacheManager` (cache de avaliação por texto de conversa) |
| `ENABLE_CALIBRATION` | `True` | Lida mas não checada em nenhum lugar — `CalibrationManager` nem é chamado pelo pipeline atual (ver [memory_manager.md](memory_manager.md)) |

## Langfuse

| Variável | Default | Uso |
|---|---|---|
| `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` | `""` | Sem ambos, tracing e prompt management ficam desligados (fallback local) em todo o pipeline |
| `LANGFUSE_HOST` (env `LANGFUSE_BASE_URL`) | `https://cloud.langfuse.com` | Host da instância Langfuse |
