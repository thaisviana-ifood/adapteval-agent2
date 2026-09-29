# `src/memory_manager/` — Cache, Histórico, Erros e Calibração

Todos esses componentes são instanciados uma vez em `PipelineNodes.__init__`
e vivem pelo tempo de vida do processo (compartilhados entre chamadas de
`run_pipeline`).

## `cache.py` — `CacheManager`

Cache em memória (dict), TTL por entrada (`CACHE_TTL_HOURS`, default 24h) e
limite de tamanho (`CACHE_SIZE_MB`, default 1024MB — ao estourar, evict das
20% entradas mais antigas via `_evict_oldest`).

- `_generate_key(data)`: MD5 do JSON serializado (`safe_json_dumps`) — é
  isso que `PipelineNodes.get_cached_evaluation(conversation)` usa como
  chave: **a chave é o texto da conversa, não o `evaluation_id`**. Duas
  chamadas com o mesmo texto de conversa (mesmo com `evaluation_id`
  diferentes) retornam o mesmo resultado cacheado, sem rodar o grafo de
  novo — isso é importante pra quem escreve testes ou datasets com
  conversas repetidas.
- `get`/`set`/`delete`/`clear`: operações básicas. `get_stats()` devolve
  hit rate, tamanho total, etc.
- Não persiste em disco nem é compartilhado entre processos.

## `history.py` — `HistoryTracker`

Guarda histórico de avaliações e métricas em duas listas em memória
(`evaluation_history`, `metrics_history`). Se construído com um
`PostgresMemoryStore` (`store=...`), toda chamada de `record_evaluation`/
`record_metrics` também persiste lá (best-effort, não bloqueia se falhar).

- `record_evaluation(evaluation_id, task_type, final_score, confidence, components)`:
  chamado por `metrics_aggregation_node` ao final de cada avaliação.
- `get_average_score_by_task()`, `get_performance_trends(window_size=10)`
  (compara média da janela recente vs. a anterior — "improving"/"declining"),
  `get_response_time_stats()`, `clear_old_history(days=30)`: análises sobre
  o histórico acumulado. Nenhuma delas é chamada automaticamente pelo
  pipeline — são ferramentas de inspeção manual/scripts.
- `load_from_store(limit=100)`: hidrata as listas em memória a partir do
  Postgres (útil ao reiniciar o processo e querer continuar o histórico).

## `error_detection.py` — `ErrorDetector`

Validação estrutural pós-hoc de um resultado de avaliação — **não** analisa
o conteúdo da resposta avaliada, analisa se o *dicionário de resultado* está
bem formado:

- `_check_missing_criteria`: campos obrigatórios ausentes
  (`final_score`, `confidence`, `reasoning`).
- `_check_invalid_values`: `final_score` fora de `[0,10]` ou não numérico;
  `confidence` fora de `[0,1]`.
- `_check_inconsistencies`: nota extrema (`<1` ou `>9`) com confiança baixa
  (`<0.5`) é sinalizada como inconsistente.
- `_check_format_issues`: `reasoning` vazio.

`metrics_aggregation_node` chama `detect_errors(final_score, {})` — repare
que passa o **`final_score` breakdown**, não o `evaluation_output` final;
por isso os campos checados (`final_score`, `confidence`, `reasoning`) nem
sempre batem 1:1 com o que existe nesse dict (ex: `reasoning` normalmente
não está presente ali, então o check de formato quase sempre "acerta" por
ausência de campo, não por conteúdo vazio).

## `calibration.py` — `CalibrationManager` (instanciado mas não usado pelo pipeline atual)

Compara notas previstas com notas de referência anotadas e calcula
MAE/RMSE/bias/correlação (`calibrate`), fatores de ajuste
(`apply_calibration`: `(score - offset) * scale`), e uma classificação de
qualidade da calibração (`evaluate_calibration_quality`:
excellent/good/acceptable/poor por MAE+correlação).
`PipelineNodes.__init__` cria `self.calibration = CalibrationManager()` mas
**nenhum método dela é chamado** em `nodes.py` — só é usada (também sem
essas chamadas, aliás) em `src/main.py`, a implementação legada.

## `postgres_store.py` — `PostgresMemoryStore`

Camada de persistência opcional (`USE_POSTGRES_MEMORY=true`) sobre
`shared/infra/database.py::PostgresConnector`. `ensure_schema()` aplica
`docker/postgres/init.sql`. `save_evaluation`/`save_metric` fazem INSERT
simples; `get_evaluation_history`/`get_metrics_history` fazem SELECT
ordenado por `created_at DESC LIMIT n` e devolvem em ordem cronológica
(`reversed`). Toda falha de conexão/query é logada e degrada para `False`/
lista vazia — nunca lança exceção pro caller, então o pipeline principal
funciona normalmente mesmo com Postgres fora do ar (só perde a persistência
extra, mantendo o histórico em memória).
