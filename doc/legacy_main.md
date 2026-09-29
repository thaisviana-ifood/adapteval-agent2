# `src/main.py` — `AdaptiveJuryAgent` (implementação legada/paralela)

**Não é o caminho usado em produção.** O pipeline real é o grafo LangGraph
em `src/agent/` (ver [agent.md](agent.md)); este arquivo é uma versão mais
antiga da mesma ideia, implementada como uma classe única sem LangGraph, sem
checkpointing e sem o `EvaluatorPanel` mais recente com votação booleana por
critério totalmente alinhado — na verdade ele **usa o mesmo
`EvaluatorPanel`** (`src.jury.EvaluatorPanel`), então se beneficia das
mudanças de rubrica booleana, mas o resto da classe segue a estrutura antiga
"nota 0-10 direto".

Só é referenciado por `example_usage.py` na raiz do projeto.

## Estrutura

`AdaptiveJuryAgent.__init__` instancia **todos** os componentes de todos os
módulos, incluindo os que o pipeline LangGraph atual não usa:
`JuryComposition`, `ThresholdCalculator` (de `metrics/`, ver
[metrics.md](metrics.md)) e `CalibrationManager` (de `memory_manager/`, ver
[memory_manager.md](memory_manager.md)) — mas, assim como no pipeline novo,
**nenhum deles é efetivamente chamado** dentro de `evaluate()`; só ficam
guardados como atributos.

`evaluate(conversation, response, query="")` (método `async`, apesar de não
ter nenhum `await` de verdade lá dentro — é só a assinatura) roda os mesmos
4 estágios manualmente, em sequência, com cache por hash de
`conversation + response`:

```
_analyze_context -> _generate_rules -> _evaluate_with_jury -> _aggregate_metrics -> _update_memory
```

Diferenças notáveis em relação ao pipeline novo (`src/agent/nodes.py`):

- **Sem checkpointing por nó** — se falhar no meio, perde todo o progresso
  daquela avaliação (o `except` em `evaluate()` só loga e relança).
- **`_evaluate_with_jury` passa o `context` cru** (sem achatar
  `task_type`/`complexity_score`) pro `EvaluatorPanel.evaluate` — tem o
  mesmo bug de contexto que existia no pipeline novo antes da correção
  documentada em [agent.md](agent.md), e aqui **não foi corrigido**.
- IDs de avaliação fixos (`"eval_001"`) em vez de gerados por chamada —
  chamadas concorrentes colidiriam no histórico/HITL.
- `_update_memory` sempre registra `task_type="general"` fixo, ignorando o
  `task_type` real classificado em `_generate_rules`.

## Quando isso importa

Se for dar manutenção ou remover código morto do projeto, `main.py` é
candidato: ele duplica a lógica de `src/agent/nodes.py` com uma versão mais
simples e desatualizada. Qualquer fix aplicado no pipeline LangGraph (como a
correção do `jury_context` em `jury_evaluation_node`) **não se propaga
automaticamente para cá** — são implementações independentes que só
compartilham as classes de baixo nível (`ContextAnalyzer`, `EvaluatorPanel`,
etc.), não o fluxo de orquestração.
