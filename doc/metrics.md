# `src/metrics/` — Estágio 4: Agregação das Métricas

## `final_score.py` — `FinalScoreCalculator` (o único usado pelo pipeline atual)

Consolida os 4 estágios anteriores numa única nota 0-10, com pesos fixos:

```python
component_weights = {
    "jury_score": 0.5,
    "heuristic_score": 0.2,
    "accuracy_score": 0.2,
    "context_adjustment": 0.1,
}
```

`calculate_final_score(jury_evaluation, heuristic_check, accuracy_metrics, context)`:

| Componente | Como é calculado |
|---|---|
| `jury_score` | `jury_evaluation["average_score"]` — o `pass_rate * 10` do painel (ver [jury.md](jury.md)), **não** uma nota dada por um juiz |
| `heuristic_score` | `heuristic_check["pass_rate"] * 10` |
| `accuracy_score` | `reference_score + (accuracy - 0.5) * 2`, onde `reference_score`/`accuracy` vêm de `AccuracyCalculator` (5.0/sem confiança se não houver dados anotados) |
| `context_adjustment` | `5.0 + (complexity_normalizada - 0.5) * 2`, capado em `[0,10]` — puxa a nota pra cima em conversas mais complexas |

`final_score = calculate_weighted_average([...], [pesos...])` (de
`shared/utils.py`). `overall_confidence` é outra média ponderada:
`jury_conf*0.5 + heuristic_conf*0.2 + accuracy_conf*0.3` (note que os pesos
aqui **não** batem com `component_weights` — não há peso de confiança pro
componente `context`).

Saída inclui `component_breakdown` (score + peso de cada componente) — é
esse dict que vira as métricas `final.jury`/`final.heuristic`/
`final.accuracy`/`final.context` na saída final do pipeline.

`generate_report(final_score, evaluation_id)`:
- `_score_to_recommendation`: texto por faixa (`>=8` Excellent,
  `>=7` Good, `>=5` Acceptable, `>=3` Needs improvement, senão Poor).
- `_identify_strengths`/`_identify_weaknesses`: qualquer componente
  `>=7`/`<5` vira uma frase tipo `"Strong jury evaluation"`/`"Weak jury
  performance"`.

`set_component_weight`/`get_weights`: permitem reconfigurar os pesos em
runtime — não há caller disso hoje, mas a classe suporta.

## `composition.py` — `JuryComposition` (não usado pelo pipeline atual)

Sistema de votação genérico e **independente** do `EvaluatorPanel` real:
jurados são adicionados manualmente (`add_juror(id, type, expertise)`),
votam com `add_vote(juror_id, score, confidence)`, e `aggregate_votes()`
calcula uma média ponderada por expertise + um "nível de concordância"
baseado no desvio-padrão dos scores. Só é instanciado em `src/main.py`
(implementação legada) — o `PipelineNodes`/LangGraph atual usa a votação
booleana por critério dentro do próprio `EvaluatorPanel`, não esta classe.

## `threshold.py` — `ThresholdCalculator` (não usado pelo pipeline atual)

Calcula um threshold de decisão adaptativo (`calculate_threshold`, ajustado
por complexidade/tamanho do júri/nº de critérios) e aplica decisões
pass/fail (`apply_threshold`) ou taxas de aprovação (`calculate_pass_rate`).
Também só é referenciado em `src/main.py`. Se algum dia quiser um "aprovado/
reprovado" com threshold configurável em vez do `pass_rate` bruto do painel,
é aqui que já existe a lógica pronta — só falta plugar no
`metrics_aggregation_node`.
