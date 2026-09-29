# `src/context_analysis/` — Estágio 1: Análise de Contexto

Recebe a conversa multi-turn crua (string com turnos `"User: ..."` /
`"Assistant: ..."`) e produz um dicionário com 4 dimensões de análise. Todos
os analisadores são **heurísticos/estatísticos** (regex, contagem de
palavras) — nenhum faz chamada de LLM.

`ContextAnalyzer.analyze(conversation)` (`analyzer.py`) apenas roda os 4
abaixo e agrupa o resultado:

```python
{
  "structural": StructuralAnalyzer().analyze(conversation),
  "semantic":   SemanticAnalyzer().analyze(conversation),
  "complexity": ComplexityAnalyzer().analyze(conversation),
  "intent":     IntentAnalyzer().analyze(conversation),
}
```

⚠️ Note que essa é a única estrutura que existe — não há chave `task_type`
nem `complexity_score` no nível raiz. Quem precisar delas (como o
`jury_evaluation_node`) tem que ler `context["complexity"]["complexity_score"]`
e pegar `task_type` de outro lugar (`rule_generation`).

## `structural.py` — `StructuralAnalyzer`

Conta turnos por linha que começa com `"User:"`/`"Assistant:"`. Produz:
`total_turns`, `user_turns`, `assistant_turns`, `speaker_pattern` (lista
`["user","assistant",...]` com repetições consecutivas colapsadas via
`itertools.groupby`), `avg_user_turn_length`/`avg_assistant_turn_length`
(em palavras) e `turn_distribution_ratio` (`user_turns/assistant_turns`).

## `semantic.py` — `SemanticAnalyzer`

- `key_phrases`: janelas de 3 palavras que contêm um termo "importante"
  (`important`, `critical`, `key`, `goal`, `task`, ...).
- `sentiment_score`: razão simples `positive_count / (positive+negative)`
  contra duas listas fixas de palavras (0.5 se nenhuma aparecer).
- `topics`: as 5 palavras (>4 letras) mais frequentes que também estão no
  wordlist em inglês embutido (`resources/english_words.txt`, carregado uma
  vez via `lru_cache`) — por isso os tópicos saem sempre em inglês,
  independente do idioma da conversa.
- `vocabulary_diversity`: `unique_words / total_words`, capado em 1.0.

## `complexity.py` — `ComplexityAnalyzer`

Produz `complexity_score` (0-10, quanto maior mais complexo), combinando:

```
score = word_length_score * 0.3 + sentence_length_score * 0.4 + diversity_score * 0.3
```

onde `word_length_score = min(avg_word_length/0.5, 10)`,
`sentence_length_score = min(avg_sentence_length/1.0, 10)` e
`diversity_score = semantic_diversity * 10`. Também conta termos técnicos
(lista fixa: `algorithm`, `api`, `database`, ...), `?` e `!`.

Esse `complexity_score` é o que `rule_generator/criteria.py` usa para decidir
se adiciona o critério dinâmico "Technical Depth" (limiar > 7), e o que
`metrics/final_score.py` usa no componente `context`.

## `intent.py` — `IntentAnalyzer`

Casamento de palavras-chave (não é NLU de verdade):

- `user_intents`: quais das 7 categorias (`help`, `explain`, `fix`,
  `create`, `analyze`, `learn`, `optimize`) têm alguma keyword presente na
  conversa inteira; fallback `["general_inquiry"]`.
- `assistant_objectives`: mesma ideia para 5 categorias (`inform`, `assist`,
  `validate`, `suggest`, `implement`).
- `intent_alignment_score`: sobreposição entre os dois conjuntos acima,
  normalizada pelo maior conjunto (0.5 se algum lado estiver vazio).
- `intent_segments`: agrupa turnos (split por `\n\n`) por intenção detectada
  turno-a-turno (`question`/`request`/`statement`/`neutral`, via
  `_detect_turn_intent`) — não é muito usado a jusante hoje.
- `primary_intent`: primeiro item de `user_intents`, ou `"unknown"`.

`user_intents` e `topics` (do `semantic.py`) são os dois sinais que
`rule_generator/classifier.py` usa para classificar o tipo de tarefa.
