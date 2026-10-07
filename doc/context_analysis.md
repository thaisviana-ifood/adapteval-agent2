# `src/context_analysis/` — Estágio 1: Análise de Contexto

Recebe a conversa multi-turn crua (string com turnos `"User: ..."` /
`"Assistant: ..."`) e produz um dicionário com 4 dimensões de análise.
`structural`, `complexity` e `intent` são **heurísticos/estatísticos**
(regex, contagem de palavras, TF-IDF) — não fazem chamada de LLM.
`semantic` é o único que chama LLM, e só para o que exige interpretação
(key phrases, resumo, sentimento), cada um com seu **classificador LLM** e
prompt gerenciado via `PromptManager`/Langfuse; `topics` continua
estatístico (TF-IDF).

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

Dois tipos de sinal, cada um com a ferramenta que faz melhor o trabalho.

### Estatístico (sem chamada de LLM)

- `topics`: até 5 termos ranqueados por **TF-IDF médio sobre os turnos** da
  conversa (`TfidfVectorizer(stop_words="english")`, mesma ideia do
  `_top_tfidf_terms` do `structural.py`). Um turno = um documento, então um
  termo que aparece em todos os turnos é descontado pelo IDF; conversa sem
  marcadores de turno vira um único documento e o TF-IDF degenera em
  frequência de termo.
  - São **palavras isoladas, minúsculas**, não frases: o
    `rule_generator/classifier.py` casa `topics` com seu mapa de keywords
    por igualdade exata, e um bigrama como `"machine learning"` nunca
    casaria.
  - O token pattern é `\b[a-zA-Z]{3,}\b` — mantém termos curtos que o
    classificador usa (`code`, `step`), que o filtro antigo (5+ letras)
    descartava em silêncio.
  - As stopwords em inglês do sklearn fazem o trabalho de filtragem que o
    wordlist `resources/english_words.txt` fazia — por isso **esse arquivo
    de 2 MB foi removido** e não há mais resource file nenhum aqui.
- `vocabulary_diversity`, `total_tokens`, `unique_tokens` e
  `emoji_sentiment`: aritmética local — LLM contaria pior e mais devagar que
  `len()`.

### Classificadores LLM (um prompt por campo)

Cada um com seu prompt em `src/shared/llm/prompts.py` (e versionado no
Langfuse) — separados de propósito, para poderem ser ajustados
individualmente:

| campo | prompt | trace |
| --- | --- | --- |
| `key_phrases` | `semantic_key_phrase_extraction` | `context_analysis.semantic_key_phrases` |
| `summary` | `semantic_summary_generation` | `context_analysis.semantic_summary` |
| `sentiment_*` | `semantic_sentiment_classification` | `context_analysis.semantic_sentiment` |

- `key_phrases`: até 10 trechos de 2-8 palavras que carregam a substância da
  conversa (pedido, restrições, compromissos) — gentilezas e filler ficam de
  fora. Acompanhado de `key_phrases_confidence` (0-1).
- `summary`: até 3 frases cobrindo o que o usuário queria, o que o assistente
  fez e onde a conversa parou — descritivo, sem julgar o assistente (isso é
  trabalho do júri). Acompanhado de `summary_confidence` (0-1).
- `sentiment_score`: 0-1 do ponto de vista do **usuário** (0 negativo, 0.5
  neutro, 1 positivo) — mesma faixa da versão heurística anterior, então
  quem consome não muda. Acompanhado de `sentiment_label`
  (`positive`/`neutral`/`negative`/`mixed`), `sentiment_confidence` e
  `sentiment_rationale`.

Todos escrevem em inglês independente do idioma da conversa (instrução do
prompt). Cada classificador degrada sozinho: se a chamada falhar ou a
resposta não for um JSON parseável, aquele campo volta vazio/neutro, o nome
do prompt entra em `classifier_failures` e os outros campos não são
afetados. `classifier_failures == []` é o sinal de que a parte LLM saiu
completa; os campos estatísticos nunca degradam.

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
