# `src/context_analysis/` — Estágio 1: Análise de Contexto

Recebe a conversa multi-turn crua (string com turnos `"User: ..."` /
`"Assistant: ..."`) e produz um dicionário com 4 dimensões de análise.
`structural` e `complexity` são **heurísticos/estatísticos** (regex,
contagem de palavras, TF-IDF) — não fazem chamada de LLM. `semantic` e
`intent` chamam LLM, e só para o que exige interpretação (key phrases,
resumo, sentimento; intenções e alinhamento), cada campo com seu
**classificador LLM** e prompt gerenciado via `PromptManager`/Langfuse —
a plumbing comum vive em `src/shared/llm/classifier.py`. `topics` continua
estatístico (TF-IDF), e todo classificador degrada para um caminho sem LLM.

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

Esse isolamento tem um custo: orçamento de tokens curto não quebra nada, só
esvazia o campo. Por isso o `SemanticAnalyzer` constrói seu `LLMClient` com
`max_tokens=CLASSIFIER_LLM_MAX_TOKENS` (default `8000`) em vez do
`LLM_MAX_TOKENS` (2048) — com o default, o classificador de key phrases
gastava os 2048 tokens inteiros em raciocínio e devolvia conteúdo vazio em
toda conversa. Ver [config.md](config.md).

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

Intenção é uma pergunta sobre significado, então o caminho principal é
**semântico**: classificadores LLM, um por campo, cada um com seu prompt
gerenciado via `PromptManager`/Langfuse (mesmo padrão do `semantic.py`):

| prompt | produz |
|---|---|
| `intent_user_classification` | `user_intents` / `primary_intent` |
| `intent_user_classification_closed_set` | idem, restrito a uma taxonomia fixa |
| `intent_assistant_objectives` | `assistant_objectives` |
| `intent_alignment_scoring` | `intent_alignment_score` |

Campos do resultado:

- `user_intents`: verbos únicos em minúsculas, ordenados por importância, com
  `primary_intent` sempre em primeiro. São os mesmos tokens que
  `rule_generator/classifier.py` compara por igualdade exata — daí o
  vocabulário preferencial `INTENT_VOCABULARY`.
- `user_intent_details`: lista paralela a `user_intents`, cada item com
  `label`, `evidence` (citação literal do usuário) e `confidence`.
- `intent_segments`: `{intenção: evidência}` — as citações acima, não mais
  buckets de keyword.
- `assistant_objectives` / `assistant_objectives_confidence`: o que o
  assistente *tentou* fazer (não se conseguiu).
- `intent_alignment_score` + `intent_alignment_rationale` + `unmet_intents`:
  se o assistente engajou com o que o usuário pediu. Mede **alinhamento, não
  qualidade** — errar a resposta certa continua alinhado.
- `primary_intent` / `primary_intent_confidence`.
- `classifier_failures`: nomes dos prompts que degradaram nesta conversa.

**Espaço de labels:** `IntentAnalyzer(taxonomy=[...])` liga o modo *closed
set* — o classificador escolhe `primary_intent` de uma lista fixa e qualquer
label fora dela é descartado. Serve para medir o analyzer contra um goldenset
rotulado, ou para fixá-lo nas intenções do próprio produto. Sem `taxonomy`,
ele emite os verbos de `INTENT_VOCABULARY`.

**Não há classificador determinístico de intenção.** O LLM é o único caminho:
não existe mais nem o mapa de keywords original nem o `_TaxonomyMatcher` que
casava os tokens do próprio label (0.29 e 0.85 de acurácia contra 0.94 do
semântico, e nenhum dos dois lê significado — lê grafia, então responde com a
mesma confiança a paráfrase e a typo). Um segundo espaço de labels para manter
em sincronia não valia isso.

**Degradação:** cada classificador degrada sozinho e registra seu nome em
`classifier_failures`, sem afetar os outros campos — mas o de intenção não tem
para onde degradar, então reporta a falha: `primary_intent` = `"unknown"`,
`user_intents` vazio, e o alinhamento fica sem intenção para comparar e se
declara indefinido (0.5) em vez de 0.0. Os outros dois continuam com fallback
determinístico (`assistant_objectives` por keyword, alinhamento por
`_OBJECTIVE_COMPATIBILITY` + TF-IDF), porque a pergunta deles é sobre um turno
do assistente que já está em mãos, não sobre o que o usuário quis dizer.

Conversa sem turno `Assistant:` não roda os classificadores de objetivo e
alinhamento (não há o que ler), então uma utterance isolada custa **1
chamada**, não 3.

### Por que não é mais keyword matching

Medido sobre o goldenset Bitext de 27.722 utterances
(`scripts/eval_intent_accuracy.py`), o caminho antigo tinha:

- **37% das utterances sem nenhuma keyword**, todas colapsando no fallback —
  as 4 confusões mais frequentes eram todas desse tipo.
- Substring não lê significado: `"how"` em qualquer posição virava
  `explain`, então um pedido para construir algo era rotulado explicação.
- `intent_alignment_score` era praticamente sempre `0.0`: ele intersectava
  `user_intents` com `assistant_objectives`, mas os dois mapas de keyword não
  compartilham nenhuma palavra por construção (`help`/`explain`/`fix` vs
  `inform`/`assist`/`validate`).

Acurácia no goldenset, nas 27 intenções e sem nenhum mapeamento dos labels:
**0.29** (keyword) → **0.937** (`deepseek-v4-pro`, 270 utterances
estratificadas, IC95% ±2.9pp). O `_OBJECTIVE_COMPATIBILITY` substitui a
interseção impossível por um mapa explícito de compatibilidade + similaridade
TF-IDF entre os dois lados da conversa.

Dos 17 erros restantes, 13 estão em quatro pares quase-sinônimos
(`check_invoice`/`get_invoice`, `set_up_shipping_address`/
`change_shipping_address`, `get_refund`/`track_refund`,
`delivery_period`/`track_order`) — a fronteira entre eles é convenção do
Bitext, não semântica: um TF-IDF supervisionado chega a 0.9987 nos mesmos
dados justamente por aprender a convenção dos exemplos. Dar essa convenção ao
prompt (regra de contraste + exemplos por par) é o próximo ganho, e precisa de
amostra maior que 270 para ser validado.

`user_intents` e `topics` (do `semantic.py`) são os dois sinais que
`rule_generator/classifier.py` usa para classificar o tipo de tarefa.
