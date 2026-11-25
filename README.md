# Semantic Cache for LLM/RAG Responses

A semantic cache that answers repeat and rephrased questions from cache
instead of calling the LLM again — with an entity-consistency guard and
per-answer TTLs so it never serves a confidently wrong price.

> **All data is SYNTHETIC** (fictional "Northwind Analytics" SaaS support
> knowledge base, seeded). This is a portfolio skill demo, not work
> experience. The "LLM" is a mock adapter with a fixed simulated latency —
> see [Honest labelling](#honest-labelling).

## The business question

Support chatbots and RAG assistants get asked the same question a hundred
ways: "What is the Starter plan price?", "How much does Starter cost per
month?", "What does Starter cost?" Every one of those is an LLM call with
its own latency and cost — even though the answer is identical. A semantic
cache stores the answer once and serves it for any question *similar
enough* to the cached one. The hard part is "similar enough": serve the
Starter price for a Growth price question and you have confidently given
the wrong answer.

## How it works

```
question -> exact match? -> semantic match (TF-IDF cosine >= 0.60
              |               + entity guard) -> HIT: serve cached answer
              |                                  MISS: call LLM, cache answer
              +-- expired? -> treat as miss, refresh
```

- **Similarity** (`semantic_cache/embeddings.py`) — pure-stdlib TF-IDF
  vectors over normalised question tokens, cosine similarity. IDF is fit
  on the cached questions, so the *discriminating* token (the plan name
  in "Starter plan price" vs "Growth plan price") weighs more than tokens
  that appear in every question ("plan", "price").
- **Normalisation** (`semantic_cache/text.py`) — lowercasing, stopword
  removal, and a small general synonym map (cost→price, monthly→month,
  seats→seat, guarantee→SLA, paying→billing). It is the lexical stand-in
  for an embedding model.
- **Entity guard** (`semantic_cache/entities.py`) — the finding of this
  project. Similarity alone cannot separate a paraphrase from a
  near-miss: "What is the monthly price of the Growth plan?" is
  0.82-similar to the cached Starter question. Before serving, the cache
  checks that the *discriminating entities* agree: plan
  (starter/growth/pro/enterprise) and format (csv/json) must match
  exactly; action (cancel/upgrade/downgrade) and region (us/eu) block
  only on direct conflict. Without the guard, 11 of 12 near-misses
  false-hit at threshold 0.60; with it, 1 of 12.
- **Freshness policy** (`semantic_cache/policies.py`) — not every answer
  goes stale at the same speed. Price/limit/seat questions are *volatile*
  (TTL 60 s by default); stable answers live 3600 s. The TTL is the
  staleness budget: inside it, a changed price is still served stale —
  the demo shows that window honestly.
- **Eviction** — LRU at `max_size` (default 100), lazy TTL expiry on
  access plus a sweep on put. Entries persist to JSON (`save`/`load`).

## Results — real numbers from actual runs

`python run.py` runs the demo, the threshold sweep, and the staleness
demo; `outputs/summary.json` reproduces byte-identically on a from-scratch
re-run (delete `outputs/` and re-run). 33/33 pytest checks pass.

### Demo workload: 72 questions, cache warm with the 24 base Q&As

| | Without cache | With cache |
|---|---|---|
| LLM calls | 72 | **27** |
| Wall-clock time | 10.81 s | **4.08 s** |
| Modelled cost (@ $0.002/call) | $0.144 | **$0.054** |

45 calls avoided (62.5% hit rate: 26 exact + 19 semantic hits, 27
misses). Modelled latency saved 6.75 s, cost saved $0.09. **1 wrong
hit**: "What is the annual price of the Starter plan?" was served the
monthly price (similarity 0.624) — the one near-miss the guard cannot
catch, because both questions name the Starter plan.

Wall-clock times are measured on this machine and vary run to run; the
counts, rates, and modelled savings are deterministic.

### Threshold sweep (24 paraphrases / 12 near-misses / 12 unrelated)

| Threshold | Paraphrase hit rate | Near-miss false-hit rate | Unrelated false-hit rate |
|---|---|---|---|
| 0.60 | **0.833** (20/24) | 0.083 (1/12) | 0.0 |
| 0.70 | 0.458 (11/24) | 0.0 | 0.0 |
| 0.80 | 0.292 (7/24) | 0.0 | 0.0 |
| 0.85 | 0.292 (7/24) | 0.0 | 0.0 |
| 0.90 | 0.167 (4/24) | 0.0 | 0.0 |

**Default threshold is 0.60** — the knee of the curve. Raising it kills
false hits but paraphrase recall collapses (0.83 → 0.29 at 0.80), because
lexical similarity is weak on reworded questions. Zero wrong-base hits at
every threshold: every served paraphrase got the right entry.

The 4 paraphrases missed at 0.60, and why (all honest lexical limits):
- "Can I get my money back if I cancel?" (sim 0.260) — "cancel" matches
  the cancel question, not the refund policy.
- "How many days does the free trial last?" (0.504) — "days/last" vs
  "long" share no tokens.
- "How many seats come with Starter?" (0.567) — the extra token "come"
  drags the score down.
- "Can I download my data as a CSV?" (0.731) — similarity passes, but
  the entity guard blocks it: the cached question never says CSV, even
  though the cached *answer* covers CSV. The guard compares question
  entities only — a real limitation, see below.

### Staleness demo (volatile TTL 2 s in the demo)

1. "What is the monthly price of the Growth plan?" → miss → LLM: **$79**.
2. Paraphrase "What does Growth cost per month?" → semantic hit, $79.
3. The world changes: Growth is now **$99**.
4. Paraphrase again inside the TTL → semantic hit, **$79 (stale — by
   design; the TTL is the staleness budget)**.
5. After the 2 s volatile TTL expires → miss → LLM: **$99**, cache updated.
6. Paraphrase again → hit, $99. A non-volatile entry (cancel policy)
   stays cached across the whole wait.

### The guard, before and after (threshold 0.60)

| | Without guard | With guard |
|---|---|---|
| Near-miss false hits | 11/12 (0.917) | **1/12 (0.083)** |
| Paraphrase recall | 18/24 (0.750) | 17/24 → **20/24 (0.833)** after synonym fixes |

(The recall change from 17 to 20 is the three general synonym mappings
added during the build: seats→seat, guarantee→SLA, paying→billing.)

## Usage

```bash
python run.py                        # demo + sweep + staleness -> outputs/
python -m pytest tests/ -q           # 33 checks
python scripts/demo.py               # cached vs uncached workload only
python scripts/eval_thresholds.py    # threshold sweep only
python scripts/staleness_demo.py     # price-change scenario only
```

```python
from semantic_cache import SemanticCache, MockLLM

cache = SemanticCache(threshold=0.60, max_size=100,
                      default_ttl=3600, volatile_ttl=60)
llm = MockLLM("data/qa_pairs.json", "data/eval_queries.json")

def answer(question):
    result = cache.get(question)
    if result.hit:
        return result.answer          # exact or semantic hit
    answer, _ = llm.ask(question)     # miss: call the LLM
    cache.put(question, answer)
    return answer
```

## Data

| File | Contents |
|---|---|
| `data/qa_pairs.json` | 24 base Q&As (fictional Northwind SaaS support) |
| `data/eval_queries.json` | 24 paraphrases (should hit), 12 near-misses (should miss — same template, different answer), 12 unrelated (should miss) |

## Known limitations

- **Lexical similarity, not embeddings.** TF-IDF + synonyms is a stand-in.
  A real deployment would use an embedding model; the paraphrase recall
  ceiling here (0.83) is a property of the stand-in, not of semantic
  caching. The guard is what makes the lexical version safe.
- **The guard compares question entities only.** A cached answer can cover
  an entity its question never names (CSV export) — the guard blocks that
  correct hit. Answer-side entity extraction would fix it.
- **Monthly vs annual is uncatchable by this guard.** Both name the same
  plan; only the time unit differs, and "monthly"/"annual" are not in the
  entity groups. That is the 1 remaining false hit.
- **The mock LLM is a lookup**, not a model. Its latency ($0.15 s/call)
  and cost ($0.002/call) are model parameters, disclosed here — the
  savings are modelled, the hit/miss counts are measured.
- **Single-node, in-memory.** No distributed invalidation, no cache
  warming from production traffic, no per-user scoping (a cached answer
  for one user is served to all — fine for public support answers, wrong
  for personal data).

## Why this matters for the roles I'm targeting

TribalScale's AI Engineer posting lists "evaluation, semantic caching"
among the core skills; Abby Care's Applied AI Engineer asks for agents
that "measure performance, analyze failures, improve models, prompts,
retrieval, tools"; Act's Applied AI Engineer asks for "evaluations to
measure task completion, response quality, system reliability". A
semantic cache is the cost/latency layer those systems need — and the
guard + TTL policy are the difference between a cache that saves money
and one that serves wrong answers with confidence.

## Honest labelling

Synthetic data everywhere (seeded, seed 42 for the workload shuffle).
TF-IDF similarity stands in for embedding similarity; the mock LLM stands
in for a hosted model. No production semantic-cache (GPTCache,
Redis Semantic Cache) experience is claimed — the cache logic, guard,
TTL policy, and eviction are a real implementation of the algorithm, not
a simulation.
