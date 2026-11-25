"""Staleness demo: a price changes, and the cache must not serve the old one.

Sequence (volatile TTL 2s, default TTL 3600s):
  1. Ask the Growth price -> miss -> LLM answers $79, cached as volatile.
  2. Ask the paraphrase -> semantic hit, $79 (correct, price unchanged).
  3. The world changes: Growth is now $99 (mock LLM updated).
  4. Ask again immediately -> still $79: the entry is inside its TTL.
     This is the honest risk window — the TTL is the staleness budget.
  5. Wait past the 2s volatile TTL, ask again -> miss -> LLM answers $99,
     cache updated. Ask once more -> hit, $99.
  6. A non-volatile entry (cancel policy) stays cached across the wait.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import time

from semantic_cache import MockLLM, SemanticCache

ROOT = Path(__file__).resolve().parent.parent


def main():
    llm = MockLLM(ROOT / "data" / "qa_pairs.json",
                  ROOT / "data" / "eval_queries.json", latency_s=0.05)
    cache = SemanticCache(threshold=0.80, default_ttl=3600, volatile_ttl=2)
    events = []

    def ask(question, note):
        result = cache.get(question)
        if result.hit:
            events.append({"step": note, "question": question,
                           "outcome": result.match_type + " hit",
                           "answer": result.answer,
                           "similarity": round(result.similarity, 3)})
            return result.answer
        answer, _ = llm.ask(question)
        cache.put(question, answer)
        events.append({"step": note, "question": question,
                       "outcome": "miss -> LLM",
                       "answer": answer, "similarity": 0.0})
        return answer

    ask("What is the monthly price of the Growth plan?", "1. first ask")
    ask("What does Growth cost per month?", "2. paraphrase, price unchanged")
    llm.update_answer("q02", "The Growth plan costs $99 per month.")
    events.append({"step": "3. world changes",
                   "question": "—",
                   "outcome": "mock LLM: Growth price $79 -> $99",
                   "answer": None, "similarity": 0.0})
    ask("What does Growth cost per month?",
        "4. paraphrase inside volatile TTL (stale $79 served)")
    time.sleep(2.2)
    ask("What does Growth cost per month?",
        "5. paraphrase after volatile TTL expired")
    ask("What does Growth cost per month?",
        "6. paraphrase again (fresh $99 cached)")
    ask("How do I cancel my subscription?",
        "7. non-volatile entry cached before the wait")
    ask("How can I cancel my subscription?",
        "8. non-volatile paraphrase after the wait (still cached)")

    summary = {"volatile_ttl_s": 2, "default_ttl_s": 3600, "events": events}
    out = ROOT / "outputs" / "staleness_summary.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
