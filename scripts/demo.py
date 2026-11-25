"""Demo: the same 72-question workload with and without the cache.

Workload: 24 base questions + 24 paraphrases + 12 near-misses + 12
unrelated, shuffled with seed 42. The cache starts warm with the 24 base
Q&As (the realistic state after a day of traffic). Every cache hit is
checked for correctness: the served entry's base question must be the
question's true base question — a hit on the wrong entry is a wrong hit,
not a win.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import random
import time

from semantic_cache import MockLLM, SemanticCache
from semantic_cache.text import normalize_tokens

ROOT = Path(__file__).resolve().parent.parent


def load_workload():
    with open(ROOT / "data" / "qa_pairs.json") as f:
        qa = json.load(f)
    with open(ROOT / "data" / "eval_queries.json") as f:
        ev = json.load(f)
    questions = [(p["question"], p["id"]) for p in qa["pairs"]]
    questions += [(p["question"], p["base_id"]) for p in ev["paraphrases"]]
    questions += [(n["question"], None) for n in ev["near_misses"]]
    questions += [(u["question"], None) for u in ev["unrelated"]]
    rng = random.Random(42)
    rng.shuffle(questions)
    return questions


def run_cached(questions, threshold=0.60):
    llm = MockLLM(ROOT / "data" / "qa_pairs.json",
                  ROOT / "data" / "eval_queries.json")
    cache = SemanticCache(threshold=threshold)
    with open(ROOT / "data" / "qa_pairs.json") as f:
        qa = json.load(f)
    for pair in qa["pairs"]:
        cache.put(pair["question"], pair["answer"])
    wrong_hits = []
    start = time.monotonic()
    for question, true_base in questions:
        result = cache.get(question)
        if result.hit:
            served_base = llm.base_id_by_key.get(
                " ".join(normalize_tokens(result.matched_question)))
            if served_base != true_base:
                wrong_hits.append({
                    "question": question,
                    "served": result.matched_question,
                    "similarity": round(result.similarity, 3),
                })
        else:
            answer, _ = llm.ask(question)
            cache.put(question, answer)
    wall_s = time.monotonic() - start
    stats = cache.summary()
    return {
        "questions": len(questions),
        "wall_clock_s": round(wall_s, 3),
        "llm_calls": llm.calls,
        "llm_calls_avoided": stats["llm_calls_avoided"],
        "exact_hits": stats["exact_hits"],
        "semantic_hits": stats["semantic_hits"],
        "misses": stats["misses"],
        "hit_rate": stats["hit_rate"],
        "wrong_hits": wrong_hits,
        "wrong_hit_count": len(wrong_hits),
        "modelled_latency_saved_s": round(
            stats["llm_calls_avoided"] * llm.latency_s, 2),
        "modelled_cost_saved_usd": round(
            stats["llm_calls_avoided"] * llm.cost_per_call, 4),
        "cache_size": stats["size"],
    }


def run_uncached(questions):
    llm = MockLLM(ROOT / "data" / "qa_pairs.json",
                  ROOT / "data" / "eval_queries.json")
    start = time.monotonic()
    for question, _ in questions:
        llm.ask(question)
    wall_s = time.monotonic() - start
    return {
        "questions": len(questions),
        "wall_clock_s": round(wall_s, 3),
        "llm_calls": llm.calls,
        "modelled_cost_usd": round(llm.calls * llm.cost_per_call, 4),
    }


def main():
    questions = load_workload()
    uncached = run_uncached(questions)
    cached = run_cached(questions)
    summary = {
        "workload": "24 base + 24 paraphrases + 12 near-misses + 12 unrelated, shuffled seed 42",
        "threshold": 0.60,
        "uncached": uncached,
        "cached": cached,
        "calls_saved": uncached["llm_calls"] - cached["llm_calls"],
        "wall_clock_saved_s": round(
            uncached["wall_clock_s"] - cached["wall_clock_s"], 3),
    }
    out = ROOT / "outputs" / "demo_summary.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
