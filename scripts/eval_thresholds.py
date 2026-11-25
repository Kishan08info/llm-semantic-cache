"""Threshold sweep: paraphrase recall vs false-hit rate, 0.60–0.90.

The cache is pre-populated with the 24 base Q&As. Each threshold is scored
on three disjoint sets:
  - paraphrases (24): should HIT, and on the right base question.
  - near-misses (12): should MISS — a hit serves a wrong answer.
  - unrelated (12): should MISS.
A hit on the wrong base question counts as a wrong hit, not a hit.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

from semantic_cache import MockLLM, SemanticCache
from semantic_cache.text import normalize_tokens

ROOT = Path(__file__).resolve().parent.parent
THRESHOLDS = [0.60, 0.70, 0.80, 0.85, 0.90]


def score_threshold(threshold, llm, qa_pairs, eval_queries):
    cache = SemanticCache(threshold=threshold)
    for pair in qa_pairs:
        cache.put(pair["question"], pair["answer"])

    def base_of(question):
        return llm.base_id_by_key.get(" ".join(normalize_tokens(question)))

    para_right = para_wrong = 0
    for item in eval_queries["paraphrases"]:
        result = cache.get(item["question"])
        if result.hit and base_of(result.matched_question) == item["base_id"]:
            para_right += 1
        elif result.hit:
            para_wrong += 1
    near_false = 0
    near_detail = []
    for item in eval_queries["near_misses"]:
        result = cache.get(item["question"])
        if result.hit:
            near_false += 1
            near_detail.append({
                "question": item["question"],
                "served": result.matched_question,
                "similarity": round(result.similarity, 3),
            })
    unrelated_false = 0
    for item in eval_queries["unrelated"]:
        if cache.get(item["question"]).hit:
            unrelated_false += 1
    n_para = len(eval_queries["paraphrases"])
    n_near = len(eval_queries["near_misses"])
    n_unrel = len(eval_queries["unrelated"])
    return {
        "threshold": threshold,
        "paraphrase_hit_rate": round(para_right / n_para, 4),
        "paraphrase_hits": para_right,
        "paraphrase_wrong_hits": para_wrong,
        "paraphrase_total": n_para,
        "near_miss_false_hit_rate": round(near_false / n_near, 4),
        "near_miss_false_hits": near_false,
        "near_miss_total": n_near,
        "near_miss_false_hit_detail": near_detail,
        "unrelated_false_hit_rate": round(unrelated_false / n_unrel, 4),
        "unrelated_false_hits": unrelated_false,
        "unrelated_total": n_unrel,
    }


def main():
    with open(ROOT / "data" / "qa_pairs.json") as f:
        qa_pairs = json.load(f)["pairs"]
    with open(ROOT / "data" / "eval_queries.json") as f:
        eval_queries = json.load(f)
    llm = MockLLM(ROOT / "data" / "qa_pairs.json",
                  ROOT / "data" / "eval_queries.json")
    results = [score_threshold(t, llm, qa_pairs, eval_queries)
               for t in THRESHOLDS]
    summary = {"thresholds": results}
    out = ROOT / "outputs" / "threshold_sweep.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
