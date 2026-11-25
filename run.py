"""Run the demo, the threshold sweep, and the staleness demo; write summary.

`outputs/summary.json` holds only deterministic metrics (counts, rates,
modelled cost/latency) so a from-scratch re-run reproduces it
byte-identically. Measured wall-clock times live in demo_summary.json
and vary run to run.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from demo import main as demo_main  # noqa: E402
from eval_thresholds import main as sweep_main  # noqa: E402
from staleness_demo import main as stale_main  # noqa: E402


def main():
    demo = demo_main()
    sweep = sweep_main()
    stale = stale_main()
    sweep_at_080 = next(
        t for t in sweep["thresholds"] if t["threshold"] == 0.60)
    summary = {
        "project": "llm-semantic-cache",
        "workload_questions": demo["cached"]["questions"],
        "threshold_default": 0.60,
        "demo": {
            "llm_calls_without_cache": demo["uncached"]["llm_calls"],
            "llm_calls_with_cache": demo["cached"]["llm_calls"],
            "llm_calls_avoided": demo["cached"]["llm_calls_avoided"],
            "exact_hits": demo["cached"]["exact_hits"],
            "semantic_hits": demo["cached"]["semantic_hits"],
            "misses": demo["cached"]["misses"],
            "hit_rate": demo["cached"]["hit_rate"],
            "wrong_hits": demo["cached"]["wrong_hit_count"],
            "modelled_latency_saved_s": demo["cached"]["modelled_latency_saved_s"],
            "modelled_cost_saved_usd": demo["cached"]["modelled_cost_saved_usd"],
        },
        "threshold_sweep": {
            str(t["threshold"]): {
                "paraphrase_hit_rate": t["paraphrase_hit_rate"],
                "paraphrase_hits": t["paraphrase_hits"],
                "paraphrase_wrong_hits": t["paraphrase_wrong_hits"],
                "near_miss_false_hit_rate": t["near_miss_false_hit_rate"],
                "near_miss_false_hits": t["near_miss_false_hits"],
                "unrelated_false_hit_rate": t["unrelated_false_hit_rate"],
                "unrelated_false_hits": t["unrelated_false_hits"],
            } for t in sweep["thresholds"]
        },
        "threshold_0.60": {
            "paraphrase_hit_rate": sweep_at_080["paraphrase_hit_rate"],
            "near_miss_false_hit_rate": sweep_at_080["near_miss_false_hit_rate"],
            "unrelated_false_hit_rate": sweep_at_080["unrelated_false_hit_rate"],
        },
        "staleness": {
            "volatile_ttl_s": stale["volatile_ttl_s"],
            "events": [
                {"step": e["step"], "outcome": e["outcome"],
                 "answer": e["answer"]}
                for e in stale["events"]
            ],
        },
    }
    out = ROOT / "outputs" / "summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out}")
    return summary


if __name__ == "__main__":
    main()
