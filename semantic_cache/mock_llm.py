"""Mock LLM adapter: a lookup stand-in for a real LLM call.

It answers from the canned knowledge base (exact normalised-text lookup
over the base questions plus the labelled eval questions), sleeps a fixed
latency per call, and charges a fixed modelled cost per call. It is NOT
an LLM — it exists so the cache demo measures real end-to-end behaviour
(calls made, wall-clock time, cost) instead of asserting it.

- `latency_s` (default 0.15) is the SIMULATED per-call latency of a hosted
  LLM. It is a model parameter, disclosed here and in the README, not a
  measured LLM latency.
- `cost_per_call` (default $0.002) is the modelled price of one short
  support answer. Same disclosure.
- `update_answer` simulates the world changing (e.g. a price change) so
  the staleness demo can show a volatile entry expiring and the fresh
  answer being served.
"""

import json
import time

from .text import normalize_tokens

ABSTAIN = "I don't have an answer for that in the Northwind knowledge base."


class MockLLM:
    def __init__(self, qa_path, eval_path=None, latency_s=0.15,
                 cost_per_call=0.002):
        with open(qa_path) as f:
            qa = json.load(f)
        self.answers = {}
        self.base_id_by_key = {}
        self.keys_by_base = {}
        for pair in qa["pairs"]:
            key = " ".join(normalize_tokens(pair["question"]))
            self.answers[key] = pair["answer"]
            self.base_id_by_key[key] = pair["id"]
            self.keys_by_base.setdefault(pair["id"], []).append(key)
        if eval_path:
            with open(eval_path) as f:
                ev = json.load(f)
            for group in ("paraphrases", "near_misses"):
                for item in ev[group]:
                    key = " ".join(normalize_tokens(item["question"]))
                    if group == "paraphrases":
                        base_key = self._key_for_base(item["base_id"])
                        self.answers[key] = self.answers[base_key]
                        self.base_id_by_key[key] = item["base_id"]
                        self.keys_by_base.setdefault(
                            item["base_id"], []).append(key)
                    else:
                        self.answers[key] = item["answer"]
                        self.base_id_by_key[key] = None
            for item in ev["unrelated"]:
                key = " ".join(normalize_tokens(item["question"]))
                self.answers[key] = ABSTAIN
                self.base_id_by_key[key] = None
        self.latency_s = latency_s
        self.cost_per_call = cost_per_call
        self.calls = 0

    def _key_for_base(self, base_id):
        for key, bid in self.base_id_by_key.items():
            if bid == base_id:
                return key
        raise KeyError(f"unknown base id {base_id}")

    def ask(self, question):
        """Return (answer, base_id or None). Sleeps the simulated latency."""
        time.sleep(self.latency_s)
        self.calls += 1
        key = " ".join(normalize_tokens(question))
        return self.answers.get(key, ABSTAIN), self.base_id_by_key.get(key)

    def update_answer(self, base_id, new_answer):
        """Simulate the world changing: the answer for a base question.

        Updates the base question and every paraphrase keyed to it, so the
        mock LLM's answer changes no matter how the question is phrased.
        """
        for key in self.keys_by_base.get(base_id, []):
            self.answers[key] = new_answer
