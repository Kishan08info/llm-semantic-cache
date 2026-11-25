"""Tests for the semantic cache: hits, guard, TTL, eviction, persistence."""

import json
from pathlib import Path

import pytest

from semantic_cache import MockLLM, SemanticCache
from semantic_cache.text import normalize_tokens

ROOT = Path(__file__).resolve().parent.parent


class FakeClock:
    def __init__(self, start=1_000_000.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def _warm_cache(**kwargs):
    cache = SemanticCache(**kwargs)
    with open(ROOT / "data" / "qa_pairs.json") as f:
        for pair in json.load(f)["pairs"]:
            cache.put(pair["question"], pair["answer"])
    return cache


def test_exact_hit():
    cache = _warm_cache()
    result = cache.get("What is the monthly price of the Starter plan?")
    assert result.hit and result.match_type == "exact"
    assert result.answer == "The Starter plan costs $29 per month."
    assert result.similarity == 1.0


def test_exact_hit_ignores_case_and_punctuation():
    cache = _warm_cache()
    result = cache.get("WHAT is the monthly price of the starter plan?!")
    assert result.hit and result.match_type == "exact"


def test_paraphrase_semantic_hit_serves_right_entry():
    cache = _warm_cache()
    result = cache.get("How much does the Starter plan cost each month?")
    assert result.hit and result.match_type == "semantic"
    assert result.answer == "The Starter plan costs $29 per month."
    assert result.matched_question == "What is the monthly price of the Starter plan?"


def test_near_miss_blocked_by_entity_guard():
    cache = _warm_cache()
    # No Team plan exists; must not serve the Starter price.
    assert not cache.get("What is the monthly price of the Team plan?").hit
    # SSO on Starter must not serve the plan-less SSO answer.
    assert not cache.get("Is SSO with SAML available on the Starter plan?").hit
    # Audit logs on Pro must not serve the plan-less audit-log answer.
    assert not cache.get("Are audit logs available on the Pro plan?").hit
    # Downgrade must not serve the upgrade answer.
    assert not cache.get("How do I downgrade my plan?").hit


def test_unrelated_misses():
    cache = _warm_cache()
    assert not cache.get("Who is the CEO of Northwind?").hit
    assert not cache.get("What is the capital of France?").hit


def test_empty_cache_misses():
    assert not SemanticCache().get("What is the Starter plan price?").hit


def test_put_updates_existing_entry():
    cache = SemanticCache()
    cache.put("What is the Starter plan price?", "$29")
    cache.put("What is the Starter plan price?", "$39")
    assert cache.get("What is the Starter plan price?").answer == "$39"


def test_ttl_expiry_with_fake_clock():
    clock = FakeClock()
    cache = SemanticCache(default_ttl=10, volatile_ttl=10, clock=clock)
    cache.put("How do I reset my password?", "Click 'Forgot password'.",
              volatile=False, ttl=10)
    assert cache.get("How do I reset my password?").hit
    clock.advance(11)
    assert not cache.get("How do I reset my password?").hit
    assert cache.summary()["expirations"] == 1


def test_volatile_ttl_shorter_than_default():
    cache = SemanticCache(default_ttl=3600, volatile_ttl=60)
    pricing = cache.put("What is the monthly price of the Starter plan?",
                        "$29")
    stable = cache.put("How do I reset my password?", "Click the link.")
    assert pricing.volatile and pricing.ttl == 60
    assert not stable.volatile and stable.ttl == 3600


def test_expired_entry_not_served_semantically():
    clock = FakeClock()
    cache = SemanticCache(default_ttl=3600, volatile_ttl=2, clock=clock)
    cache.put("What is the monthly price of the Growth plan?", "$79")
    clock.advance(3)
    assert not cache.get("What does Growth cost per month?").hit


def test_lru_eviction_at_capacity():
    cache = SemanticCache(max_size=3)
    cache.put("Question one?", "a1")
    cache.put("Question two?", "a2")
    cache.put("Question three?", "a3")
    cache.get("Question one?")  # touch: one is now most recently used
    cache.put("Question four?", "a4")
    assert cache.size() == 3
    assert cache.summary()["evictions"] == 1
    assert not cache.get("Question two?").hit  # least recently used is gone
    assert cache.get("Question one?").hit


def test_stats_accounting():
    cache = _warm_cache()
    cache.get("What is the monthly price of the Starter plan?")  # exact
    cache.get("How much does the Starter plan cost each month?")  # semantic
    cache.get("Who is the CEO of Northwind?")  # miss
    s = cache.summary()
    assert s["exact_hits"] == 1
    assert s["semantic_hits"] == 1
    assert s["misses"] == 1
    assert s["hit_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert s["llm_calls_avoided"] == 2


def test_save_load_round_trip(tmp_path):
    cache = _warm_cache()
    path = tmp_path / "cache.json"
    cache.save(path)
    loaded = SemanticCache.load(path)
    assert loaded.size() == 24
    result = loaded.get("How much does the Starter plan cost each month?")
    assert result.hit and result.answer == "The Starter plan costs $29 per month."


def test_threshold_sweep_at_default():
    """Pin the measured operating point: 0.60 -> 20/24 paraphrases, 1/12
    near-miss false hits, 0/12 unrelated false hits, 0 wrong hits."""
    with open(ROOT / "data" / "qa_pairs.json") as f:
        qa_pairs = json.load(f)["pairs"]
    with open(ROOT / "data" / "eval_queries.json") as f:
        ev = json.load(f)
    llm = MockLLM(ROOT / "data" / "qa_pairs.json",
                  ROOT / "data" / "eval_queries.json")
    cache = SemanticCache(threshold=0.60)
    for pair in qa_pairs:
        cache.put(pair["question"], pair["answer"])

    def base_of(question):
        return llm.base_id_by_key.get(" ".join(normalize_tokens(question)))

    para_hits = para_wrong = near_false = unrel_false = 0
    for item in ev["paraphrases"]:
        r = cache.get(item["question"])
        if r.hit and base_of(r.matched_question) == item["base_id"]:
            para_hits += 1
        elif r.hit:
            para_wrong += 1
    for item in ev["near_misses"]:
        if cache.get(item["question"]).hit:
            near_false += 1
    for item in ev["unrelated"]:
        if cache.get(item["question"]).hit:
            unrel_false += 1
    assert para_hits == 20
    assert para_wrong == 0
    assert near_false == 1  # annual-vs-monthly Starter price (sim 0.621)
    assert unrel_false == 0


def test_staleness_price_change_with_fake_clock():
    """Volatile entry serves the stale price inside its TTL, then the
    fresh price after expiry — the TTL is the staleness budget."""
    clock = FakeClock()
    llm = MockLLM(ROOT / "data" / "qa_pairs.json",
                  ROOT / "data" / "eval_queries.json", latency_s=0)
    cache = SemanticCache(default_ttl=3600, volatile_ttl=2, clock=clock)

    def ask(question):
        r = cache.get(question)
        if r.hit:
            return r.answer
        answer, _ = llm.ask(question)
        cache.put(question, answer)
        return answer

    assert ask("What is the monthly price of the Growth plan?") == \
        "The Growth plan costs $79 per month."
    assert ask("What does Growth cost per month?") == \
        "The Growth plan costs $79 per month."
    llm.update_answer("q02", "The Growth plan costs $99 per month.")
    clock.advance(1)
    assert ask("What does Growth cost per month?") == \
        "The Growth plan costs $79 per month."  # inside TTL: stale, by design
    clock.advance(2)
    assert ask("What does Growth cost per month?") == \
        "The Growth plan costs $99 per month."  # expired: fresh answer
    assert ask("What does Growth cost per month?") == \
        "The Growth plan costs $99 per month."  # now cached fresh
