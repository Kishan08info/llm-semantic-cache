"""Tests for the freshness policy, the entity guard, and the mock LLM."""

from pathlib import Path

from semantic_cache.entities import entities_compatible
from semantic_cache.mock_llm import ABSTAIN, MockLLM
from semantic_cache.policies import is_volatile, ttl_for
from semantic_cache.text import normalize_tokens

ROOT = Path(__file__).resolve().parent.parent


def test_is_volatile():
    assert is_volatile("What is the monthly price of the Starter plan?")
    assert is_volatile("What is the API rate limit on the Pro plan?")
    assert not is_volatile("How do I cancel my subscription?")
    assert not is_volatile("How do I reset my password?")


def test_ttl_for():
    assert ttl_for("What is the Starter plan price?", 3600, 60) == 60
    assert ttl_for("How do I cancel my subscription?", 3600, 60) == 3600


def test_guard_blocks_plan_mismatch():
    q = normalize_tokens("What is the monthly price of the Team plan?")
    c = normalize_tokens("What is the monthly price of the Starter plan?")
    assert not entities_compatible(q, c)


def test_guard_blocks_format_mismatch():
    q = normalize_tokens("Can I export my data as JSON?")
    c = normalize_tokens("How do I export my data?")
    assert not entities_compatible(q, c)


def test_guard_allows_same_plan():
    q = normalize_tokens("How much does the Starter plan cost each month?")
    c = normalize_tokens("What is the monthly price of the Starter plan?")
    assert entities_compatible(q, c)


def test_guard_allows_one_sided_action():
    # "if I cancel" is context in a refund question, not the discriminator.
    q = normalize_tokens("Can I get my money back if I cancel?")
    c = normalize_tokens("What is the refund policy?")
    assert entities_compatible(q, c)


def test_guard_blocks_action_conflict():
    q = normalize_tokens("How do I downgrade my plan?")
    c = normalize_tokens("How do I upgrade my plan?")
    assert not entities_compatible(q, c)


def test_mock_llm_answers_and_counts_calls():
    llm = MockLLM(ROOT / "data" / "qa_pairs.json",
                  ROOT / "data" / "eval_queries.json", latency_s=0)
    answer, base_id = llm.ask("What is the monthly price of the Starter plan?")
    assert answer == "The Starter plan costs $29 per month."
    assert base_id == "q01"
    assert llm.calls == 1
    answer, base_id = llm.ask("How much does the Starter plan cost each month?")
    assert answer == "The Starter plan costs $29 per month."
    assert base_id == "q01"


def test_mock_llm_abstains_on_unrelated():
    llm = MockLLM(ROOT / "data" / "qa_pairs.json",
                  ROOT / "data" / "eval_queries.json", latency_s=0)
    answer, base_id = llm.ask("Who is the CEO of Northwind?")
    assert answer == ABSTAIN
    assert base_id is None


def test_mock_llm_update_changes_paraphrase_answers_too():
    llm = MockLLM(ROOT / "data" / "qa_pairs.json",
                  ROOT / "data" / "eval_queries.json", latency_s=0)
    llm.update_answer("q02", "The Growth plan costs $99 per month.")
    answer, _ = llm.ask("What is the monthly price of the Growth plan?")
    assert answer == "The Growth plan costs $99 per month."
    answer, _ = llm.ask("What does Growth cost per month?")
    assert answer == "The Growth plan costs $99 per month."
