"""Tests for TF-IDF vectors, cosine similarity, and text normalisation."""

import pytest

from semantic_cache.embeddings import TfidfVectorizer, cosine, similarity
from semantic_cache.text import content_tokens, normalize_tokens


def test_cosine_identical_is_one():
    vec = TfidfVectorizer([["starter", "plan", "price"]])
    a = vec.vector(["starter", "plan", "price"])
    assert cosine(a, a) == pytest.approx(1.0)


def test_cosine_disjoint_is_zero():
    vec = TfidfVectorizer([["starter", "plan"], ["cancel", "subscription"]])
    a = vec.vector(["starter", "plan"])
    b = vec.vector(["cancel", "subscription"])
    assert cosine(a, b) == 0.0


def test_cosine_empty_is_zero():
    vec = TfidfVectorizer([["starter"]])
    assert cosine({}, vec.vector(["starter"])) == 0.0
    assert cosine(vec.vector(["starter"]), {}) == 0.0


def test_similarity_is_symmetric():
    docs = [["starter", "plan", "price", "month"],
            ["growth", "plan", "price", "month"]]
    vec = TfidfVectorizer(docs)
    a = "What is the monthly price of the Starter plan?"
    b = "How much does the Starter plan cost each month?"
    assert similarity(a, b, vec) == similarity(b, a, vec)


def test_rare_token_weighs_more_than_common_token():
    # "starter" appears in 1 doc, "plan" in both: the IDF of the rare
    # token must be higher, which is what separates the plan-price
    # questions from each other.
    docs = [["starter", "plan", "price"], ["growth", "plan", "price"]]
    vec = TfidfVectorizer(docs)
    assert vec.idf["starter"] > vec.idf["plan"]
    assert vec.idf["starter"] == vec.idf["growth"]


def test_paraphrase_beats_unrelated():
    docs = [["starter", "plan", "price", "month"],
            ["growth", "plan", "price", "month"],
            ["cancel", "subscription"]]
    vec = TfidfVectorizer(docs)
    base = "What is the monthly price of the Starter plan?"
    para = "How much does the Starter plan cost each month?"
    unrel = "Who is the CEO of Northwind?"
    assert similarity(base, para, vec) > 0.6
    assert similarity(base, unrel, vec) < similarity(base, para, vec)


def test_normalize_maps_synonyms_and_drops_stopwords():
    assert normalize_tokens("How much does the Starter plan cost?") == \
        ["much", "starter", "plan", "price"]
    assert normalize_tokens("What is the monthly price?") == \
        ["month", "price"]


def test_content_tokens_deduplicates():
    assert content_tokens("price price price plan") == ["price", "plan"]
