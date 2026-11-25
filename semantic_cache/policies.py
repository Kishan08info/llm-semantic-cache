"""Freshness policy: how long a cached answer is allowed to live.

Not every answer goes stale at the same speed. A pricing answer can be
wrong the day a price changes; a "how do I reset my password" answer is
stable for months. The policy classifies a question as volatile when its
normalised tokens mention price, cost, limits, or seat counts, and gives
volatile entries a much shorter TTL. This is the cache-side answer to the
stale-document lesson from the hallucination-guardrails project: a cache
that never expires will eventually serve a wrong price with total
confidence.
"""

from .text import normalize_tokens

# Normalised tokens that mark an answer as fast-changing.
VOLATILE_TOKENS = {
    "price", "pricing", "cost", "limit", "seats", "seat", "trial",
    "discount", "response", "sla",
}


def is_volatile(question):
    """True when the question is about a fast-changing fact."""
    return bool(set(normalize_tokens(question)) & VOLATILE_TOKENS)


def ttl_for(question, default_ttl, volatile_ttl):
    """TTL in seconds for a question under the policy."""
    return volatile_ttl if is_volatile(question) else default_ttl
