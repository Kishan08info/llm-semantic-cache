"""Lexical normalisation for short support questions.

Lowercases, strips punctuation, drops stopwords, and maps common synonyms
to one canonical token (price/cost/charge -> price, ...). This is the
lexical stand-in for an embedding model: a real deployment would embed the
question with a model (e.g. text-embedding-3-small) and compare vectors.
The normalisation is deliberately small and documented so its limits are
visible — it cannot catch meaning that shares no words.
"""

import re

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "do", "does", "did",
    "can", "could", "will", "would", "should", "i", "me", "my", "we",
    "you", "your", "it", "its", "of", "to", "in", "on", "for", "with",
    "at", "by", "from", "and", "or", "how", "what", "when", "where",
    "which", "who", "am", "be", "been", "there", "here", "this", "that",
    "if", "so", "as", "an", "per",
}

# Canonical form -> variants. Applied after stopword removal.
SYNONYMS = {
    "cost": "price",
    "charge": "price",
    "fee": "price",
    "pricing": "price",
    "monthly": "month",
    "month": "month",
    "yearly": "annual",
    "year": "annual",
    "cancel": "cancel",
    "terminate": "cancel",
    "close": "cancel",
    "higher": "upgrade",
    "forgot": "reset",
    "single": "sso",
    "sign": "sso",
    "limit": "limit",
    "allowance": "limit",
    "quota": "limit",
    "fast": "response",
    "quickly": "response",
    "reply": "response",
    "respond": "response",
    "responds": "response",
    "get": "get",
    "receive": "get",
    "download": "export",
    "save": "discount",
    "certified": "compliant",
    "certification": "compliant",
    "delete": "kept",
    "kept": "kept",
    "retained": "kept",
    "store": "hosted",
    "stored": "hosted",
    "locked": "minimum",
    "contract": "minimum",
    "seats": "seat",
    "guarantee": "sla",
    "paying": "billing",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    """Lowercase, split on non-alphanumerics, drop stopwords."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def normalize_tokens(text):
    """Tokenize then map synonyms to their canonical form."""
    return [SYNONYMS.get(t, t) for t in tokenize(text)]


def content_tokens(text):
    """Deduplicated, order-stable content tokens for a question."""
    seen = set()
    out = []
    for t in normalize_tokens(text):
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out
