"""TF-IDF vectors and cosine similarity over normalised question tokens.

Pure stdlib. IDF is fitted on the cached questions (the knowledge-base
questions already in the cache), so discriminating tokens — the plan name
in "Starter plan price" vs "Growth plan price" — weigh more than tokens
that appear in almost every question ("plan", "price"). That is what lets
the cache tell a paraphrase ("How much does Starter cost each month?")
from a near-miss ("What is the monthly price of the Growth plan?").
"""

import math
from collections import Counter

from .text import normalize_tokens


class TfidfVectorizer:
    """Fit IDF on a corpus of token lists; vectorise new token lists."""

    def __init__(self, docs):
        self.n_docs = len(docs)
        df = Counter()
        for doc in docs:
            for tok in set(doc):
                df[tok] += 1
        self.idf = {
            tok: math.log((self.n_docs + 1) / (count + 1)) + 1.0
            for tok, count in df.items()
        }
        # Tokens never seen in the corpus get the max IDF (rare = informative).
        self.unseen_idf = math.log(self.n_docs + 1) + 1.0

    def vector(self, tokens):
        """L2-normalised sparse TF-IDF vector as {token: weight}."""
        counts = Counter(tokens)
        vec = {}
        norm_sq = 0.0
        for tok, count in counts.items():
            w = count * self.idf.get(tok, self.unseen_idf)
            vec[tok] = w
            norm_sq += w * w
        norm = math.sqrt(norm_sq)
        if norm == 0:
            return {}
        return {tok: w / norm for tok, w in vec.items()}


def cosine(a, b):
    """Cosine similarity of two L2-normalised sparse vectors."""
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    return sum(w * b.get(tok, 0.0) for tok, w in a.items())


def similarity(question_a, question_b, vectorizer):
    """Cosine similarity between two raw questions under a vectorizer."""
    va = vectorizer.vector(normalize_tokens(question_a))
    vb = vectorizer.vector(normalize_tokens(question_b))
    return cosine(va, vb)
