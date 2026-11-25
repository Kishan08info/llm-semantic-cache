"""Semantic cache: exact match, then similarity match, with TTL + LRU.

Lookup order for a question:
  1. Exact normalised-text match -> EXACT hit (similarity 1.0).
  2. Best cosine similarity over cached entries >= threshold -> SEMANTIC hit.
  3. Otherwise a MISS — the caller asks the LLM and then puts the answer.

Eviction: entries expire after their TTL (lazy expiry on access, plus a
sweep on put); when the cache is full the least-recently-used entry is
evicted. A clock function is injectable so TTL behaviour is testable
without sleeping.
"""

import json
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field

from .embeddings import TfidfVectorizer, cosine
from .entities import entities_compatible
from .policies import is_volatile, ttl_for
from .text import normalize_tokens


@dataclass
class Entry:
    question: str
    answer: str
    tokens: list
    created_at: float
    last_accessed: float
    ttl: float
    volatile: bool
    hits: int = 0

    def expired(self, now):
        return now - self.created_at >= self.ttl

    def to_dict(self):
        d = asdict(self)
        d["vector"] = None  # vectors are rebuilt from tokens on load
        return d


@dataclass
class Lookup:
    hit: bool
    match_type: str  # "exact" | "semantic" | "miss"
    answer: str | None
    similarity: float
    matched_question: str | None = None


class SemanticCache:
    def __init__(self, threshold=0.60, max_size=100, default_ttl=3600,
                 volatile_ttl=60, clock=None):
        self.threshold = threshold
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.volatile_ttl = volatile_ttl
        self.clock = clock or time.time
        self.entries = OrderedDict()  # normalised key -> Entry (LRU order)
        self.vectorizer = None
        self.vectors = {}  # normalised key -> sparse TF-IDF vector
        self.stats = {
            "exact_hits": 0,
            "semantic_hits": 0,
            "misses": 0,
            "puts": 0,
            "evictions": 0,
            "expirations": 0,
            "llm_calls_avoided": 0,
        }

    # -- internal ---------------------------------------------------------
    def _now(self):
        return self.clock()

    def _key(self, question):
        return " ".join(normalize_tokens(question))

    def _refit(self):
        """Refit IDF on the current entries and rebuild their vectors."""
        docs = [e.tokens for e in self.entries.values()]
        self.vectorizer = TfidfVectorizer(docs) if docs else None
        self.vectors = {
            key: self.vectorizer.vector(e.tokens)
            for key, e in self.entries.items()
        } if self.vectorizer else {}

    def _sweep_expired(self):
        now = self._now()
        expired = [k for k, e in self.entries.items() if e.expired(now)]
        for key in expired:
            del self.entries[key]
            self.vectors.pop(key, None)
            self.stats["expirations"] += 1
        if expired:
            self._refit()

    # -- public API -------------------------------------------------------
    def put(self, question, answer, volatile=None, ttl=None):
        self._sweep_expired()
        key = self._key(question)
        now = self._now()
        if volatile is None:
            volatile = is_volatile(question)
        entry_ttl = ttl if ttl is not None else ttl_for(
            question, self.default_ttl, self.volatile_ttl)
        tokens = normalize_tokens(question)
        if key in self.entries:
            entry = self.entries[key]
            entry.answer = answer
            entry.tokens = tokens
            entry.created_at = now
            entry.last_accessed = now
            entry.ttl = entry_ttl
            entry.volatile = volatile
            self.entries.move_to_end(key)
        else:
            self.entries[key] = Entry(
                question=question, answer=answer, tokens=tokens,
                created_at=now, last_accessed=now, ttl=entry_ttl,
                volatile=volatile)
            self.stats["puts"] += 1
            while len(self.entries) > self.max_size:
                old_key, _ = self.entries.popitem(last=False)
                self.vectors.pop(old_key, None)
                self.stats["evictions"] += 1
        self._refit()
        return self.entries[key]

    def get(self, question):
        now = self._now()
        key = self._key(question)
        entry = self.entries.get(key)
        if entry is not None:
            if entry.expired(now):
                del self.entries[key]
                self.vectors.pop(key, None)
                self.stats["expirations"] += 1
                self._refit()
            else:
                entry.hits += 1
                entry.last_accessed = now
                self.entries.move_to_end(key)
                self.stats["exact_hits"] += 1
                self.stats["llm_calls_avoided"] += 1
                return Lookup(True, "exact", entry.answer, 1.0, entry.question)
        if self.vectorizer is not None and self.vectors:
            query_tokens = normalize_tokens(question)
            query_vec = self.vectorizer.vector(query_tokens)
            best_key, best_sim = None, 0.0
            for cand_key, cand_vec in self.vectors.items():
                # Entity guard: the discriminating entity (plan, format)
                # must agree before a similarity score can serve an answer.
                if not entities_compatible(
                        query_tokens, self.entries[cand_key].tokens):
                    continue
                sim = cosine(query_vec, cand_vec)
                if sim > best_sim:
                    best_key, best_sim = cand_key, sim
            if best_key is not None and best_sim >= self.threshold:
                entry = self.entries[best_key]
                if entry.expired(now):
                    del self.entries[best_key]
                    self.vectors.pop(best_key, None)
                    self.stats["expirations"] += 1
                    self._refit()
                else:
                    entry.hits += 1
                    entry.last_accessed = now
                    self.entries.move_to_end(best_key)
                    self.stats["semantic_hits"] += 1
                    self.stats["llm_calls_avoided"] += 1
                    return Lookup(True, "semantic", entry.answer, best_sim,
                                  entry.question)
        self.stats["misses"] += 1
        return Lookup(False, "miss", None, 0.0)

    def size(self):
        return len(self.entries)

    def summary(self):
        hits = self.stats["exact_hits"] + self.stats["semantic_hits"]
        total = hits + self.stats["misses"]
        return {
            **self.stats,
            "size": self.size(),
            "threshold": self.threshold,
            "max_size": self.max_size,
            "default_ttl_s": self.default_ttl,
            "volatile_ttl_s": self.volatile_ttl,
            "hit_rate": round(hits / total, 4) if total else 0.0,
        }

    # -- persistence ------------------------------------------------------
    def save(self, path):
        with open(path, "w") as f:
            json.dump({
                "threshold": self.threshold,
                "max_size": self.max_size,
                "default_ttl": self.default_ttl,
                "volatile_ttl": self.volatile_ttl,
                "entries": [e.to_dict() for e in self.entries.values()],
            }, f, indent=2)

    @classmethod
    def load(cls, path, clock=None):
        with open(path) as f:
            data = json.load(f)
        cache = cls(threshold=data["threshold"], max_size=data["max_size"],
                    default_ttl=data["default_ttl"],
                    volatile_ttl=data["volatile_ttl"], clock=clock)
        for raw in data["entries"]:
            raw.pop("vector", None)
            cache.entries[cache._key(raw["question"])] = Entry(**raw)
        cache._refit()
        return cache
