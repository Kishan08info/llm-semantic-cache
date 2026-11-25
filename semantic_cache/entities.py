"""Entity-consistency guard for semantic cache hits.

Similarity alone cannot tell a paraphrase from a near-miss: "What is the
monthly price of the Growth plan?" is 0.82-similar to the cached Starter
price question, but serving the Starter answer would be wrong. The
discriminating token is the plan name.

The guard extracts small entity groups from the normalised tokens and
requires the query and the cached question to agree on them:
  - plan (starter/growth/pro/enterprise) and format (csv/json) are
    ANSWER-DETERMINING in this knowledge base — price, limits, seats and
    support all vary by plan — so the two sides must name the same set,
    including one side naming a plan the other does not.
  - action (cancel/upgrade/downgrade) and region (us/eu) are treated as
    conflicting-only: they block a hit only when both sides name
    different ones ("if I cancel" is context in a refund question, not
    the discriminating entity).

This is the toy version of what production semantic caches do with
structured metadata / extracted slots. It is a heuristic, documented as
one — it does not replace an embedding model.
"""

PLAN = {"starter", "growth", "pro", "enterprise"}
ACTION = {"cancel", "upgrade", "downgrade"}
FORMAT = {"csv", "json"}
REGION = {"us", "eu"}

# Groups where the sets must be identical.
STRICT_GROUPS = (PLAN, FORMAT)
# Groups where only a direct conflict blocks.
CONFLICT_GROUPS = (ACTION, REGION)


def extract_entities(tokens):
    """{group_index: set of entities} for the groups a token list touches."""
    found = {}
    for i, group in enumerate(STRICT_GROUPS + CONFLICT_GROUPS):
        hit = set(tokens) & group
        if hit:
            found[i] = hit
    return found


def entities_compatible(query_tokens, candidate_tokens):
    """True when the query and the cached question agree on entities."""
    q = extract_entities(query_tokens)
    c = extract_entities(candidate_tokens)
    for i, group in enumerate(STRICT_GROUPS):
        if q.get(i, set()) != c.get(i, set()):
            return False
    for j, group in enumerate(CONFLICT_GROUPS, start=len(STRICT_GROUPS)):
        if q.get(j) and c.get(j) and q[j] != c[j]:
            return False
    return True
