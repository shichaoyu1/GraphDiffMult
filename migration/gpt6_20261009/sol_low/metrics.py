"""Retrieval metrics over explicitly known positive and negative candidates."""


def masked_retrieval_metrics(ranked_keys, positive_keys, negative_keys, k):
    """Return per-query hit@k, recall@k and full-ranking reciprocal rank.

    Keys must be hashable. Unknown candidates are removed before assigning
    ranks. Missing positive keys still count in the recall denominator.
    Invalid inputs are rejected even if no positive label is available.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError('k must be a positive integer')
    positives, negatives = set(positive_keys), set(negative_keys)
    if positives & negatives:
        raise ValueError('positive and negative keys overlap')
    ranking = list(ranked_keys)
    if len(ranking) != len(set(ranking)):
        raise ValueError('ranked keys contain duplicates')
    if not positives:
        return dict(hit_at_k=None, recall_at_k=None, reciprocal_rank=None)
    known = positives | negatives
    ranking = [key for key in ranking if key in known]
    hits = len(positives.intersection(ranking[:k]))
    first = next((rank for rank, key in enumerate(ranking, 1)
                  if key in positives), None)
    return dict(hit_at_k=float(hits > 0), recall_at_k=hits / len(positives),
                reciprocal_rank=1 / first if first is not None else 0.0)
