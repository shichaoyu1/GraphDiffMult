"""Retrieval over explicitly labelled, hashable candidate keys only."""


def masked_retrieval_metrics(ranked_keys, positive_keys, negative_keys, k):
    """Return hit@k, recall@k and full-ranking reciprocal rank.

    Unknown candidates are removed before ranks are counted. Unranked positives
    remain in the recall denominator. Invalid inputs raise ValueError even when
    there are no positives. Boolean k is not accepted as an integer cutoff.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError('k must be a positive integer')
    positives, negatives = set(positive_keys), set(negative_keys)
    if positives & negatives:
        raise ValueError('positive and negative keys overlap')
    ranking = list(ranked_keys)
    if len(ranking) != len(set(ranking)):
        raise ValueError('ranked keys must be unique')
    if not positives:
        return dict(hit_at_k=None, recall_at_k=None, reciprocal_rank=None)
    known = positives | negatives
    ranking = [key for key in ranking if key in known]
    hits = len(positives.intersection(ranking[:k]))
    first = next((i for i, key in enumerate(ranking, 1) if key in positives), None)
    return dict(hit_at_k=float(hits > 0), recall_at_k=hits / len(positives),
                reciprocal_rank=1 / first if first is not None else 0.0)
