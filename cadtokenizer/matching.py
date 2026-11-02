"""Deterministic assignment of distinct native token IDs from ranked candidates."""
import heapq


def unique_assignment(candidate_ids, candidate_scores):
    heap, used, assignment = [], set(), [None] * len(candidate_ids)
    for code, (ids, scores) in enumerate(zip(candidate_ids, candidate_scores)):
        if not ids or len(ids) != len(scores):
            raise ValueError('Invalid candidate list')
        heapq.heappush(heap, (-float(scores[0]), code, 0))
    while heap:
        score, code, rank = heapq.heappop(heap)
        token = candidate_ids[code][rank]
        if token not in used:
            assignment[code] = token
            used.add(token)
        else:
            rank += 1
            if rank >= len(candidate_ids[code]):
                raise ValueError('Candidate lists cannot supply distinct native IDs; increase candidate_count')
            heapq.heappush(heap, (-float(candidate_scores[code][rank]), code, rank))
    return assignment
