"""Dynamic time warping alignment, from scratch (Sakoe-Chiba banded DP).

Frame distance is cosine distance on cepstral vectors, which is robust to
overall level differences between recordings.
"""
import numpy as np

INF = float("inf")


def dtw_cost(a, b, band=None, weights=(1.0, 1.0, 2.0)):
    """Average per-frame cost of aligning feature sequences a and b.

    Both sequences are (n_frames, n_dim). DP uses a Sakoe-Chiba band and
    classic 1/1/2 step weights; returns the accumulated cost normalized by
    path length, in [0, 2].
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return INF
    band = min(band if band is not None else max(30, abs(n - m) + 12),
               max(n, m))
    an = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
    bn = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-9)
    d = 1.0 - an @ bn.T
    acc = np.full((n + 1, m + 1), INF)
    acc[0, 0] = 0.0
    w1, w2, w3 = weights
    for i in range(1, n + 1):
        lo = max(1, i - band)
        hi = min(m, i + band)
        row = acc[i]
        prev = acc[i - 1]
        di = d[i - 1]
        for j in range(lo, hi + 1):
            best = INF
            diag = prev[j - 1]
            up = prev[j]
            left = row[j - 1]
            x = di[j - 1]
            if diag < INF:
                best = diag + w3 * x
            if up < INF:
                best = min(best, up + w1 * x)
            if left < INF:
                best = min(best, left + w1 * x)
            row[j] = best
    if acc[n, m] == INF:
        return INF
    denom = max((n * w1 + m * w2) / 2.0, 1.0)
    return acc[n, m] / denom
