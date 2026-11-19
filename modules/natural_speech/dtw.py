"""Dynamic time warping alignment, from scratch (Sakoe-Chiba banded DP).

Frame distance is cosine distance on cepstral vectors, which is robust to
overall level differences between recordings.

The DP is exact but vectorized: each row is computed with numpy operations
over the band, and the sequential "left step" dependency is resolved exactly
with binary min-plus jumps over prefix sums of the row's local costs.
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
    w1, w2, w3 = weights
    acc = np.full((n + 1, m + 1), INF)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        lo = max(1, i - band)
        hi = min(m, i + band)
        if lo > hi:
            continue
        cols = np.arange(lo, hi + 1)
        x = d[i - 1, lo - 1:hi]
        prev = acc[i - 1]
        row = acc[i]
        # diag step (w3) and up step (w1)
        t = np.minimum(prev[cols - 1] + w3 * x, prev[cols] + w1 * x)
        # exact left-step chains: binary doubling over prefix sums of x
        cs = np.cumsum(x)
        delta = 1
        while delta < len(cols):
            span = w1 * (cs[delta:] - cs[:len(cols) - delta])
            t[delta:] = np.minimum(t[delta:], t[:len(cols) - delta] + span)
            delta *= 2
        row[lo:hi + 1] = t
    if acc[n, m] == INF:
        return INF
    denom = max((n * w1 + m * w2) / 2.0, 1.0)
    return acc[n, m] / denom
