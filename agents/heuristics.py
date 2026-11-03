"""
heuristics.py - baseline handover policies (no learning).

Each policy maps an environment state + metric snapshot to an action in the
HandoverEnv action space (0 = keep, 1..K = candidate by descending elevation).
These are the classical, deployment-simple policies used as comparison baselines.
"""
import numpy as np


def _cand_indices(m, K):
    vis = np.where(m["visible"])[0]
    if len(vis) == 0:
        return np.array([], dtype=int)
    return vis[np.argsort(-m["els"][vis])][:K]


class KeepOrNearest:
    """Keep the current satellite while visible; otherwise switch to the highest
    satellite (closest in ground-range terms). Classic 'stay put' behaviour."""

    name = "keep-or-nearest"

    def __init__(self, K=4):
        self.K = K

    def act(self, state, m, serving):
        cand = _cand_indices(m, self.K)
        if serving is not None and serving in np.where(m["visible"])[0]:
            return 0
        if len(cand) == 0:
            return 0
        # map the chosen satellite (first candidate) back to an action index
        return 1  # candidate slot 0 is index 1 in the action space


class NearestSatellite:
    """Always hand over to the visible satellite with the highest elevation
    (geometrically nearest in this spherical model)."""

    name = "nearest"

    def __init__(self, K=4):
        self.K = K

    def act(self, state, m, serving):
        cand = _cand_indices(m, self.K)
        if len(cand) == 0:
            return 0
        # action 1 always targets the highest-elevation candidate
        if serving is not None and serving == cand[0]:
            return 0
        return 1


class MaxSINR:
    """Always hand over to the visible satellite with the best instantaneous
    SINR (classic strongest-signal policy)."""

    name = "max-sinr"

    def __init__(self, K=4):
        self.K = K

    def act(self, state, m, serving):
        vis = np.where(m["visible"])[0]
        if len(vis) == 0:
            return 0
        best = int(vis[np.argmax(m["sinr"][vis])])
        if serving == best:
            return 0
        cand = _cand_indices(m, self.K)
        pos = np.where(cand == best)[0]
        if len(pos) > 0:
            return int(pos[0]) + 1
        return 1  # fallback: highest-elevation candidate


class LongestLOS:
    """Always hand over to the visible satellite with the longest remaining
    visibility time (minimizes near-term handover frequency)."""

    name = "longest-los"

    def __init__(self, K=4):
        self.K = K

    def act(self, state, m, serving):
        vis = np.where(m["visible"])[0]
        if len(vis) == 0:
            return 0
        best = int(vis[np.argmax(m["los"][vis])])
        if serving == best:
            return 0
        cand = _cand_indices(m, self.K)
        pos = np.where(cand == best)[0]
        if len(pos) > 0:
            return int(pos[0]) + 1
        return 1
