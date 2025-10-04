
# server/security.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import os, math
import numpy as np

# Optional dependencies
try:
    from sklearn.svm import OneClassSVM
    from sklearn.cluster import KMeans
except Exception:  # pragma: no cover
    OneClassSVM = None
    KMeans = None

@dataclass
class SecurityState:
    scenario: str = os.getenv("SCENARIO", "A").upper()  # "A"|"B"|"C"
    phase: str = os.getenv("PHASE", "attack").lower()   # "attack"|"defend"
    window_id: str = os.getenv("WINDOW_ID", "")

def flatten_params(params_ndarrays: List[np.ndarray]) -> np.ndarray:
    if not params_ndarrays:
        return np.array([], dtype=np.float32)
    flat = np.concatenate([p.astype(np.float32).ravel() for p in params_ndarrays])
    return flat

def l2_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x) if x.size else 0.0)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 1.0
    da = np.linalg.norm(a); db = np.linalg.norm(b)
    if da == 0 or db == 0:
        return 1.0
    return float(np.dot(a, b) / (da*db))

class OCOutlier:
    """One-Class SVM wrapper with lazy-fit by reference window"""
    def __init__(self, nu=0.1, gamma="scale"):
        self.nu = nu; self.gamma = gamma
        self.model = None

    def fit(self, feats: np.ndarray):
        if OneClassSVM is None:
            self.model = None
            return
        self.model = OneClassSVM(nu=self.nu, gamma=self.gamma)
        self.model.fit(feats)

    def predict(self, feats: np.ndarray) -> np.ndarray:
        if self.model is None:
            # no sklearn, fall back: no outlier
            return np.ones((feats.shape[0],), dtype=int)
        return self.model.predict(feats)  # +1 normal, -1 outlier

class KMeansGuard:
    """K-means(2) to split benign cluster vs colluding cluster"""
    def __init__(self, k=2, random_state=0):
        self.k = k; self.random_state = random_state

    def split(self, feats: np.ndarray) -> Dict[str, Any]:
        if KMeans is None or feats.shape[0] < 2:
            # nothing to do
            return {"labels": np.zeros((feats.shape[0],), dtype=int), "centers": np.zeros((0, feats.shape[1]))}
        km = KMeans(n_clusters=self.k, n_init="auto", random_state=self.random_state)
        labels = km.fit_predict(feats)
        centers = km.cluster_centers_
        # Heuristic: smaller cluster or farther-from-origin -> suspect
        counts = [(lab, int(np.sum(labels==lab))) for lab in range(self.k)]
        counts.sort(key=lambda x: x[1])
        suspect_label = counts[0][0]  # smaller group as suspect by default
        return {"labels": labels, "centers": centers, "suspect_label": suspect_label}

def build_feature_vector(delta: np.ndarray, prev_global: np.ndarray, client_metrics: Dict[str, Any]) -> np.ndarray:
    # Simple robust features
    n = l2_norm(delta)
    c = cosine_sim(delta, prev_global)
    loss = float(client_metrics.get("loss", 0.0))
    acc = float(client_metrics.get("acc", 0.0))
    return np.array([n, c, loss, acc], dtype=np.float32)

def clip_and_check(delta: np.ndarray, norm_limit: float = 5.0, cos_min: float = -0.5) -> Dict[str, Any]:
    """Apply norm clipping and cosine sanity check. Return possibly clipped delta and flags."""
    n = l2_norm(delta)
    clipped = False
    if n > norm_limit and n > 0:
        delta = (delta / n) * norm_limit
        clipped = True
    cos_bad = False
    # If delta is too opposite to previous global direction, flag
    # (Using cos against zeros would be undefined, previous global provided externally)
    if cos_min is not None:
        # cos to an all-zeros 'direction' equals 1.0, so check only if prev_global has energy externally
        cos_bad = False  # evaluated at caller using real prev_global
    return {"delta": delta, "clipped": clipped, "cos_bad": cos_bad}
