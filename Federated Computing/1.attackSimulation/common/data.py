# common/data.py
import numpy as np
from typing import Tuple, Dict, List

def make_synthetic_binary(N: int, d: int, class_probs=(0.5,0.5), seed=0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    # features
    X = rng.normal(0,1,(N,d))
    # true weights
    w_true = rng.normal(0,1,(d,))
    # base logits
    logits = X @ w_true
    # force distribution by overriding labels per probs
    labels = rng.choice([0,1], size=N, p=class_probs)
    # separate margins per class to create learnable pattern
    X[labels==1] += 0.8
    X[labels==0] -= 0.8
    return X.astype(np.float32), labels.astype(np.int64)

def train_logreg_numpy(X, y, w, b, lr=0.1, epochs=3, batch=64, rng=None):
    rng = rng or np.random.default_rng()
    N, d = X.shape
    for _ in range(epochs):
        idx = rng.permutation(N)
        for i in range(0, N, batch):
            sel = idx[i:i+batch]
            xb, yb = X[sel], y[sel]
            z = xb @ w + b
            p = 1/(1+np.exp(-z))
            # loss grad
            grad_w = xb.T @ (p - yb) / len(xb)
            grad_b = np.mean(p - yb)
            w -= lr * grad_w
            b -= lr * grad_b
    # metrics
    z = X @ w + b
    p = 1/(1+np.exp(-z))
    loss = -np.mean(y*np.log(p+1e-8) + (1-y)*np.log(1-p+1e-8))
    acc = float(np.mean((p>=0.5) == y))
    return w, b, {"loss": float(loss), "acc": acc}
