
# client/attack_plugin.py
import os, numpy as np, hashlib, random

class AttackPlugin:
    """Simulates assistant-level attacks by tampering client updates/metrics during ATTACK phase."""
    def __init__(self):
        self.scenario = os.getenv("SCENARIO", "A").upper()
        self.phase = os.getenv("PHASE", "attack").lower()

    # === Parameter tampering ===
    def tamper_params(self, w: np.ndarray, b: float) -> tuple[np.ndarray, float]:
        if self.phase != "attack":
            return w, b
        if self.scenario == "A":
            # Sign flip on a random slice
            if w.size > 0:
                idx = np.random.choice(w.size, size=max(1, w.size//5), replace=False)
                w = w.copy(); w[idx] *= -3.0
        elif self.scenario == "B":
            # Colluding drift: push in a fixed random direction
            rnd = np.random.RandomState(123)
            direction = rnd.normal(0,1,w.shape).astype(w.dtype)
            direction /= (np.linalg.norm(direction)+1e-8)
            w = w.copy(); w += 2.5*direction
        elif self.scenario == "C":
            # Phishing: try to inject huge bias
            b = b + 5.0
        return w, b

    # === Metrics tampering ===
    def tamper_metrics(self, metrics: dict) -> dict:
        if self.phase != "attack":
            return metrics
        m = dict(metrics)
        if self.scenario == "A":
            m["acc"] = max(0.0, min(1.0, float(m.get("acc",0.0)) + 0.2))
        elif self.scenario == "B":
            m["loss"] = float(m.get("loss",0.0)) * 0.5
        elif self.scenario == "C":
            m["acc"] = 1.0; m["loss"] = 0.0
        return m
