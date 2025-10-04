# client/client3.py
from .fl_client import run_client

if __name__ == "__main__":
    # 반대 불균형 (class 0 우세)
    run_client(
        client_id="client-3",
        N=800, d=20,
        class_probs=(0.75, 0.25),
        net_profile={"latency_ms": 120, "loss_pct": 2.0, "bandwidth_mbps": 30.0, "jitter_ms": 20},
        seed=3
    )
