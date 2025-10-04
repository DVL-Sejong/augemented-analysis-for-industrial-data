# client/client2.py
from .fl_client import run_client

if __name__ == "__main__":
    # 균등 데이터
    run_client(
        client_id="client-2",
        N=1000, d=20,
        class_probs=(0.5, 0.5),
        net_profile={"latency_ms": 80, "loss_pct": 1.0, "bandwidth_mbps": 50.0, "jitter_ms": 15},
        seed=2
    )
