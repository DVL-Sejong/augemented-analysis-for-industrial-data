# client/client1.py
from .fl_client import run_client

if __name__ == "__main__":
    # 불균형 데이터 (class 1 우세)
    run_client(
        client_id="client-1",
        N=1200, d=20,
        class_probs=(0.3, 0.7),
        net_profile={"latency_ms": 40, "loss_pct": 0.5, "bandwidth_mbps": 80.0, "jitter_ms": 8},
        seed=1
    )
