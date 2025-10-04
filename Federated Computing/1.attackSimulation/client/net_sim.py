# client/net_sim.py
import random, time

class NetworkSimulator:
    def __init__(self, latency_ms=50, loss_pct=1.0, bandwidth_mbps=100.0, jitter_ms=10):
        self.latency_ms = latency_ms
        self.loss_pct = loss_pct
        self.bandwidth_mbps = bandwidth_mbps
        self.jitter_ms = jitter_ms

    def transmit(self, payload_bytes: int) -> bool:
        # sleep by latency + size/bandwidth + jitter
        base = self.latency_ms/1000.0
        jitter = random.uniform(0, self.jitter_ms)/1000.0
        # bytes -> bits / (Mbps*1e6) = seconds
        bw_delay = (payload_bytes*8) / (self.bandwidth_mbps*1e6)
        time.sleep(base + bw_delay + jitter)
        # drop?
        if random.random()*100 < self.loss_pct:
            return False
        return True
