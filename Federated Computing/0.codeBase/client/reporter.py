# client/reporter.py
import time, json, requests
from typing import Dict, Any
from .net_sim import NetworkSimulator

class Reporter:
    def __init__(self, client_id: str, server_url="http://127.0.0.1:5000", net: NetworkSimulator=None):
        self.client_id = client_id
        self.server_url = server_url
        self.net = net or NetworkSimulator()

    def _post(self, path: str, payload: Dict[str, Any]):
        data = json.dumps(payload).encode("utf-8")
        t0 = time.time()
        ok = self.net.transmit(len(data))
        if not ok:
            # simulate drop
            return {"ok": False, "dropped": True, "rtt_ms": (time.time()-t0)*1000.0}
        try:
            r = requests.post(f"{self.server_url}{path}", json=payload, timeout=5)
            r.raise_for_status()
            rtt = (time.time()-t0)*1000.0
            return {"ok": True, "resp": r.json(), "rtt_ms": rtt}
        except Exception:
            return {"ok": False, "rtt_ms": (time.time()-t0)*1000.0}

    def register(self, hw: Dict[str, str], net_profile: Dict[str, float], n_i: int):
        payload = {
            "client_id": self.client_id,
            "cpu": hw.get("cpu",""),
            "mem": hw.get("mem",""),
            "net_profile": net_profile,
            "n_i": n_i
        }
        res = self._post("/api/v1/client/register", payload)
        nt = 0
        if res.get("ok") and isinstance(res.get("resp"), dict):
            nt = res["resp"].get("N_T", 0)
        return nt

    def send_metrics(self, round_id: int, R: float, H: float, train: Dict[str, Any], sys: Dict[str, Any], bytes_up=0, bytes_down=0):
        payload = {
            "client_id": self.client_id,
            "round": round_id,
            "R": R, "H": H,
            "train": train,
            "comm": {"rtt_ms": None, "bytes_up": bytes_up, "bytes_down": bytes_down},
            "sys": sys
        }
        res = self._post("/api/v1/metrics", payload)
        if res.get("ok"):
            payload["comm"]["rtt_ms"] = res.get("rtt_ms")
        return res

    def send_local_analysis(self, round_id: int, analysis: Dict[str,Any], R: float):
        payload = { "client_id": self.client_id, "round": round_id, "analysis": analysis, "R": R }
        return self._post("/api/v1/analysis/local", payload)

    def send_model_update_meta(self, round_id: int, model_digest: str, strategy="FedAvg"):
        payload = { "client_id": self.client_id, "round": round_id, "model_digest": model_digest, "strategy": strategy }
        return self._post("/api/v1/model/update", payload)
