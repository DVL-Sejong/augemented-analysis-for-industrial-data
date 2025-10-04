
# client/reporter.py
import time, json, requests, os, hmac, hashlib
from typing import Dict, Any
from .net_sim import NetworkSimulator

class Reporter:
    def __init__(self, client_id: str, server_url="http://127.0.0.1:5000", net: NetworkSimulator=None):
        self.client_id = client_id
        self.server_url = server_url
        self.net = net or NetworkSimulator()
        self.secret = os.getenv("FEDCOMP_SECRET", "")

    def _sign(self, data: bytes) -> str:
        if not self.secret:
            return ""
        return hmac.new(self.secret.encode("utf-8"), data, hashlib.sha256).hexdigest()

    def _post(self, path: str, payload: Dict[str, Any]):
        data = json.dumps(payload).encode("utf-8")
        t0 = time.time()
        ok = self.net.transmit(len(data))
        if not ok:
            return {"ok": False, "error": "network_drop"}
        try:
            headers = {"X-Client-Id": self.client_id}
            sig = self._sign(data)
            if sig:
                headers["X-Signature"] = sig
            res = requests.post(self.server_url + path, data=data, headers=headers, timeout=5)
            dt = (time.time() - t0) * 1000.0
            if res.status_code != 200:
                return {"ok": False, "status": res.status_code, "resp": res.text}
            return {"ok": True, "status": 200, "resp": res.json(), "rtt_ms": dt}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def send_register(self, info: Dict[str, Any]):
        payload = dict(info)
        payload["client_id"] = self.client_id
        return self._post("/api/v1/client/register", payload)

    def fetch_total_samples(self):
        try:
            r = requests.get(self.server_url + "/api/v1/summary", timeout=5).json()
            return int(r.get("N_T", 0))
        except Exception:
            return 0

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
