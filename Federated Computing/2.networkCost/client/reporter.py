# client/reporter.py
import time, json, requests, platform, psutil
from typing import Dict, Any, Optional
from .net_sim import NetworkSimulator
import numpy as np  # <-- 추가

def _json_safe(obj):
    """numpy/ndarray/np.scalar, dict-keys 등 JSON 직렬화 안전 변환"""
    if isinstance(obj, dict):
        # 키는 전부 문자열화
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.generic,)):  # np.int64, np.float32, np.bool_ 등
        return obj.item()
    return obj

class Reporter:
    def __init__(self, client_id: str, server_url="http://127.0.0.1:5000", net: NetworkSimulator=None):
        self.client_id = client_id
        self.server_url = server_url.rstrip("/")
        self.net = net or NetworkSimulator()

    def _post(self, path: str, payload: Dict[str, Any]):
        # JSON-safe 변환
        safe_payload = _json_safe(payload)
        data = json.dumps(safe_payload, ensure_ascii=False).encode("utf-8")

        t0 = time.time()
        ok = self.net.transmit(len(data))  # 시뮬 지연
        try:
            res = requests.post(self.server_url + path, json=safe_payload, timeout=5)
            rt = (time.time() - t0)*1000.0
            if res.ok:
                out = res.json() if res.headers.get("content-type","").startswith("application/json") else {}
                out["ok"] = True
                out["rtt_ms"] = out.get("rtt_ms", rt)
                return out
            return {"ok": False, "rtt_ms": rt, "error": f"HTTP {res.status_code}"}
        except Exception as e:
            rt = (time.time() - t0)*1000.0
            return {"ok": False, "rtt_ms": rt, "error": str(e)}

    def send_metrics(
        self,
        round_id: int,
        *,
        loss: float,
        acc: float,
        bytes_up: int,
        bytes_down: int = 0,
        R: Optional[float] = None,
        H: Optional[float] = None,
        extra_comm: Optional[Dict[str,Any]] = None
    ):
        sysinfo = {
            "platform": platform.platform(),
            "cpu": psutil.cpu_count(logical=True),
            "mem_gb": round(psutil.virtual_memory().total/1024**3, 2),
        }
        comm = {
            "bytes_up": int(bytes_up),
            "bytes_down": int(bytes_down),
        }
        if extra_comm:
            comm.update(extra_comm)

        payload = {
            "type": "metrics",
            "client_id": self.client_id,
            "round": int(round_id),
            "perf": {"loss": float(loss), "acc": float(acc)},
            "comm": comm,
            "R": None if R is None else float(R),
            "H": None if H is None else float(H),
            "sys": sysinfo,
        }
        res = self._post("/api/v1/metrics", payload)
        if res.get("ok"):
            payload["comm"]["http_rtt_ms"] = res.get("rtt_ms")
        return res

    def send_local_analysis(self, round_id: int, analysis: Dict[str,Any], R: float):
        payload = { "type": "local_analysis", "client_id": self.client_id, "round": int(round_id), "analysis": analysis, "R": float(R) }
        return self._post("/api/v1/analysis/local", payload)

    def send_model_update_meta(self, round_id: int, model_digest: str, strategy="FedAvg"):
        payload = { "type":"model_update", "client_id": self.client_id, "round": int(round_id), "model_digest": model_digest, "strategy": strategy }
        return self._post("/api/v1/model/update", payload)
