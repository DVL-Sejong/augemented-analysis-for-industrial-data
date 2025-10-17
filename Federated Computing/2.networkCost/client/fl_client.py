# client/fl_client.py
import hashlib
from typing import Dict, Tuple, List, Any
import numpy as np
import flwr as fl

from common.data import make_synthetic_binary, train_logreg_numpy
from .reporter import Reporter
from .analyzer import label_hist, entropy_from_hist
from .net_sim import NetworkSimulator, make_default_multitier

FLASK_URL = "http://127.0.0.1:5000"

def _digest_of_params(w: np.ndarray, b: float) -> str:
    h = hashlib.sha256()
    h.update(w.astype(np.float32).tobytes())
    h.update(np.array([b], dtype=np.float32).tobytes())
    return h.hexdigest()[:16]

class NumpyLogRegClient(fl.client.NumPyClient):
    def __init__(self, client_id: str, N: int, d: int, class_probs: Tuple[float,float], net_profile: Dict, seed=0):
        self.client_id = client_id
        self.N, self.d = N, d
        self.class_probs = class_probs
        self.seed = seed
        self.X, self.y = make_synthetic_binary(N, d, class_probs=class_probs, seed=seed)
        # 파라미터 초기화
        self.w = np.zeros(d, dtype=np.float32)
        self.b = 0.0
        # 보고자/네트워크
        self.http_net = NetworkSimulator(**net_profile)
        self.reporter = Reporter(client_id, FLASK_URL, net=self.http_net)
        # 다계층 DAG 네트워크 (모델 업데이트 전송 비용 계산용)
        self.tier = make_default_multitier(net_profile, edge_mode="quant8")  # 기본: quant8
        self.edge_mode = "quant8"

    # --- Flower 필수 ---
    def get_parameters(self, config: Dict[str,Any]):
        return [self.w.copy(), np.array([self.b], dtype=np.float32)]

    def set_parameters(self, parameters: List[np.ndarray]):
        self.w = parameters[0].astype(np.float32).copy()
        self.b = float(parameters[1].astype(np.float32).ravel()[0])

    def fit(self, parameters, config):
        if parameters is not None:
            self.set_parameters(parameters)

        # 1) 로컬 학습
        w, b, trainm = train_logreg_numpy(
            self.X, self.y, w=self.w, b=self.b,
            epochs=5, lr=0.1, batch=64
        )
        self.w, self.b = w.astype(np.float32), float(b)

        # 2) 로컬 분석 (R/H) + JSON-safe hist
        hist = label_hist(self.y)  # 예: {0: n0, 1: n1} (np.int64 가능)
        hist_json = {str(int(k)): int(v) for k, v in hist.items()}
        R = float(sum(int(v) for v in hist.values()) / max(len(self.y), 1))
        H = float(entropy_from_hist(hist))

        # 3) 모델 업데이트 바이트/다이제스트
        model_bytes = int((self.w.size + 1) * 4)          # float32 파라미터 + bias(1)
        model_digest = _digest_of_params(self.w, self.b)  # 전송 키

        # 4) 다계층 네트워크 전송 시뮬레이션 (Edge/Cache/DAG)
        tier_res = self.tier.transmit(
            cache_key=f"model:{model_digest}",
            payload_bytes=model_bytes
        )

        # 5) 메트릭 리포트 (네트워크 비용/경로 포함)
        extra_comm = {
            "tier_rtt_ms": tier_res["rtt_ms"],
            "tier_cost": tier_res["cost"],
            "tier_path": "->".join(tier_res["path"]),
            "cache_hit": bool(tier_res["cache_hit_at"]),
            "bytes_after_preproc": tier_res["bytes_after_preproc"],
            "edge_mode": tier_res["edge_mode"],
        }
        self.reporter.send_metrics(
            int(config.get("server_round", 0)),
            loss=float(trainm["loss"]), acc=float(trainm["acc"]),
            bytes_up=model_bytes,  # 원본 크기(전처리 전) 기준; 필요하면 bytes_after_preproc로 바꿔도 됨
            R=R, H=H, extra_comm=extra_comm
        )

        # 6) 로컬 분석 테이블(히트맵용)
        self.reporter.send_local_analysis(
            int(config.get("server_round", 0)),
            {"label_hist": hist_json},
            R
        )

        # 7) Flower 규약 반환
        return self.get_parameters(config), self.N, {
            "loss": float(trainm["loss"]),
            "acc": float(trainm["acc"])
        }



    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        z = self.X @ self.w + self.b
        p = 1/(1+np.exp(-z))
        loss = float(-np.mean(self.y*np.log(p+1e-8) + (1-self.y)*np.log(1-p+1e-8)))
        acc = float(np.mean((p>=0.5) == self.y))
        return 0.0, self.N, {"loss": loss, "acc": acc}

def run_client(client_id: str, N: int, d: int, class_probs: Tuple[float,float], net_profile: Dict, seed=0):
    client = NumpyLogRegClient(client_id, N, d, class_probs, net_profile, seed=seed)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
