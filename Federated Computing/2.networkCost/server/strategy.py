# server/strategy.py
from typing import Callable, Dict, List, Optional, Tuple
import time
import requests
import flwr as fl
from flwr.common import FitRes, Scalar
from flwr.server.strategy import FedAvg

FLASK_HOST = "http://127.0.0.1:5000"

# === 가중 평균 집계 (num_examples 비율 가중) ===
def default_metrics_agg(results: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    if not results:
        return {}
    total = sum(n for n, _ in results)
    if total <= 0:
        return {}
    agg: Dict[str, float] = {}
    for n, m in results:
        w = n / total
        for k, v in (m or {}).items():
            try:
                agg[k] = agg.get(k, 0.0) + w * float(v)
            except Exception:
                pass  # 숫자 아닌 값은 무시
    return agg

class ReportingFedAvg(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._round_start: float = time.time()

    # 클라이언트로 현재 라운드 번호 전달 + 라운드 시작 시각 기록
    def configure_fit(self, server_round, parameters, client_manager):
        self._round_start = time.time()  # 라운드 전체 시간: 여기서 시작
        cfg = super().configure_fit(server_round, parameters, client_manager)
        patched = []
        for cp, fitins in cfg:
            fitins.config = dict(fitins.config or {})
            fitins.config["server_round"] = server_round
            patched.append((cp, fitins))
        return patched

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[BaseException],
    ):
        # 기본 FedAvg 집계 수행
        out = super().aggregate_fit(server_round, results, failures)
        duration_ms = int((time.time() - self._round_start) * 1000)

        # super가 None을 반환할 수도 있으므로 방어
        if out is None:
            try:
                requests.post(
                    f"{FLASK_HOST}/api/v1/round_summary",
                    json={
                        "round": server_round,
                        "participants": len(results),
                        "failures": len(failures),
                        "duration_ms": duration_ms,
                        "agg_metrics": {},
                    },
                    timeout=3,
                )
            except Exception:
                pass
            return None

        params_agg, metrics_agg_from_super = out

        # 클라 개별 metrics 수집해서 가중 평균(안 들어오면 빈 dict)
        fit_metrics_for_agg: List[Tuple[int, Dict[str, Scalar]]] = []
        for _cp, fitres in results:
            n = int(getattr(fitres, "num_examples", 0))
            m = fitres.metrics or {}
            fit_metrics_for_agg.append((n, m))

        # super().aggregate_fit가 이미 집계해 준 값이 있으면 우선 사용
        agg = dict(metrics_agg_from_super or {})
        if not agg:
            agg = default_metrics_agg(fit_metrics_for_agg)

        # 우리가 대시보드에서 쓰는 핵심 지표만 float로 보장
        acc = float(agg.get("acc", 0.0))
        loss = float(agg.get("loss", 0.0))

        payload = {
            "round": server_round,
            "participants": len(results),
            "failures": len(failures),
            "duration_ms": duration_ms,     # 라운드 전체 소요
            "agg_metrics": {"acc": acc, "loss": loss},
        }
        try:
            requests.post(f"{FLASK_HOST}/api/v1/round_summary", json=payload, timeout=3)
        except Exception:
            pass

        # metrics에도 acc/loss를 넣어 반환(선택)
        return params_agg, {"acc": acc, "loss": loss}

def make_strategy() -> fl.server.strategy.Strategy:
    return ReportingFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=2,
        min_available_clients=3,
        evaluate_metrics_aggregation_fn=default_metrics_agg,
        fit_metrics_aggregation_fn=default_metrics_agg,
    )
