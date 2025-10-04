# server/strategy.py
from typing import Callable, Dict, List, Optional, Tuple
import requests
import time
import flwr as fl
from flwr.common import FitRes, Parameters, Scalar
from flwr.server.strategy import FedAvg

FLASK_HOST = "http://127.0.0.1:5000"

def default_metrics_agg(results: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    if not results:
        return {}
    total_examples = sum(n for n,_ in results)
    avg = {}
    for n, metrics in results:
        for k,v in metrics.items():
            try:
                avg[k] = avg.get(k, 0.0) + (n/total_examples)*float(v)
            except Exception:
                pass
    return avg

class ReportingFedAvg(FedAvg):
    def __init__(self, *args, **kwargs):
        self._round = 0
        super().__init__(*args, **kwargs)

    # >>> 추가: 클라이언트로 현재 라운드를 config에 박아줌
    def configure_fit(self, server_round, parameters, client_manager):
        cfg = super().configure_fit(server_round, parameters, client_manager)
        new_cfg = []
        for cp, fitins in cfg:
            fitins.config = dict(fitins.config or {})
            fitins.config["server_round"] = server_round
            new_cfg.append((cp, fitins))
        return new_cfg
    # <<< 추가 끝

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[BaseException],
    ):
        start = time.time()
        params, metrics = super().aggregate_fit(server_round, results, failures)
        dur = (time.time() - start)*1000.0
        try:
            part = len(results)
            errs = len(failures)
            fit_metrics = [ (int(r.num_examples), r.metrics or {}) for _, r in results ]
            agg = default_metrics_agg(fit_metrics)
            payload = {
                "round": server_round,
                "participants": part,
                "failures": errs,
                "duration_ms": dur,
                "agg_metrics": agg
            }
            requests.post(f"{FLASK_HOST}/api/v1/round_summary", json=payload, timeout=5)
        except Exception:
            pass
        self._round = server_round
        return params, metrics

def make_strategy() -> fl.server.strategy.Strategy:
    return ReportingFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=2,
        min_available_clients=3,
        evaluate_metrics_aggregation_fn=default_metrics_agg,
        fit_metrics_aggregation_fn=default_metrics_agg,
    )
