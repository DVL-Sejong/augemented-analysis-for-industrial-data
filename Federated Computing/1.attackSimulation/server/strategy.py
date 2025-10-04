
# server/strategy.py
from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple, Any
import os, time, json
import requests
import numpy as np
import flwr as fl
from flwr.common import FitRes, Scalar, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.strategy import FedAvg

from .security import SecurityState, flatten_params, build_feature_vector, OCOutlier, KMeansGuard, l2_norm, cosine_sim

FLASK_HOST = os.getenv("FLASK_HOST", "http://127.0.0.1:5000")

def default_metrics_agg(results: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    if not results:
        return {}
    total_examples = sum(n for n,_ in results)
    avg: Dict[str, float] = {}
    for n, metrics in results:
        for k, v in (metrics or {}).items():
            if isinstance(v, (int, float)):
                avg[k] = avg.get(k, 0.0) + (float(v) * n) / max(total_examples, 1)
    return avg

class SecurityFedAvg(FedAvg):
    """FedAvg + (optional) integrity checks and attack detection.
       - During ATTACK phase: behave like FedAvg (collect evidence only).
       - During DEFEND phase: apply OCSVM (A) or KMeans (B) and weight down suspicious updates.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._round = 0
        self._prev_global: Optional[List[np.ndarray]] = None
        self._ocsvm: Optional[OCOutlier] = None

    # attach scenario/phase to client config
    def configure_fit(self, server_round: int, parameters, client_manager):
        cfg = super().configure_fit(server_round, parameters, client_manager)
        st = SecurityState()
        new_cfg = []
        for cp, fitins in cfg:
            c = dict(fitins.config or {})
            c["server_round"] = server_round
            c["scenario"] = st.scenario
            c["phase"] = st.phase
            fitins.config = c
            new_cfg.append((cp, fitins))
        # keep a copy of current global for delta computation
        self._prev_global = parameters_to_ndarrays(parameters) if parameters is not None else None
        return new_cfg

    def _post_json(self, path: str, payload: Dict[str, Any]):
        try:
            requests.post(f"{FLASK_HOST}{path}", json=payload, timeout=5)
        except Exception:
            pass

    def aggregate_fit(self, server_round: int,
                      results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
                      failures: List[BaseException]):

        st = SecurityState()
        prev_global = flatten_params(self._prev_global or [])
        start = time.time()

        # === Build features per client ===
        feats, cids, n_i, client_metrics, deltas = [], [], [], [], []
        for cp, fr in results:
            cids.append(getattr(cp, "cid", "unknown"))
            nds = parameters_to_ndarrays(fr.parameters)
            curr = flatten_params(nds)
            delta = curr - prev_global if prev_global.size else curr.copy()
            deltas.append(delta)
            n_i.append(int(fr.num_examples))
            client_metrics.append(fr.metrics or {})
            feats.append(build_feature_vector(delta, prev_global, fr.metrics or {}))
        X = np.stack(feats) if feats else np.zeros((0, 4), dtype=np.float32)

        # === Decide weights during DEFEND phase ===
        weights = [1.0]*len(results)
        alerts: List[Dict[str, Any]] = []
        exclusions: List[str] = []
        if st.phase == "defend" and len(results) >= 2:
            if st.scenario == "A":
                # OCSVM on features
                if self._ocsvm is None:
                    self._ocsvm = OCOutlier(nu=0.2, gamma="scale")
                    self._ocsvm.fit(X if X.shape[0] >= 2 else np.zeros((2, X.shape[1] if X.size else 1), dtype=np.float32))
                pred = self._ocsvm.predict(X)  # -1 outlier
                for i, p in enumerate(pred):
                    if p == -1:
                        weights[i] = 0.0
                        exclusions.append(cids[i])
                        alerts.append({"level":"warn","source":"ocsvm","message":"Detected anomalous update",
                                       "round_id":server_round,"client_id":cids[i]})
            elif st.scenario == "B":
                kmg = KMeansGuard(k=2, random_state=0)
                out = kmg.split(X)
                labels = out.get("labels")
                suspect = out.get("suspect_label", 1)
                if labels is not None:
                    # 의심 군집 비율이 100%가 되지 않도록 최소 1명은 살림
                    for i, lab in enumerate(labels):
                        if lab == suspect:
                            weights[i] = 0.0
                            exclusions.append(cids[i])
                            alerts.append({"level":"warn","source":"kmeans","message":"Suspected colluding cluster",
                                           "round_id":server_round,"client_id":cids[i]})
            elif st.scenario == "C":
                norms = [l2_norm(d) for d in deltas]
                q = np.percentile(norms, 90) if norms else 0.0
                for i, nrm in enumerate(norms):
                    if q > 0 and nrm > 3.0*q:
                        weights[i] = 0.0
                        exclusions.append(cids[i])
                        alerts.append({"level":"warn","source":"norm","message":"Abnormal magnitude (phishing/poison?)",
                                       "round_id":server_round,"client_id":cids[i]})

        # === Fallback guards: ensure at least one client contributes ===
        if len(results) > 0 and sum(1 for w in weights if w > 0) == 0:
            # 가장 '정상'에 가까운 1명 살리기: Δθ L2 기준 최소값
            keep_idx = int(np.argmin([l2_norm(d) for d in deltas])) if deltas else 0
            weights[keep_idx] = 1.0
            alerts.append({"level":"info","source":"guard","message":"All filtered; kept one for aggregation fallback",
                           "round_id":server_round,"client_id":cids[keep_idx]})

        # Post alerts
        for al in alerts:
            al["scenario"] = st.scenario; al["phase"] = st.phase
            self._post_json("/api/v1/alert", al)

        # === Build modified results with adjusted num_examples ===
        mod_results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]] = []
        for w, (cp, fr) in zip(weights, results):
            if w >= 1.0:
                mod_results.append((cp, fr))
            elif w <= 0.0:
                new_fr = FitRes(
                    status=fr.status,
                    num_examples=0,
                    metrics=fr.metrics,
                    parameters=fr.parameters
                )
                mod_results.append((cp, new_fr))
            else:
                new_fr = FitRes(
                    status=fr.status,
                    num_examples=max(1, int(fr.num_examples * w)),
                    metrics=fr.metrics,
                    parameters=fr.parameters
                )
                mod_results.append((cp, new_fr))

        # === LAST SAFETY: if total becomes 0, fall back to original results ===
        if sum(fr.num_examples for _, fr in mod_results) == 0:
            self._post_json("/api/v1/alert", {
                "level":"info","source":"guard","message":"Zero total weight; falling back to original results",
                "round_id":server_round,"client_id":"-",
                "scenario":st.scenario,"phase":st.phase
            })
            mod_results = results

        # === Aggregate using parent ===
        params, metrics = super().aggregate_fit(server_round, mod_results, failures)
        dur_ms = (time.time() - start) * 1000.0

        # === Round summary -> Flask ===
        agg = default_metrics_agg([(int(fr.num_examples), fr.metrics or {}) for _, fr in results])
        payload = {
            "round": server_round,
            "participants": cids,
            "failures": len(failures),
            "duration_ms": dur_ms,
            "agg_metrics": agg,
            "scenario": st.scenario,
            "phase": st.phase,
            "exclusions": exclusions,
        }
        self._post_json("/api/v1/round_summary", payload)

        self._round = server_round
        # Update previous global to new aggregated
        if params is not None:
            self._prev_global = [a.copy() for a in parameters_to_ndarrays(params)]
        return params, metrics

def make_strategy() -> fl.server.strategy.Strategy:
    # Keep default FedAvg fractions to ensure all 3 clients participate
    return SecurityFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=2,
        min_available_clients=2,
        on_fit_config_fn=None,
    )
