
# client/fl_client.py
import hashlib, os
from typing import Dict, Tuple, List
import numpy as np
import flwr as fl
import platform, psutil

from common.data import make_synthetic_binary, train_logreg_numpy
from .reporter import Reporter
from .analyzer import label_hist, entropy_from_hist
from .net_sim import NetworkSimulator
from .attack_plugin import AttackPlugin

FLASK_URL = os.getenv("FLASK_URL", "http://127.0.0.1:5000")

class NumpyLogRegClient(fl.client.NumPyClient):
    def __init__(self, client_id: str, N: int, d: int, class_probs: Tuple[float,float], net_profile: Dict, seed=0):
        self.client_id = client_id
        self.N = N; self.d = d; self.class_probs = class_probs
        self.X, self.y = make_synthetic_binary(N, d, class_probs=class_probs, seed=seed)
        self.w = np.zeros((d,), dtype=np.float32); self.b = 0.0
        self.net = NetworkSimulator(**net_profile)
        self.reporter = Reporter(client_id, server_url=FLASK_URL, net=self.net)
        self.N_T = self.reporter.fetch_total_samples() or (N*3)
        self.plugin = AttackPlugin()

        # register to Flask
        self.reporter.send_register({
            "cpu": platform.processor(),
            "mem": f"{round(psutil.virtual_memory().total/1e9,1)}GB",
            "net_profile": net_profile,
            "n_i": N
        })

    # Flower NumPyClient API
    def get_parameters(self, config):
        return [self.w, np.array([self.b], dtype=np.float32)]

    def set_parameters(self, parameters):
        self.w = parameters[0].astype(np.float32)
        self.b = float(parameters[1][0])

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        # local train
        self.w, self.b, trainm = train_logreg_numpy(self.X, self.y, self.w, self.b, lr=0.2, epochs=2, batch=64)

        # === ATTACK SIM: tamper updates/metrics when in attack phase ===
        tw, tb = self.plugin.tamper_params(self.w, self.b)
        # Use tampered for reporting to server aggregation (Flower), but keep honest metrics by default then tamper
        self.w, self.b = tw, tb

        # metrics + analysis (may be tampered for reporting to Flask)
        hist = label_hist(self.y.tolist())
        H = entropy_from_hist(hist, base="e")
        R = float(self.N)/float(self.N_T or self.N)
        sys = {"cpu_pct": psutil.cpu_percent(interval=None), "mem_pct": psutil.virtual_memory().percent}

        tm = self.plugin.tamper_metrics({"loss": trainm["loss"], "acc": trainm["acc"]})
        self.reporter.send_metrics(round_id=int(config.get("server_round", 0)), R=R, H=H, train=tm, sys=sys,
                                   bytes_up=self.w.nbytes + 4, bytes_down=0)
        self.reporter.send_local_analysis(round_id=int(config.get("server_round", 0)),
                                          analysis={"label_hist": hist}, R=R)
        # model update meta
        digest = hashlib.sha256(np.concatenate([self.w, np.array([self.b], dtype=np.float32)]).tobytes()).hexdigest()
        self.reporter.send_model_update_meta(round_id=int(config.get("server_round", 0)), model_digest=digest)
        return self.get_parameters(config), self.N, {"loss": float(trainm["loss"]), "acc": float(trainm["acc"])}

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
