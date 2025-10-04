
# server/app.py
from __future__ import annotations
from flask import Flask, request, jsonify, Response, render_template, abort
from datetime import datetime
import json, time, os, hmac, hashlib, threading
from typing import Dict, Any, Tuple, List
from .db import init_db, upsert_client, total_samples, execute, query_all, query_one, ensure_columns
from .utils import subscribe, unsubscribe, push_event
from .aggregator import SimpleFederatedAggregator

app = Flask(__name__, template_folder="templates", static_folder="static")
init_db()
# Ensure schema extensions (scenario/phase/window)
ensure_columns()
agg = SimpleFederatedAggregator()

# ---- in-memory state for scenario/phase ----
_state_lock = threading.Lock()
_state = {"scenario": None, "phase": None, "window_id": None}
def get_state():
    with _state_lock:
        return dict(_state)
def set_state(scenario: str, phase: str, window_id: str):
    with _state_lock:
        _state["scenario"] = scenario
        _state["phase"] = phase
        _state["window_id"] = window_id

def require_hmac() -> bool:
    s = get_state()
    return (s.get("phase") == "defend")

def verify_hmac(raw: bytes) -> Tuple[bool, str]:
    secret = os.getenv("FEDCOMP_SECRET", "")
    if not secret:
        return (True, "no-secret-set")
    sig = request.headers.get("X-Signature", "")
    calc = hmac.new(secret.encode("utf-8"), raw, hashlib.sha256).hexdigest()
    return (hmac.compare_digest(sig, calc), calc)

@app.route("/")
def index():
    return render_template("index.html")

@app.post("/api/v1/client/register")
def post_client_register():
    data = request.get_json(force=True)
    upsert_client(data)
    push_event({"type":"client_register","data":data})
    return jsonify({"ok": True})

@app.post("/api/v1/metrics")
def post_metrics():
    raw = request.get_data()
    ok, calc = verify_hmac(raw)
    data = request.get_json(force=True)
    st = get_state()
    # Enforce only in defend phase
    if require_hmac() and not ok:
        # record alert and reject
        execute("INSERT INTO alerts(level, source, message, round_id, client_id) VALUES (?,?,?,?,?)",
                ("error","hmac","invalid signature (metrics)",
                 int(data.get("round",0)), data.get("client_id","")))
        push_event({"type":"alert","data":{"level":"error","source":"hmac","message":"invalid signature (metrics)","round":data.get("round"),"client_id":data.get("client_id")}})
        abort(401)
    # store metrics
    execute("""
      INSERT INTO metrics(round_id, client_id, R, H, loss, acc, rtt_ms, bytes_up, bytes_down, cpu_pct, mem_pct, scenario, phase, window_id)
      VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
      data.get("round"), data.get("client_id"),
      data.get("R"), data.get("H"),
      data.get("train",{}).get("loss"), data.get("train",{}).get("acc"),
      data.get("comm",{}).get("rtt_ms"), data.get("comm",{}).get("bytes_up"), data.get("comm",{}).get("bytes_down"),
      data.get("sys",{}).get("cpu_pct"), data.get("sys",{}).get("mem_pct"),
      st.get("scenario"), st.get("phase"), st.get("window_id"),
    ))
    push_event({"type":"metrics","data":{**data, **st}})
    return jsonify({"ok": True, "rtt_ms": (time.time()*1000)%500})

@app.post("/api/v1/analysis/local")
def post_local_analysis():
    raw = request.get_data()
    ok, calc = verify_hmac(raw)
    data = request.get_json(force=True)
    st = get_state()
    if require_hmac() and not ok:
        execute("INSERT INTO alerts(level, source, message, round_id, client_id) VALUES (?,?,?,?,?)",
                ("error","hmac","invalid signature (analysis)",
                 int(data.get("round",0)), data.get("client_id","")))
        push_event({"type":"alert","data":{"level":"error","source":"hmac","message":"invalid signature (analysis)","round":data.get("round"),"client_id":data.get("client_id")}})
        abort(401)
    payload = json.dumps(data.get("analysis",{}), ensure_ascii=False)
    execute("""
      INSERT INTO local_analysis(round_id, client_id, payload_json)
      VALUES (?,?,?)
    """, (data.get("round"), data.get("client_id"), payload))
    R = float(data.get("R", 1.0))
    agg.add_local(int(data.get("round",0)), data.get("client_id",""), payload, R=R)
    push_event({"type":"analysis_local","data":{**data, **st}})
    summary = agg.summarize_round(int(data.get("round",0)))
    push_event({"type":"analysis_round","data":summary})
    return jsonify({"ok": True})

@app.post("/api/v1/model/update")
def post_model_update():
    raw = request.get_data()
    ok, _ = verify_hmac(raw)
    data = request.get_json(force=True)
    if require_hmac() and not ok:
        execute("INSERT INTO alerts(level, source, message, round_id, client_id) VALUES (?,?,?,?,?)",
                ("error","hmac","invalid signature (model/update)",
                 int(data.get("round",0)), data.get("client_id","")))
        push_event({"type":"alert","data":{"level":"error","source":"hmac","message":"invalid signature (model/update)","round":data.get("round"),"client_id":data.get("client_id")}})
        abort(401)
    execute("""INSERT INTO models(round_id, model_digest, strategy, notes) VALUES (?,?,?,?)""",
            (int(data.get("round",0)), data.get("model_digest",""), data.get("strategy",""), ""))
    push_event({"type":"model_update","data":data})
    return jsonify({"ok": True})

@app.post("/api/v1/round_summary")
def post_round_summary():
    data = request.get_json(force=True)
    rid = int(data.get("round", 0))
    st = get_state()
    row = query_one("SELECT round_id FROM rounds WHERE round_id=?", (rid,))
    if row:
        execute("""UPDATE rounds SET aggregated=1, finished_at=datetime('now'),
                   global_acc=?, global_loss=?, scenario=?, phase=?, window_id=? WHERE round_id=?""",
                (data.get("agg_metrics",{}).get("acc"),
                 data.get("agg_metrics",{}).get("loss"),
                 st.get("scenario"), st.get("phase"), st.get("window_id"), rid))
    else:
        execute("""INSERT INTO rounds(round_id, aggregated, finished_at, global_acc, global_loss, scenario, phase, window_id)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (rid, 1, datetime.utcnow().isoformat(),
                 data.get("agg_metrics",{}).get("acc"), data.get("agg_metrics",{}).get("loss"),
                 st.get("scenario"), st.get("phase"), st.get("window_id")))
    push_event({"type":"round_summary","data":{**data, **st}})

    # Cross-check: any metrics for this round whose client_id not in participants?
    participants = set(data.get("participants") or [])
    bads = query_all("SELECT DISTINCT client_id FROM metrics WHERE round_id=?", (rid,))
    for b in bads:
        cid = b["client_id"]
        if cid and participants and cid not in participants:
            execute("INSERT INTO alerts(level, source, message, round_id, client_id) VALUES (?,?,?,?,?)",
                    ("warn","crosscheck","metrics from non-participant", rid, cid))
            push_event({"type":"alert","data":{"level":"warn","source":"crosscheck","message":"metrics from non-participant","round":rid,"client_id":cid}})
    return jsonify({"ok": True})

@app.post("/api/v1/phase")
def post_phase():
    data = request.get_json(force=True)
    scenario = str(data.get("scenario","A")).upper()
    phase = str(data.get("phase","attack")).lower()
    window_id = data.get("window_id") or f"{int(time.time())%100000}-{scenario}-{phase}"
    set_state(scenario, phase, window_id)
    push_event({"type":"phase","data":{"scenario":scenario,"phase":phase,"window_id":window_id}})
    return jsonify({"ok": True, "window_id": window_id})

@app.post("/api/v1/alert")
def post_alert():
    data = request.get_json(force=True)
    execute("INSERT INTO alerts(level, source, message, round_id, client_id) VALUES (?,?,?,?,?)",
            (data.get("level","info"), data.get("source","sys"), data.get("message",""),
             int(data.get("round_id",0)), data.get("client_id","")))
    push_event({"type":"alert","data":data})
    return jsonify({"ok": True})

@app.get("/api/v1/summary")
def get_summary():
    nt = total_samples()
    latest = query_one("SELECT * FROM rounds ORDER BY round_id DESC LIMIT 1")
    latest_dict = dict(latest) if latest else {}
    return jsonify({"N_T": nt, "latest_round": latest_dict})

# ===== SSE stream with bootstrap that includes rounds/metrics with scenario/phase =====
@app.get("/api/v1/dashboard/stream")
def sse_stream():
    q = subscribe()
    def gen():
        try:
            # bootstrap
            clients = [dict(r) for r in query_all("SELECT * FROM clients ORDER BY client_id ASC")]
            rounds = [dict(r) for r in query_all("SELECT * FROM rounds ORDER BY round_id ASC")]
            metrics_rows = [dict(r) for r in query_all("SELECT * FROM metrics ORDER BY round_id ASC, client_id ASC")]
            logs = [dict(r) for r in query_all("SELECT * FROM alerts ORDER BY id DESC LIMIT 200")]
            init_payload = {
                "type":"bootstrap",
                "data":{
                    "clients": clients,
                    "rounds": rounds,
                    "metrics": metrics_rows,
                    "logs": logs,
                }
            }
            yield f"data: {json.dumps(init_payload, ensure_ascii=False)}\n\n"
            # stream
            while True:
                data = q.get()
                yield f"data: {data}\n\n"
        except GeneratorExit:
            unsubscribe(q)
    return Response(gen(), mimetype="text/event-stream")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
