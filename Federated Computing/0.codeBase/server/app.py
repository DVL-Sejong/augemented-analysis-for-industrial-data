# server/app.py
from flask import Flask, request, jsonify, Response, render_template
from datetime import datetime
import json, time
from db import init_db, upsert_client, total_samples, execute, query_all, query_one
from utils import subscribe, unsubscribe, push_event
from aggregator import SimpleFederatedAggregator

app = Flask(__name__, template_folder="templates", static_folder="static")
init_db()
agg = SimpleFederatedAggregator()

@app.route("/")
def index():
    # 대시보드
    return render_template("index.html")

@app.post("/api/v1/client/register")
def register_client():
    data = request.get_json(force=True)
    upsert_client(data)
    nt = total_samples()
    evt = {"type":"client_register","data":{"client_id":data["client_id"],"n_i":data.get("n_i",0),"N_T":nt}}
    push_event(evt)
    return jsonify({"ok": True, "N_T": nt})

@app.post("/api/v1/metrics")
def post_metrics():
    data = request.get_json(force=True)
    execute("""
      INSERT INTO metrics(round_id, client_id, R, H, loss, acc, rtt_ms, bytes_up, bytes_down, cpu_pct, mem_pct)
      VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (
      data.get("round"), data.get("client_id"),
      data.get("R"), data.get("H"),
      data.get("train",{}).get("loss"), data.get("train",{}).get("acc"),
      data.get("comm",{}).get("rtt_ms"), data.get("comm",{}).get("bytes_up"), data.get("comm",{}).get("bytes_down"),
      data.get("sys",{}).get("cpu_pct"), data.get("sys",{}).get("mem_pct"),
    ))
    push_event({"type":"metrics","data":data})
    return jsonify({"ok": True})

@app.post("/api/v1/analysis/local")
def post_local_analysis():
    data = request.get_json(force=True)
    payload = json.dumps(data.get("analysis",{}), ensure_ascii=False)
    execute("""
      INSERT INTO local_analysis(round_id, client_id, payload_json)
      VALUES (?,?,?)
    """, (data.get("round"), data.get("client_id"), payload))
    # R 가중치(없으면 1)
    R = float(request.args.get("R", data.get("R", 1.0)))
    agg.add_local(int(data.get("round",0)), data.get("client_id",""), payload, R=R)
    push_event({"type":"analysis_local","data":data})
    # 라운드 요약도 푸시
    summary = agg.summarize_round(int(data.get("round",0)))
    push_event({"type":"analysis_round","data":summary})
    return jsonify({"ok": True})

@app.post("/api/v1/model/update")
def post_model_update():
    data = request.get_json(force=True)
    execute("""
      INSERT INTO models(round_id, model_digest, strategy, notes)
      VALUES (?,?,?,?)
    """, (data.get("round"), data.get("model_digest"), data.get("strategy","FedAvg"), data.get("notes","")))
    push_event({"type":"model_update","data":data})
    return jsonify({"ok": True})

@app.post("/api/v1/round_summary")
def post_round_summary():
    data = request.get_json(force=True)
    rid = int(data.get("round", 0))
    # rounds 테이블 upsert 유사: 존재하면 업데이트
    row = query_one("SELECT round_id FROM rounds WHERE round_id=?", (rid,))
    if row:
        execute("""UPDATE rounds SET aggregated=1, finished_at=datetime('now'),
                   global_acc=?, global_loss=? WHERE round_id=?""",
                (data.get("agg_metrics",{}).get("acc"), data.get("agg_metrics",{}).get("loss"), rid))
    else:
        execute("""INSERT INTO rounds(round_id, aggregated, finished_at, global_acc, global_loss)
                   VALUES (?,?,?,?,?)""",
                (rid, 1, datetime.utcnow().isoformat(),
                 data.get("agg_metrics",{}).get("acc"), data.get("agg_metrics",{}).get("loss")))
    push_event({"type":"round_summary","data":data})
    return jsonify({"ok": True})

@app.get("/api/v1/summary")
def get_summary():
    nt = total_samples()
    latest = query_one("SELECT * FROM rounds ORDER BY round_id DESC LIMIT 1") or {}
    return jsonify({"N_T": nt, "latest_round": latest})

from collections import Counter

def _compute_analysis_round(round_id: int):
    if not round_id:
        return None
    # R 맵 (round, client -> R)
    rmap_rows = query_all("SELECT client_id, R FROM metrics WHERE round_id=?", (round_id,))
    Rmap = {r["client_id"]: float(r["R"] or 1.0) for r in rmap_rows}
    la = query_all("SELECT client_id, payload_json FROM local_analysis WHERE round_id=?", (round_id,))
    total = Counter()
    denom = 0.0
    for r in la:
        try:
            payload = json.loads(r.get("payload_json") or "{}")
        except Exception:
            payload = {}
        hist = payload.get("label_hist", {})
        Ri = Rmap.get(r["client_id"], 1.0)
        for k, v in hist.items():
            total[k] += Ri * float(v)
        denom += Ri * sum(hist.values())
    dist = {k: (float(v)/denom if denom else 0.0) for k, v in total.items()}
    return {"round": round_id, "weighted_label_dist": dist, "contributors": len(la)}


@app.get("/api/v1/dashboard/stream")
def stream():
    q = subscribe()
    def gen():
        try:
            # ===== 부트스트랩: 과거 데이터 싹 보내기 =====
            clients = query_all("SELECT * FROM clients")
            rounds = query_all("SELECT * FROM rounds ORDER BY round_id ASC LIMIT 200")
            metrics_rows = query_all("""
                SELECT round_id, client_id, R, H
                FROM metrics
                ORDER BY id ASC LIMIT 2000
            """)
            logs = query_all("""
                SELECT
                    r.round_id AS round,
                    r.global_acc AS acc,
                    r.global_loss AS loss,
                    (SELECT COUNT(DISTINCT m.client_id)
                       FROM metrics m
                      WHERE m.round_id = r.round_id) AS participants,
                    0 AS failures
                FROM rounds r
                ORDER BY r.round_id ASC
            """)
            latest_r = rounds[-1]["round_id"] if rounds else None
            analysis = _compute_analysis_round(latest_r) if latest_r else None

            init_payload = {
                "type":"bootstrap",
                "data":{
                    "clients": clients,
                    "trend": {
                        "labels": [f"#{r['round_id']}" for r in rounds],
                        "acc":    [r["global_acc"] for r in rounds],
                        "loss":   [r["global_loss"] for r in rounds],
                    },
                    "heat": { "metrics": metrics_rows },
                    "analysis_round": analysis,
                    "logs": logs
                }
            }
            yield f"data: {json.dumps(init_payload, ensure_ascii=False)}\n\n"
            # ===== 실시간 스트림 =====
            while True:
                data = q.get()
                yield f"data: {data}\n\n"
        except GeneratorExit:
            unsubscribe(q)
    return Response(gen(), mimetype="text/event-stream")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
