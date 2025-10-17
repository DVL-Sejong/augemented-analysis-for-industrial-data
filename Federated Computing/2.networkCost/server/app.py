# server/app.py
from flask import Flask, request, jsonify, Response, render_template
import json, time
from db import init_db, execute, query_all, query_one
from utils import subscribe, unsubscribe, push_event
from aggregator import SimpleFederatedAggregator

app = Flask(__name__, template_folder="templates", static_folder="static")
init_db()
agg = SimpleFederatedAggregator()

# --------------------------
# View
# --------------------------
@app.route("/")
def index():
    return render_template("index.html")

# --------------------------
# API: Metrics (clients -> server)
# --------------------------
@app.post("/api/v1/metrics")
def post_metrics():
    d = request.get_json(force=True) or {}

    # ---- 유연 파싱
    client_id = d.get("client_id") or d.get("client")
    round_id  = int(d.get("round") or d.get("round_id") or 0)
    perf      = d.get("perf") or {}
    loss      = float(perf.get("loss") or d.get("loss") or 0.0)
    acc       = float(perf.get("acc")  or d.get("acc")  or 0.0)
    comm      = d.get("comm") or {}
    bytes_up   = int(comm.get("bytes_up")   or d.get("bytes_up")   or 0)
    bytes_down = int(comm.get("bytes_down") or d.get("bytes_down") or 0)
    rtt_ms     = float(comm.get("tier_rtt_ms") or d.get("rtt_ms") or 0.0)
    R = d.get("R"); H = d.get("H")

    # ---- DB 저장(스키마 맞는 필드만)
    try:
        execute(
            """
            INSERT INTO metrics(round_id, client_id, R, H, loss, acc, rtt_ms, bytes_up, bytes_down)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (round_id, client_id, R, H, loss, acc, rtt_ms, bytes_up, bytes_down)
        )
    except Exception:
        # 스키마가 다르면 저장을 건너뛰고 SSE만 보낸다
        pass

    # ---- SSE로 원본을 그대로 내보냄(네트워크 지표 포함)
    event = {
        "type": "metrics",
        "client_id": client_id,
        "round": round_id,
        "perf": {"loss": loss, "acc": acc},
        "comm": {
            "bytes_up": bytes_up,
            "bytes_down": bytes_down,
            "tier_rtt_ms": comm.get("tier_rtt_ms"),
            "tier_cost": comm.get("tier_cost"),
            "tier_path": comm.get("tier_path"),
            "cache_hit": comm.get("cache_hit"),
            "bytes_after_preproc": comm.get("bytes_after_preproc"),
            "edge_mode": comm.get("edge_mode"),
        },
        "R": R, "H": H,
    }
    push_event(event)

    # ---- 안전망: 라운드 요약이 아직 안 왔어도 트렌드가 비지 않게 평균 내서 즉시 푸시
    try:
        if round_id:
            rows = query_all("SELECT acc, loss, client_id FROM metrics WHERE round_id=?", (round_id,))
            if rows:
                acc_avg  = sum(float(r["acc"]  or 0.0) for r in rows) / len(rows)
                loss_avg = sum(float(r["loss"] or 0.0) for r in rows) / len(rows)
                participants = len({r["client_id"] for r in rows})
                push_event({
                    "type":"round_summary",
                    "round": round_id,
                    "participants": participants,
                    "failures": 0,
                    "duration_ms": 0,
                    "agg_metrics": {"acc": acc_avg, "loss": loss_avg},
                })
    except Exception:
        pass

    return jsonify({"ok": True})

# --------------------------
# API: Local analysis (clients -> server)
# --------------------------
@app.post("/api/v1/analysis/local")
def post_local_analysis():
    d = request.get_json(force=True) or {}
    round_id = int(d.get("round") or 0)
    client_id = d.get("client_id")
    payload = d.get("analysis") or {}
    R = float(d.get("R") or 1.0)

    try:
        execute(
            "INSERT INTO local_analysis(round_id, client_id, payload_json) VALUES (?, ?, ?)",
            (round_id, client_id, json.dumps(payload, ensure_ascii=False))
        )
    except Exception:
        pass

    # KIST 담당 모듈(집계기)에도 넣어줌
    agg.add_local(round_id, client_id, json.dumps(payload, ensure_ascii=False), R)

    push_event({
        "type": "local_analysis",
        "round": round_id,
        "client_id": client_id,
        "analysis": payload,
        "R": R,
    })
    return jsonify({"ok": True})

# --------------------------
# API: Round summary (Flower strategy -> server)
# --------------------------
@app.post("/api/v1/round_summary")
def post_round_summary():
    d = request.get_json(force=True) or {}
    r = int(d.get("round") or 0)
    metrics = d.get("agg_metrics") or {}
    acc = float(metrics.get("acc") or 0.0)
    loss = float(metrics.get("loss") or 0.0)
    dur = int(d.get("duration_ms") or 0)
    part = int(d.get("participants") or 0)
    fails = int(d.get("failures") or 0)

    try:
        execute(
            """
            INSERT INTO rounds(round_id, global_acc, global_loss, duration_ms, participants, failures)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(round_id) DO UPDATE SET
              global_acc=excluded.global_acc,
              global_loss=excluded.global_loss,
              duration_ms=excluded.duration_ms,
              participants=excluded.participants,
              failures=excluded.failures
            """,
            (r, acc, loss, dur, part, fails)
        )
    except Exception:
        pass

    push_event({
        "type":"round_summary",
        "round": r,
        "participants": part,
        "failures": fails,
        "duration_ms": dur,
        "agg_metrics": {"acc": acc, "loss": loss},
    })
    return jsonify({"ok": True})

# --------------------------
# SSE Stream (dashboard)
# --------------------------
@app.get("/api/v1/dashboard/stream")
def dashboard_stream():
    q = subscribe()

    def bootstrap():
        # rounds/metrics 테이블이 없거나 비어도 안전하게
        rounds = []
        metrics = []
        try:
            rounds = query_all("SELECT round_id AS round, global_acc, global_loss, duration_ms FROM rounds ORDER BY round_id ASC")
        except Exception:
            rounds = []
        try:
            metrics = query_all("""
                SELECT round_id AS round, client_id, acc, loss, rtt_ms, bytes_up, bytes_down
                FROM metrics ORDER BY id ASC
            """)
        except Exception:
            metrics = []

        init_payload = {
            "init": {
                "history": {
                    "rounds": [
                        {"round": r["round"], "global_acc": r.get("global_acc"), "global_loss": r.get("global_loss")}
                        for r in rounds
                    ],
                    "metrics": [
                        {
                            "type":"metrics",
                            "client_id": m["client_id"],
                            "round": m["round"],
                            "perf":{"acc": m.get("acc"), "loss": m.get("loss")},
                            "comm":{"bytes_up": m.get("bytes_up"), "bytes_down": m.get("bytes_down"), "tier_rtt_ms": m.get("rtt_ms")}
                        } for m in metrics
                    ]
                }
            }
        }
        return f"data: {json.dumps(init_payload, ensure_ascii=False)}\n\n"

    def gen():
        try:
            # 부트스트랩 한 번 쏘고
            yield bootstrap()
            # 라이브 스트림
            while True:
                data = q.get()
                yield f"data: {data}\n\n"
        finally:
            unsubscribe(q)

    return Response(gen(), mimetype="text/event-stream")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
