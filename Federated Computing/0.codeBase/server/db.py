# server/db.py
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, Tuple

DB_PATH = Path(__file__).resolve().parents[1] / "fedcomp.db"
_lock = threading.Lock()

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS clients (
  client_id TEXT PRIMARY KEY,
  registered_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  cpu TEXT, mem TEXT,
  net_latency_ms REAL, net_loss_pct REAL, net_bw_mbps REAL,
  n_i INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS rounds (
  round_id INTEGER PRIMARY KEY,
  started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  finished_at DATETIME,
  sampled INTEGER DEFAULT 0,
  aggregated INTEGER DEFAULT 0,
  global_acc REAL, global_loss REAL
);

CREATE TABLE IF NOT EXISTS metrics (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts DATETIME DEFAULT CURRENT_TIMESTAMP,
  round_id INTEGER, client_id TEXT,
  R REAL, H REAL,
  loss REAL, acc REAL,
  rtt_ms REAL, bytes_up INTEGER, bytes_down INTEGER,
  cpu_pct REAL, mem_pct REAL
);

CREATE TABLE IF NOT EXISTS local_analysis (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts DATETIME DEFAULT CURRENT_TIMESTAMP,
  round_id INTEGER, client_id TEXT,
  payload_json TEXT
);

CREATE TABLE IF NOT EXISTS alerts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts DATETIME DEFAULT CURRENT_TIMESTAMP,
  level TEXT, source TEXT, message TEXT,
  round_id INTEGER, client_id TEXT
);

CREATE TABLE IF NOT EXISTS models (
  round_id INTEGER, model_digest TEXT, strategy TEXT, notes TEXT
);
"""

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with _lock:
        conn = get_conn()
        try:
            conn.executescript(SCHEMA)
            conn.commit()
        finally:
            conn.close()

def execute(query: str, params: Tuple = ()):
    with _lock:
        conn = get_conn()
        try:
            cur = conn.execute(query, params)
            conn.commit()
            return cur
        finally:
            conn.close()

def query_all(query: str, params: Tuple = ()):
    conn = get_conn()
    try:
        cur = conn.execute(query, params)
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()

def query_one(query: str, params: Tuple = ()):
    conn = get_conn()
    try:
        cur = conn.execute(query, params)
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

def total_samples() -> int:
    row = query_one("SELECT SUM(n_i) as total FROM clients")
    return int(row["total"] or 0)

def upsert_client(info: Dict[str, Any]):
    execute("""
    INSERT INTO clients (client_id, cpu, mem, net_latency_ms, net_loss_pct, net_bw_mbps, n_i)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(client_id) DO UPDATE SET
      cpu=excluded.cpu, mem=excluded.mem,
      net_latency_ms=excluded.net_latency_ms,
      net_loss_pct=excluded.net_loss_pct,
      net_bw_mbps=excluded.net_bw_mbps,
      n_i=excluded.n_i
    """, (
        info["client_id"], info.get("cpu",""), info.get("mem",""),
        float(info["net_profile"]["latency_ms"]),
        float(info["net_profile"]["loss_pct"]),
        float(info["net_profile"]["bandwidth_mbps"]),
        int(info.get("n_i", 0))
    ))
