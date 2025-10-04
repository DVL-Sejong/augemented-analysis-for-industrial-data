
# server/db.py
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, Tuple, List

DB_PATH = Path(__file__).resolve().parents[1] / "fedcomp.db"
_lock = threading.Lock()

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS clients (
  client_id TEXT PRIMARY KEY,
  registered_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  cpu TEXT, mem TEXT,
  net_latency_ms REAL, net_loss_pct REAL, net_bw_mbps REAL,
  n_i INTEGER
);

CREATE TABLE IF NOT EXISTS rounds (
  round_id INTEGER PRIMARY KEY,
  started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  finished_at DATETIME,
  aggregated INTEGER DEFAULT 0,
  global_acc REAL, global_loss REAL,
  scenario TEXT, phase TEXT, window_id TEXT
);

CREATE TABLE IF NOT EXISTS metrics (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts DATETIME DEFAULT CURRENT_TIMESTAMP,
  round_id INTEGER, client_id TEXT,
  R REAL, H REAL, loss REAL, acc REAL,
  rtt_ms REAL, bytes_up INTEGER, bytes_down INTEGER,
  cpu_pct REAL, mem_pct REAL,
  scenario TEXT, phase TEXT, window_id TEXT
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

def ensure_columns():
    """Idempotent ALTER TABLE to add new columns if missing."""
    with _lock:
        conn = get_conn()
        try:
            def has_col(table: str, col: str) -> bool:
                cur = conn.execute(f"PRAGMA table_info({table})")
                cols = [r[1] for r in cur.fetchall()]
                return col in cols
            # rounds
            for c in ["scenario","phase","window_id"]:
                if not has_col("rounds", c):
                    conn.execute(f"ALTER TABLE rounds ADD COLUMN {c} TEXT")
            # metrics
            for c in ["scenario","phase","window_id"]:
                if not has_col("metrics", c):
                    conn.execute(f"ALTER TABLE metrics ADD COLUMN {c} TEXT")
            conn.commit()
        finally:
            conn.close()

def execute(query: str, params: Tuple = ()):
    with _lock:
        conn = get_conn()
        try:
            cur = conn.execute(query, params)
            conn.commit()
            return {"rowcount": cur.rowcount}
        finally:
            conn.close()

def query_all(query: str, params: Tuple = ()):
    with _lock:
        conn = get_conn()
        try:
            cur = conn.execute(query, params)
            return cur.fetchall()
        finally:
            conn.close()

def query_one(query: str, params: Tuple = ()):
    rows = query_all(query, params)
    if rows:
        return rows[0]
    return None

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
        float(info.get("net_profile",{}).get("latency_ms",0.0)),
        float(info.get("net_profile",{}).get("loss_pct",0.0)),
        float(info.get("net_profile",{}).get("bandwidth_mbps",0.0)),
        int(info.get("n_i", 0))
    ))

def total_samples() -> int:
    r = query_one("SELECT SUM(n_i) as total FROM clients")
    return int((r and r["total"]) or 0)
