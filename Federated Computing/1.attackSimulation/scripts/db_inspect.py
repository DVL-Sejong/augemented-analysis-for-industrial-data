# scripts/db_inspect.py
import sqlite3, json, os
DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fedcomp.db")
con = sqlite3.connect(DB)
con.row_factory = sqlite3.Row

def rows(q, p=()):
    cur = con.execute(q, p)
    return [dict(r) for r in cur.fetchall()]

print("== tables ==")
print(rows("SELECT name FROM sqlite_master WHERE type='table'"))

print("\n== clients ==")
print(rows("SELECT * FROM clients"))

print("\n== rounds (latest 5) ==")
print(rows("SELECT * FROM rounds ORDER BY round_id DESC LIMIT 5"))

print("\n== metrics (latest 5) ==")
print(rows("SELECT round_id, client_id, R, H, loss, acc, rtt_ms FROM metrics ORDER BY id DESC LIMIT 5"))

print("\n== local_analysis (latest 3) ==")
print(rows("SELECT round_id, client_id, payload_json FROM local_analysis ORDER BY id DESC LIMIT 3"))

con.close()
