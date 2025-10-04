
# scripts/run_scenarios.py
import os, subprocess, time, sys, json, http.client

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PY = sys.executable

def post_phase(scenario: str, phase: str):
    try:
        import requests
        requests.post("http://127.0.0.1:5000/api/v1/phase", json={
            "scenario": scenario, "phase": phase
        }, timeout=3)
    except Exception:
        pass

def run_once(scenario: str, phase: str, rounds: int = 5):
    env = os.environ.copy()
    env["SCENARIO"] = scenario; env["PHASE"] = phase; env["ROUNDS"] = str(rounds)
    print(f"[RUN] Scenario={scenario} Phase={phase} Rounds={rounds}")
    post_phase(scenario, phase)
    # Launch 3 clients (new consoles so they can run concurrently)
    clients = []
    client_mods = ["client.client1", "client.client2", "client.client3"]
    for m in client_mods:
        p = subprocess.Popen([PY, "-m", m], cwd=ROOT, env=env, creationflags=subprocess.CREATE_NEW_CONSOLE)
        clients.append(p)
        time.sleep(0.3)
    # Run server in this process and wait
    code = subprocess.call([PY, "-m", "server.flower_server"], cwd=ROOT, env=env)
    print(f"[DONE] Server exit code {code}")
    # Give clients a moment to finish then terminate if still alive
    time.sleep(2.0)
    for p in clients:
        if p.poll() is None:
            try:
                p.terminate()
            except Exception:
                pass

def main():
    rounds = int(os.getenv("ROUNDS", "5"))
    plan = [("A","attack"), ("A","defend"), ("B","attack"), ("B","defend"), ("C","attack"), ("C","defend")]
    print("[INFO] Orchestrating scenarios: ", plan)
    for sc, ph in plan:
        run_once(sc, ph, rounds=rounds)
    print("[ALL DONE]")

if __name__ == "__main__":
    main()
