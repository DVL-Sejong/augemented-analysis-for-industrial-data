const evt = new EventSource("/api/v1/dashboard/stream");

// === State ===
const state = {
  rounds: [],
  metrics: [],
  logs: [],
  clients: [],
  currentScenario: "A",
};
// expose for plugins/patches (and debugging)
window.state = state;

// === Utility ===
function mean(xs) { const v = xs.filter(x => Number.isFinite(x)); return v.length ? v.reduce((a,b)=>a+b,0)/v.length : null; }
function sum(xs) { return xs.reduce((a,b)=>a+(b||0),0); }
function by(arr, key) { const m = {}; arr.forEach(x => { const k = x[key]; m[k] = m[k] || []; m[k].push(x); }); return m; }
function last(arr) { return arr.length ? arr[arr.length-1] : null; }
function uniq(xs) { return Array.from(new Set(xs)); }

function windowGroupsForScenario(scn) {
  const rs = state.rounds.filter(r => r.scenario === scn);
  const byWin = by(rs, "window_id");
  const groups = Object.entries(byWin).map(([wid, arr]) => {
    const sorted = arr.slice().sort((a,b)=>a.round_id-b.round_id);
    return { window_id: wid, phase: (sorted[0]?.phase || "attack"), rounds: sorted, minRound: sorted[0]?.round_id ?? 0, maxRound: last(sorted)?.round_id ?? 0 };
  }).sort((a,b)=>a.minRound-b.minRound);
  return groups;
}

function lastAttackDefendPair(scn) {
  const groups = windowGroupsForScenario(scn);
  if (groups.length < 2) return null;
  const g1 = groups[groups.length-2];
  const g2 = groups[groups.length-1];
  const attack = g1.phase === "attack" ? g1 : g2;
  const defend = g1.phase === "defend" ? g1 : g2;
  if (!attack || !defend) return null;
  attack.last5 = attack.rounds.slice(-5);
  defend.last5 = defend.rounds.slice(-5);
  return { attack, defend };
}

// === Charts ===

// Global trend (latest window pair only)
const ctxTrend = document.getElementById("chartTrend");
const trend = new Chart(ctxTrend, {
  type: "line",
  data: { labels: [], datasets: [
    { label: "Acc", data: [], yAxisID: "y1", tension: 0.25 },
    { label: "Loss", data: [], yAxisID: "y2", tension: 0.25 },
  ]},
  options: {
    responsive: true,
    interaction: { mode: "nearest", intersect: false },
    maintainAspectRatio: false,
    scales: {
      y1: { type: "linear", position: "left", min: 0, max: 1, grid: { color: "#e2e8f0" }, ticks:{ color:"#334155" } },
      y2: { type: "linear", position: "right", grid: { color: "#f1f5f9" }, ticks:{ color:"#334155" } },
      x:  { ticks: { color: "#334155" }, grid: { color: "#f1f5f9" } },
    },
    plugins: { legend: { position: "bottom", labels:{ color:"#334155" } } },
  },
});

// Compare line charts (Acc & Loss across 5 rounds)
const cmpAcc = new Chart(document.getElementById("cmpAcc"), {
  type: "line",
  data: { labels: ["1","2","3","4","5"], datasets: [
    { label: "Attack (Acc)", data: [], borderDash:[4,2], tension: 0.25 },
    { label: "Defend (Acc)", data: [], tension: 0.25 },
  ]},
  options: {
    responsive: true,
    scales: { y: { min:0.9, max:1.0, grid:{color:"#e2e8f0"}, ticks:{color:"#334155"} }, x: { ticks:{color:"#334155"}, grid:{color:"#f1f5f9"} } },
    plugins: { legend: { position:"bottom", labels:{color:"#334155"} } }
  }
});

const cmpLoss = new Chart(document.getElementById("cmpLoss"), {
  type: "line",
  data: { labels: ["1","2","3","4","5"], datasets: [
    { label: "Attack (Loss)", data: [], borderDash:[4,2], tension: 0.25 },
    { label: "Defend (Loss)", data: [], tension: 0.25 },
  ]},
  options: {
    responsive: true,
    scales: { y: { beginAtZero:true, grid:{color:"#e2e8f0"}, ticks:{color:"#334155"} }, x: { ticks:{color:"#334155"}, grid:{color:"#f1f5f9"} } },
    plugins: { legend: { position:"bottom", labels:{color:"#334155"} } }
  }
});

// Alerts stacked bar (integrated)
const alertStack = new Chart(document.getElementById("alertStack"), {
  type: "bar",
  data: { labels: ["Attack","Defend"], datasets: [
    { label:"ocsvm", data:[0,0], stack:"A" },
    { label:"kmeans", data:[0,0], stack:"A" },
    { label:"hmac", data:[0,0], stack:"A" },
    { label:"crosscheck", data:[0,0], stack:"A" },
    { label:"other", data:[0,0], stack:"A" },
  ]},
  options: {
    responsive: true,
    scales: { y: { beginAtZero:true, grid:{color:"#e2e8f0"}, ticks:{color:"#334155"} }, x: { ticks:{color:"#334155"}, grid:{display:false} } },
    plugins: { legend: { position:"bottom", labels:{color:"#334155"} } }
  }
});

// Heatmaps (R/H), with missing-cell fill
function makeHeatmapConfig(title, colorFn) {
  return {
    type: "matrix",
    data: { datasets: [{
      label: title, data: [],
      width: 40, height: 34,
      backgroundColor: ctx => colorFn((ctx.raw && ctx.raw.v) ?? 0),
    }]},
    options: {
      plugins: { legend: { display: false } },
      maintainAspectRatio: false,
      scales: {
        x: { ticks: { color: "#475569" }, offset: true, title: { display:true, text: "Round", color:"#334155" }, grid:{ color:"#e2e8f0" } },
        y: { ticks: { color: "#475569" }, offset: true, title: { display:true, text: "Client", color:"#334155" }, grid:{ color:"#e2e8f0" } },
      },
      elements: { rectangle: { borderWidth: 1, borderColor: "#e2e8f0" } },
    },
  };
}
const colorR = v => (v < 0 ? "rgba(148,163,184,0.35)" : `rgba(37,99,235, ${0.15 + 0.85*Math.max(0, Math.min(1, v))})`);
const colorH = v => (v < 0 ? "rgba(148,163,184,0.35)" : `rgba(5,150,105, ${0.15 + 0.85*Math.max(0, Math.min(1, v))})`);
const heatR = new Chart(document.getElementById("heatR"), makeHeatmapConfig("R", colorR));
const heatH = new Chart(document.getElementById("heatH"), makeHeatmapConfig("H", colorH));

function clientIndexMap() {
  const ids = state.clients.length ? state.clients.map(c => c.client_id).sort() : Array.from(new Set(state.metrics.map(m => m.client_id))).sort();
  const mp = {}; ids.forEach((cid, i) => mp[cid] = i+1);
  return mp;
}
function pushHeat(chart, cid, round, val) {
  const yIndex = clientIndexMap()[cid] || 0;
  chart.data.datasets[0].data.push({ x: round, y: yIndex, v: val });
  chart.update("none");
}
function fillMissingHeatCells() {
  const roundsInScen = uniq(state.metrics.filter(m => (m.scenario||"") === state.currentScenario).map(m => m.round_id));
  const cids = state.clients.length ? state.clients.map(c => c.client_id) : uniq(state.metrics.map(m => m.client_id));
  const key = (cid, r) => `${cid}@@${r}`;
  const have = new Set(state.metrics.filter(m => (m.scenario||"")===state.currentScenario).map(m => key(m.client_id, m.round_id)));
  roundsInScen.forEach(rid => {
    cids.forEach(cid => {
      if (!have.has(key(cid, rid))) {
        pushHeat(heatR, cid, rid, -1);
        pushHeat(heatH, cid, rid, -1);
      }
    });
  });
}

// === KPIs update ===
function setText(id, val, suffix="") {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = (val===null || val===undefined) ? "—" : (typeof val==="number" ? (val.toFixed ? val.toFixed(3) : val)+suffix : val+suffix);
}
function refreshKPIs() {
  const pair = lastAttackDefendPair(state.currentScenario);
  if (!pair) {
    ["kpiAccAtk","kpiAccDef","kpiAccDelta","kpiLossAtk","kpiLossDef","kpiLossDelta","kpiExclAtk","kpiExclDef"].forEach(id => setText(id, null));
    alertStack.data.datasets.forEach(ds => { ds.data = [0,0]; });
    alertStack.update("none");
    return;
  }
  const atkAcc = mean(pair.attack.last5.map(r=>r.global_acc));
  const defAcc = mean(pair.defend.last5.map(r=>r.global_acc));
  const atkLoss = mean(pair.attack.last5.map(r=>r.global_loss));
  const defLoss = mean(pair.defend.last5.map(r=>r.global_loss));
  const atkExcl = mean(pair.attack.last5.map(r=>r.exclusions_count||0));
  const defExcl = mean(pair.defend.last5.map(r=>r.exclusions_count||0));

  setText("kpiAccAtk", atkAcc);
  setText("kpiAccDef", defAcc);
  setText("kpiAccDelta", (defAcc!==null && atkAcc!==null) ? (defAcc - atkAcc) : null);
  setText("kpiLossAtk", atkLoss);
  setText("kpiLossDef", defLoss);
  setText("kpiLossDelta", (defLoss!==null && atkLoss!==null) ? (atkLoss - defLoss) : null);
  setText("kpiExclAtk", atkExcl);
  setText("kpiExclDef", defExcl);

  // fill line charts
  cmpAcc.data.datasets[0].data = pair.attack.last5.map(r=>r.global_acc ?? null);
  cmpAcc.data.datasets[1].data = pair.defend.last5.map(r=>r.global_acc ?? null);
  cmpLoss.data.datasets[0].data = pair.attack.last5.map(r=>r.global_loss ?? null);
  cmpLoss.data.datasets[1].data = pair.defend.last5.map(r=>r.global_loss ?? null);
  cmpAcc.update("none"); cmpLoss.update("none");

  // --- update alert stacked bar ---
  const atkRounds = new Set(pair.attack.last5.map(r=>r.round_id));
  const defRounds = new Set(pair.defend.last5.map(r=>r.round_id));
  const counts = { atk:{ocsvm:0,kmeans:0,hmac:0,crosscheck:0,other:0}, def:{ocsvm:0,kmeans:0,hmac:0,crosscheck:0,other:0} };
  state.logs.forEach(L => {
    const src = (L.source || "").toLowerCase();
    const key = (src==="ocsvm"||src==="kmeans"||src==="hmac"||src==="crosscheck") ? src : "other";
    const rid = L.round_id ?? L.round;
    if (atkRounds.has(rid)) counts.atk[key]++;
    if (defRounds.has(rid)) counts.def[key]++;
  });
  const order = ["ocsvm","kmeans","hmac","crosscheck","other"];
  alertStack.data.datasets.forEach((ds, i) => {
    const k = order[i]; ds.data = [counts.atk[k], counts.def[k]];
  });
  alertStack.update("none");
}
window.refreshKPIs = refreshKPIs; // expose for any extensions

function refreshTrend() {
  const pair = lastAttackDefendPair(state.currentScenario);
  let rs = [];
  if (pair) rs = pair.attack.rounds.concat(pair.defend.rounds);
  trend.data.labels = rs.map(r => `#${r.round_id}`);
  trend.data.datasets[0].data = rs.map(r => r.global_acc);
  trend.data.datasets[1].data = rs.map(r => r.global_loss);
  trend.update("none");
}
window.refreshTrend = refreshTrend;

function rebuildHeat() {
  heatR.data.datasets[0].data = [];
  heatH.data.datasets[0].data = [];
  state.metrics.filter(m => (m.scenario || "") === state.currentScenario).forEach(m => {
    pushHeat(heatR, m.client_id, m.round_id, (m.R ?? 0));
    pushHeat(heatH, m.client_id, m.round_id, (m.H ?? 0));
  });
  fillMissingHeatCells();
}

// === Logs rendering ===
function logLine(html) {
  const el = document.getElementById("log");
  const div = document.createElement("div");
  div.innerHTML = html;
  el.prepend(div);
}

// === SSE ===
evt.onmessage = (ev) => {
  const msg = JSON.parse(ev.data);
  const t = msg.type;
  const d = msg.data || {};

  if (t === "bootstrap") {
    state.clients = d.clients || [];
    state.rounds = (d.rounds || []).map(r => ({
      round_id: r.round_id, global_acc: r.global_acc, global_loss: r.global_loss,
      scenario: r.scenario || "A", phase: r.phase || "attack", window_id: r.window_id || `${r.scenario||'A'}-${r.phase||'attack'}`,
      exclusions_count: r.exclusions_count || 0
    }));
    state.metrics = (d.metrics || []).map(m => ({...m, round_id: m.round_id, client_id: m.client_id}));
    state.logs = (d.logs || []).map(L => ({
      id: L.id, ts: L.ts, level: L.level, source: L.source, message: L.message, round_id: L.round_id, client_id: L.client_id
    }));
    state.logs.forEach(L => logLine(`<span class="text-sky-600">[${L.level}]</span> ${L.source}: ${L.message}`));
    rebuildHeat();
    refreshTrend();
    refreshKPIs();
  }

  if (t === "metrics") {
    state.metrics.push({ ...d, round_id: d.round, client_id: d.client_id });
    if ((d.scenario||"A") === state.currentScenario) {
      pushHeat(heatR, d.client_id, d.round, (d.R ?? 0));
      pushHeat(heatH, d.client_id, d.round, (d.H ?? 0));
      fillMissingHeatCells();
    }
  }

  if (t === "round_summary") {
    state.rounds.push({
      round_id: d.round,
      global_acc: d.agg_metrics?.acc ?? null,
      global_loss: d.agg_metrics?.loss ?? null,
      scenario: d.scenario || "A",
      phase: d.phase || "attack",
      window_id: d.window_id || `${d.scenario||'A'}-${d.phase||'attack'}`,
      exclusions_count: Array.isArray(d.exclusions) ? d.exclusions.length : 0
    });
    refreshTrend();
    refreshKPIs();
    logLine(`<span class="text-purple-600">[ROUND ${d.round}]</span> acc=${(d.agg_metrics?.acc??'--').toFixed?.(3) ?? d.agg_metrics?.acc}, loss=${(d.agg_metrics?.loss??'--').toFixed?.(3) ?? d.agg_metrics?.loss} <span class="text-slate-500">(${d.scenario}/${d.phase})</span>`);
  }

  if (t === "alert") {
    const L = { level: d.level || "info", source: d.source || "sys", message: d.message || "", round_id: d.round || d.round_id || null, client_id: d.client_id || "" };
    state.logs.push(L);
    logLine(`<span class="text-red-600">[ALERT]</span> ${L.source}: ${L.message} <span class="text-slate-500">#${L.round_id ?? '-'} ${L.client_id||''}</span>`);
    // chart will refresh on next KPI update, but also update immediately
    refreshKPIs();
  }

  if (t === "phase") {
    document.getElementById("phaseBanner").innerText = `Scenario ${d.scenario} – ${d.phase.toUpperCase()}`;
  }
};

// scenario switch
document.querySelectorAll("[data-scenario]").forEach(btn => {
  btn.addEventListener("click", () => {
    state.currentScenario = btn.getAttribute("data-scenario");
    document.querySelectorAll("[data-scenario]").forEach(b=>b.classList.remove("bg-slate-900","text-white"));
    btn.classList.add("bg-slate-900","text-white");
    rebuildHeat();
    refreshTrend();
    refreshKPIs();
  });
});