// server/static/dashboard.js
const evt = new EventSource("/api/v1/dashboard/stream");

// --- Global trend chart
const ctxTrend = document.getElementById("chartTrend");
const trend = new Chart(ctxTrend, {
  type: "line",
  data: { labels: [], datasets: [
    { label: "Global Acc", data: [], yAxisID: "y1", tension: 0.25 },
    { label: "Global Loss", data: [], yAxisID: "y2", tension: 0.25 },
  ]},
  options: {
    responsive: true,
    interaction: { mode: "nearest", intersect: false },
    scales: {
      y1: { type: "linear", position: "left", min: 0, max: 1, grid: { color: "#e2e8f0" }, ticks:{ color:"#334155" } },
      y2: { type: "linear", position: "right", grid: { color: "#e2e8f0" }, ticks:{ color:"#334155" } },
      x:  { grid: { color: "#e2e8f0" }, ticks:{ color:"#334155" } },
    },
    plugins: { legend: { position: "bottom", labels:{ color:"#334155" } } },
  },
});

// --- Heatmap helpers (밝은 배경용 색상)
function makeHeatmapConfig(title, colorFn) {
  return {
    type: "matrix",
    data: { datasets: [{
      label: title, data: [], width: 28, height: 22,
      backgroundColor: ctx => colorFn((ctx.raw && ctx.raw.v) ?? 0),
    }]},
    options: {
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#475569" }, offset: true, title: { display: true, text: "Round", color:"#334155" }, grid:{ color:"#e2e8f0" } },
        y: { ticks: { color: "#475569" }, offset: true, title: { display: true, text: "Client", color:"#334155" }, grid:{ color:"#e2e8f0" } },
      },
      elements: { rectangle: { borderWidth: 1, borderColor: "#e2e8f0" } },
    },
  };
}

// 색상 맵핑: R은 파란톤, H는 초록톤
const colorR = v => `rgba(37,99,235, ${0.15 + 0.85*Math.max(0, Math.min(1, v))})`;
const colorH = v => `rgba(5,150,105, ${0.15 + 0.85*Math.max(0, Math.min(1, v))})`;

const heatR = new Chart(document.getElementById("heatR"), makeHeatmapConfig("R", colorR));
const heatH = new Chart(document.getElementById("heatH"), makeHeatmapConfig("H", colorH));

// --- Federated analysis
const analysis = new Chart(document.getElementById("chartAnalysis"), {
  type: "bar",
  data: { labels: [], datasets: [{ label: "weighted p(label)", data: [] }] },
  options: { plugins: { legend: { display: false } }, scales: { y: { min: 0, max: 1, grid:{ color:"#e2e8f0" }, ticks:{ color:"#334155" } }, x:{ ticks:{ color:"#334155" }, grid:{ color:"#e2e8f0" } } } },
});

// --- Poincaré disk
const canvas = document.getElementById("poincare");
const ctx = canvas.getContext("2d");
const center = { x: canvas.width / 2, y: canvas.height / 2 };
const radius = Math.min(center.x, center.y) - 6;
let netStats = {};

function drawDisk() {
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.beginPath();
  ctx.arc(center.x, center.y, radius, 0, Math.PI*2);
  ctx.strokeStyle = "#94a3b8";   // 밝은 경계
  ctx.lineWidth = 2;
  ctx.stroke();

  const cids = Object.keys(netStats).sort();
  cids.forEach((cid, idx) => {
    const rtt = netStats[cid].rtt_ms || 50;
    const d = Math.min(rtt/150, 1.6);
    const r = Math.tanh(d/2) * (radius*0.95);
    const theta = (2*Math.PI/cids.length)*idx;
    const x = center.x + r * Math.cos(theta);
    const y = center.y + r * Math.sin(theta);

    ctx.beginPath();
    ctx.arc(x, y, 10, 0, Math.PI*2);
    ctx.fillStyle = "#38bdf8";
    ctx.fill();
    ctx.fillStyle = "#334155";
    ctx.font = "12px sans-serif";
    ctx.fillText(cid, x+12, y+4);
  });
}

function logLine(html) {
  const el = document.getElementById("log");
  const div = document.createElement("div");
  div.innerHTML = html;
  el.prepend(div);
}

function pushHeat(chart, clientId, round, val) {
  const yIndex = clientIdToIdx(clientId);
  chart.data.datasets[0].data.push({ x: round, y: yIndex, v: val });
  chart.update("none");
}

const clientOrder = {};
function clientIdToIdx(cid) {
  if (!(cid in clientOrder)) clientOrder[cid] = Object.keys(clientOrder).length + 1;
  return clientOrder[cid];
}

// ===== SSE =====
evt.onmessage = (ev) => {
  const msg = JSON.parse(ev.data);
  const t = msg.type;
  const d = msg.data || {};

  if (t === "bootstrap") {
    (d.clients || []).forEach(c => { netStats[c.client_id] = { rtt_ms: c.net_latency_ms } });
    drawDisk();
    // server/static/dashboard.js  (bootstrap 처리 안에)
    (d.logs || []).forEach(x =>
      logLine(`<span class="text-sky-600">[ROUND ${x.round}]</span> acc=${(+x.acc).toFixed(3)}, loss=${(+x.loss).toFixed(4)}, participants=${x.participants}`)
    );


    if (d.trend) {
      trend.data.labels = d.trend.labels || [];
      trend.data.datasets[0].data = d.trend.acc || [];
      trend.data.datasets[1].data = d.trend.loss || [];
      trend.update("none");
    }

    if (d.heat && Array.isArray(d.heat.metrics)) {
      d.heat.metrics.forEach(m => {
        pushHeat(heatR, m.client_id, m.round_id, m.R);
        pushHeat(heatH, m.client_id, m.round_id, m.H);
      });
      heatR.update("none");
      heatH.update("none");
    }

    if (d.analysis_round) {
      const dist = d.analysis_round.weighted_label_dist || {};
      const labels = Object.keys(dist).sort();
      analysis.data.labels = labels;
      analysis.data.datasets[0].data = labels.map(k => dist[k]);
      analysis.update("none");
    }
    return;
  }

  if (t === "client_register") {
    netStats[d.client_id] = { rtt_ms: 50 };
    drawDisk();
    logLine(`<span class="text-emerald-600">[REGISTER]</span> ${d.client_id} (N_i=${d.n_i}, N_T=${d.N_T})`);
  }
  if (t === "metrics") {
    const rid = d.round ?? 0;
    const cid = d.client_id;
    pushHeat(heatR, cid, rid, d.R ?? 0);
    pushHeat(heatH, cid, rid, d.H ?? 0);
    if (d.comm && d.comm.rtt_ms != null) {
      netStats[cid] = { rtt_ms: d.comm.rtt_ms };
      drawDisk();
    }
  }
  if (t === "analysis_round") {
    const dist = d.weighted_label_dist || {};
    const labels = Object.keys(dist).sort();
    analysis.data.labels = labels;
    analysis.data.datasets[0].data = labels.map(k => dist[k]);
    analysis.update("none");
  }
  if (t === "round_summary") {
    const r = d.round, m = d.agg_metrics || {};
    trend.data.labels.push(`#${r}`);
    trend.data.datasets[0].data.push(m.acc ?? null);
    trend.data.datasets[1].data.push(m.loss ?? null);
    trend.update("none");
    logLine(`<span class="text-sky-600">[ROUND ${r}]</span> acc=${(m.acc??'-')}, loss=${(m.loss??'-')}, participants=${d.participants}, failures=${d.failures}`);
  }
  if (t === "model_update") {
    logLine(`<span class="text-fuchsia-600">[MODEL]</span> r${d.round} digest=${(d.model_digest||'').slice(0,8)}...`);
  }
};
