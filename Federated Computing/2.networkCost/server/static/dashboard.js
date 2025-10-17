// server/static/dashboard.js
const evt = new EventSource("/api/v1/dashboard/stream");

// ---------- utils ----------
const fmt2 = (x)=>Number(x||0).toFixed(2);
const byRoundAgg = {}; // round -> {bytes, cost, rtt, hits, cnt}
const lastPath = {};   // client_id -> latest row

function bumpAgg(round, comm) {
  const r = byRoundAgg[round] = byRoundAgg[round] || {bytes:0, cost:0, rtt:0, hits:0, cnt:0};
  r.bytes += Number(comm?.bytes_up ?? 0);
  r.cost  += Number(comm?.tier_cost ?? 0);
  r.rtt   += Number(comm?.tier_rtt_ms ?? 0);
  r.hits  += comm?.cache_hit ? 1 : 0;
  r.cnt   += 1;
}

// ---------- charts ----------
const ctxTrend   = document.getElementById("chartTrend");
const ctxNetRtt  = document.getElementById("chartNetRtt");
const ctxNetCost = document.getElementById("chartNetCost");
const ctxHit     = document.getElementById("chartCacheHit");
const ctxBytes   = document.getElementById("chartBytes");

const trend = new Chart(ctxTrend, {
  type: "line",
  data: { labels: [], datasets: [
    { label: "Global Acc",  data: [], yAxisID: "y1", tension: 0.25 },
    { label: "Global Loss", data: [], yAxisID: "y2", tension: 0.25 },
  ]},
  options: {
    responsive: true,
    interaction: { mode: "nearest", intersect: false },
    scales: {
      y1:{ type:"linear", position:"left", min:0, max:1 },
      y2:{ type:"linear", position:"right" }
    }
  }
});

const netRtt = new Chart(ctxNetRtt, {
  type: "line",
  data: { labels: [], datasets: [{ label:"Avg Tier RTT (ms)", data: [], tension:0.25 }]},
  options: { responsive:true }
});
const netCost = new Chart(ctxNetCost, {
  type: "line",
  data: { labels: [], datasets: [{ label:"Sum Network Cost", data: [], tension:0.25 }]},
  options: { responsive:true }
});
const hitRate = new Chart(ctxHit, {
  type: "line",
  data: { labels: [], datasets: [{ label:"Cache Hit Rate (%)", data: [], tension:0.25 }]},
  options: { responsive:true, scales:{ y:{ min:0, max:100 } } }
});
const bytesUp = new Chart(ctxBytes, {
  type: "bar",
  data: { labels: [], datasets: [{ label:"Bytes Up (sum)", data: [] }]},
  options: { responsive:true }
});

// ---------- path table ----------
const tbody = document.getElementById("tblPathsBody");
function renderPathTable() {
  tbody.innerHTML = "";
  Object.values(lastPath)
    .sort((a,b)=>String(a.client_id).localeCompare(String(b.client_id)))
    .forEach(x=>{
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${x.client_id}</td>
        <td>${x.round ?? "-"}</td>
        <td>${x.tier_path || "-"}</td>
        <td>${fmt2(x.tier_rtt_ms ?? 0)}</td>
        <td>${fmt2(x.tier_cost   ?? 0)}</td>
        <td>${x.cache_hit ? "HIT":"MISS"}</td>
        <td>${x.bytes_after_preproc ?? 0}</td>
      `;
      tbody.appendChild(tr);
    });
}

// ---------- handlers ----------
function applyMetric(m) {
  if (!m) return;
  const comm = m.comm || {};
  const rnd  = Number(m.round ?? m.round_id ?? 0);

  // latest path row
  lastPath[m.client_id] = {
    client_id: m.client_id, round: rnd,
    tier_path: comm.tier_path,
    tier_rtt_ms: Number(comm.tier_rtt_ms ?? m.rtt_ms ?? 0),
    tier_cost: Number(comm.tier_cost ?? 0),
    cache_hit: !!comm.cache_hit,
    bytes_after_preproc: Number(comm.bytes_after_preproc ?? 0),
  };
  renderPathTable();

  // round aggregates
  if (rnd > 0) {
    bumpAgg(rnd, {
      bytes_up: Number(comm.bytes_up ?? m.bytes_up ?? 0),
      tier_cost: Number(comm.tier_cost ?? 0),
      tier_rtt_ms: Number(comm.tier_rtt_ms ?? m.rtt_ms ?? 0),
      cache_hit: !!comm.cache_hit,
    });

    const lbl = `R${rnd}`;
    if (!netRtt.data.labels.includes(lbl)) {
      netRtt.data.labels.push(lbl);
      netCost.data.labels.push(lbl);
      hitRate.data.labels.push(lbl);
      bytesUp.data.labels.push(lbl);
    }
    const agg = byRoundAgg[rnd];
    const avgRtt = agg.cnt ? agg.rtt/agg.cnt : 0;
    const hitPct = agg.cnt ? (agg.hits/agg.cnt)*100.0 : 0;

    netRtt.data.datasets[0].data[netRtt.data.labels.indexOf(lbl)]  = avgRtt;
    netCost.data.datasets[0].data[netCost.data.labels.indexOf(lbl)] = agg.cost;
    hitRate.data.datasets[0].data[hitRate.data.labels.indexOf(lbl)] = hitPct;
    bytesUp.data.datasets[0].data[bytesUp.data.labels.indexOf(lbl)] = agg.bytes;

    netRtt.update("none"); netCost.update("none"); hitRate.update("none"); bytesUp.update("none");
  }
}

function applyRoundSummary(rs) {
  if (!rs) return;
  const r = Number(rs.round || rs.round_id || 0);
  const m = rs.agg_metrics || {};
  const lbl = `R${r}`;
  const i = trend.data.labels.indexOf(lbl);
  if (i === -1) {
    trend.data.labels.push(lbl);
    trend.data.datasets[0].data.push(m.acc ?? null);
    trend.data.datasets[1].data.push(m.loss ?? null);
  } else {
    // 중복/갱신 안전
    trend.data.datasets[0].data[i] = m.acc ?? trend.data.datasets[0].data[i];
    trend.data.datasets[1].data[i] = m.loss ?? trend.data.datasets[1].data[i];
  }
  trend.update("none");
}

// ---------- SSE ----------
evt.onmessage = (e)=>{
  try {
    const msg = JSON.parse(e.data);

    // 초기 히스토리
    if (msg.init) {
      const hist = msg.init.history || {};
      (hist.rounds || []).forEach(r=>{
        applyRoundSummary({ round: r.round, agg_metrics: { acc: r.global_acc, loss: r.global_loss } });
      });
      (hist.metrics || []).forEach(applyMetric);
      return;
    }

    const t = msg.type || msg.data?.type;
    const d = msg.data || msg;

    if (t === "metrics") {
      applyMetric(d);
      return;
    }
    if (t === "round_summary") {
      applyRoundSummary(d);
      return;
    }
    if (t === "local_analysis") {
      // 현재는 별도 차트 없음. 필요 시 추가.
      return;
    }
  } catch(err) {
    console.error("SSE parse error", err);
  }
};
