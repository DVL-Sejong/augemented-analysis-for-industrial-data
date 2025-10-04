// PATCH: restore alert stacked bar (ocsvm/kmeans/hmac/crosscheck/other)
(function(){
  if (!window.Chart) return;

  // create chart instance if not existing
  const el = document.getElementById("alertStack");
  if (!el) return;
  if (!window.__alertStackChart) {
    window.__alertStackChart = new Chart(el, {
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
  }

  // hook into refreshKPIs if provided
  const stk = window.__alertStackChart;
  const oldRefresh = window.refreshKPIs;
  window.refreshKPIs = function(){
    if (typeof oldRefresh === "function") oldRefresh();
    try {
      const state = window.state || (window.__state || {}); // be generous
      const pair = (typeof window.lastAttackDefendPair === "function") ? window.lastAttackDefendPair(state.currentScenario || "A") : null;
      if (!pair || !state.logs) {
        stk.data.datasets.forEach(ds => ds.data = [0,0]);
        stk.update("none");
        return;
      }
      const atkRounds = new Set(pair.attack.last5.map(r=>r.round_id));
      const defRounds = new Set(pair.defend.last5.map(r=>r.round_id));
      const counts = { atk:{ocsvm:0,kmeans:0,hmac:0,crosscheck:0,other:0}, def:{ocsvm:0,kmeans:0,hmac:0,crosscheck:0,other:0} };
      state.logs.forEach(L => {
        const src = (L.source || "").toLowerCase();
        const key = (src==="ocsvm"||src==="kmeans"||src==="hmac"||src==="crosscheck") ? src : "other";
        const rid = L.round_id;
        if (atkRounds.has(rid)) counts.atk[key]++;
        if (defRounds.has(rid)) counts.def[key]++;
      });
      const order = ["ocsvm","kmeans","hmac","crosscheck","other"];
      stk.data.datasets.forEach((ds, i) => {
        const k = order[i]; ds.data = [counts.atk[k], counts.def[k]];
      });
      stk.update("none");
    } catch(e) {
      // no-op
    }
  };
})();
