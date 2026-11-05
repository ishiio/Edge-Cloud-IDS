"use strict";
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const f1 = (x, d = 1) => (x == null || isNaN(x)) ? "–" : Number(x).toFixed(d);
const NS = "http://www.w3.org/2000/svg";
let S = null;            // last /api/state
let busy = false;
let offlineMode = location.protocol === "file:";
function savedResponse(path) {
  const u = new URL(path, "http://localhost"), data = window.IDS_SNAPSHOT;
  if (u.pathname === "/api/state") return structuredClone(data.state);
  if (u.pathname === "/api/benchmark") return data.benchmark || {available:false};
  if (u.pathname === "/api/audit") return data.audit || {available:false};
  if (u.pathname.startsWith("/api/results/")) return data.results[u.pathname.split("/").pop()];
  if (u.pathname === "/api/schema") return data.schemas[S?.config.dataset || data.state.config.dataset];
  if (u.pathname === "/api/curve") return data.curves[u.searchParams.get("dataset") + ":" + u.searchParams.get("delta_ms")];
  throw new Error("Start the Python server to use this feature.");
}
function updateConnection() {
  $("#connectionNotice").hidden = !offlineMode;
  $("#modeLabel").textContent = offlineMode ? "Saved evaluation · real data" : "Real datasets · seed 0 models";
  $$(".controls input, .controls select, .controls button, #fCsv, #bClassify, #bSample").forEach(el => {el.disabled = offlineMode; el.title = offlineMode ? "Start the Python server for live inference" : "";});
}

async function api(path, opt = {}) {
  if (offlineMode) {
    if (opt.body !== undefined) throw new Error("Start the Python server for live replay and classification.");
    return savedResponse(path);
  }
  const o = { signal: AbortSignal.timeout(8000), method: opt.body !== undefined ? "POST" : "GET", ...opt };
  if (opt.body !== undefined && !(opt.body instanceof FormData)) { o.headers = { "content-type": "application/json" }; o.body = JSON.stringify(opt.body); }
  const r = await fetch(path, o);
  if (!r.ok) { let m = r.statusText; try { const j = await r.json(); m = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch {} throw new Error(m); }
  return r.json();
}
const pill = (txt, cls) => { const p = $("#statusPill"); p.className = "pill " + (cls || ""); $("span", p).textContent = txt; };

/* ---------- tiny SVG chart helpers ---------- */
function svg(el, html) { el.innerHTML = html; }
function scale(d0, d1, r0, r1) { return v => r0 + (v - d0) * (r1 - r0) / ((d1 - d0) || 1); }
function ticks(a, b, n = 5) { const out = [], st = (b - a) / n; for (let i = 0; i <= n; i++) out.push(a + st * i); return out; }
const tip = (() => { const t = document.createElement("div"); t.className = "tip"; document.body.appendChild(t);
  return { show(e, html) { t.innerHTML = html; t.style.display = "block"; t.style.left = Math.min(e.clientX + 14, innerWidth - 220) + "px"; t.style.top = (e.clientY + 14) + "px"; }, hide() { t.style.display = "none"; } }; })();
document.addEventListener("mouseover", e => { const t = e.target.closest("[data-tip]"); t ? tip.show(e, t.dataset.tip) : tip.hide(); });
document.addEventListener("mousemove", e => { if (e.target.closest("[data-tip]")) tip.show(e, e.target.closest("[data-tip]").dataset.tip); });

/* ---------- tabs ---------- */
$$("#tabs button").forEach(b => b.onclick = () => {
  $$("#tabs button").forEach(x => x.classList.toggle("on", x === b));
  $$(".tab").forEach(t => t.classList.toggle("on", t.id === "tab-" + b.dataset.tab));
  history.replaceState(null, "", "#" + b.dataset.tab);
  if (b.dataset.tab === "lab") loadLab().catch(e => toast(e.message)); if (b.dataset.tab === "bench") loadBench().catch(e => toast(e.message)); if (b.dataset.tab === "arch") renderArch();
});

/* ---------- controls ---------- */
const scenOpts = () => ["natural", "benign", "attack", ...(S?.datasets[S.config.dataset].classes || [])];
function syncControls() {
  const c = S.config;
  const ds = $("#cDataset");
  if (!ds.options.length) Object.entries(S.datasets).forEach(([k, v]) => ds.add(new Option(v.title + (v.source === "synthetic" ? " (synthetic)" : ""), k)));
  const ls = $("#labDataset");
  if (!ls.options.length) Object.entries(S.datasets).forEach(([k, v]) => ls.add(new Option(v.title, k)));
  ds.value = c.dataset;
  const sc = $("#cScenario"); const want = scenOpts().join();
  if (sc.dataset.k !== want) { const cur = S.stream.scenario; sc.innerHTML = ""; scenOpts().forEach(s => sc.add(new Option(s === "natural" ? "natural mix" : s, s))); sc.dataset.k = want; sc.value = scenOpts().includes(cur) ? cur : "natural"; }
  $$("#cPolicy button").forEach(b => b.classList.toggle("on", b.dataset.v === c.policy));
  $("#tauVal").textContent = c.tau.toFixed(3) + (c.tau_is_auto ? " (τ*)" : "");
  if (document.activeElement !== $("#cTau")) $("#cTau").value = c.tau;
  $("#cAuto").checked = c.tau_is_auto;
  $("#deltaVal").textContent = c.delta_ms + " ms";
  if (document.activeElement !== $("#cDelta")) $("#cDelta").value = c.delta_ms;
  $("#rateVal").textContent = $("#cRate").value + "/s";
  const sb = $("#bStream"); sb.textContent = S.stream.running ? "Stop stream" : "Start stream"; sb.classList.toggle("stop", S.stream.running);
  const b = $("#banner");
  const syn = Object.values(S.datasets).some(d => d.source === "synthetic");
  b.hidden = false;
  if (!syn) b.innerHTML = "<b>Real data.</b> Deployed models: seed 0 · Benchmark page: five seeds. Independent evaluation using the included CSVs. Live mode replays held-out flows; WAN delay and INT8 are simulated.";
  if (syn) b.innerHTML = "<b>Synthetic data.</b> Edge-IIoTset and TON_IoT could not be downloaded here, so flows come from a built-in generator that mimics their structure (class imbalance, heavy-tailed counters, overlapping attack families, identifier columns). Absolute scores differ from the paper. Put the real CSVs in <code>data/</code> and retrain to use them (see README).";
}
async function pushConfig(body) { try { S = await api("/api/config", { body }); syncControls(); render(); } catch (e) { toast(e.message); } }
function toast(m) { const el = $("#toast"); el.textContent = m; el.hidden = false; clearTimeout(window.toastTimer); window.toastTimer = setTimeout(() => el.hidden = true, 5000); }

$("#cDataset").onchange = e => pushConfig({ dataset: e.target.value });
$$("#cPolicy button").forEach(b => b.onclick = () => pushConfig({ policy: b.dataset.v }));
let tauT; $("#cTau").oninput = e => { $("#tauVal").textContent = (+e.target.value).toFixed(3); $("#cAuto").checked = false; clearTimeout(tauT); tauT = setTimeout(() => pushConfig({ tau: +e.target.value }), 120); };
$("#cAuto").onchange = e => e.target.checked ? pushConfig({ auto_tau: true }) : pushConfig({ tau: S.config.tau });
let dT; $("#cDelta").oninput = e => { $("#deltaVal").textContent = e.target.value + " ms"; clearTimeout(dT); dT = setTimeout(() => pushConfig({ delta_ms: +e.target.value }), 120); };
$("#cRate").oninput = e => { $("#rateVal").textContent = e.target.value + "/s"; if (S?.stream.running) api("/api/stream", { body: { running: true, rate: +e.target.value } }); };
$("#cScenario").onchange = e => api("/api/stream", { body: { running: !!S?.stream.running, scenario: e.target.value, rate: +$("#cRate").value } }).catch(x => toast(x.message));
$("#bStream").onclick = async () => { try { await api("/api/stream", { body: { running: !S.stream.running, rate: +$("#cRate").value, scenario: $("#cScenario").value } }); poll(); } catch (e) { toast(e.message); } };
$("#bBurst").onclick = async () => { await api("/api/burst?seconds=8", { body: {} }); poll(); };
$("#bReset").onclick = async () => { await api("/api/reset", { body: {} }); poll(); };

/* ---------- monitor rendering ---------- */
function kpi(cls, label, val, sub) { return `<div class="kpi ${cls}"><span>${label}</span><b>${val}</b><small>${sub}</small></div>`; }
function render() {
  if (!S) return;
  const t = S.totals;
  $("#kpis").innerHTML =
    kpi("", "Flows processed", t.total.toLocaleString(), `${t.edge_decided.toLocaleString()} decided at edge`) +
    kpi("cl", "Forwarded to cloud", f1(t.forwarded_pct) + "%", `${t.forwarded.toLocaleString()} flows · break-even &lt; ${f1(t.breakeven_fraction_pct, 2)}%`) +
    kpi("good", "Cloud traffic saved", f1(t.traffic_reduction_pct) + "%", `${(t.bytes_forwarded / 1024).toFixed(1)} KB sent vs ${(t.bytes_cloud_only / 1024).toFixed(1)} KB`) +
    kpi("good", "Mean latency", f1(t.mean_latency_ms, 2) + " ms", `Cloud-Only ${f1(t.cloud_only_latency_ms, 2)} ms · −${f1(t.latency_reduction_pct)}%`) +
    kpi("alert", "Alerts", t.alerts.toLocaleString(), t.total ? f1(100 * t.alerts / t.total) + "% of flows" : "–") +
    kpi("", "Accuracy vs ground truth", t.accuracy_pct == null ? "–" : f1(t.accuracy_pct) + "%", t.edge_only_accuracy_pct == null ? "replayed test flows" : `edge alone ${f1(t.edge_only_accuracy_pct)}%`);
  const live = S.stream.running;
  pill(offlineMode ? "saved snapshot" : S.stream.burst ? "attack burst" : live ? `streaming · ${S.cloud_tier.split(" ")[0]} cloud` : "server connected", S.stream.burst ? "bad" : live ? "live" : "");
  drawScatter(); drawBars(); drawAlerts(); drawMix();
}
function drawScatter() {
  const W = 720, H = 260, m = { l: 40, r: 10, t: 8, b: 22 }, R = S.recent, n = 250;
  const x = scale(0, n - 1, m.l, W - m.r), y = scale(0, 1, H - m.b, m.t);
  let h = `<g class="axis">`;
  for (const v of [0, .25, .5, .75, 1]) h += `<line class="grid-l" x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}"/><text x="${m.l - 6}" y="${y(v) + 4}" text-anchor="end">${v}</text>`;
  h += `</g><line x1="${m.l}" x2="${W - m.r}" y1="${y(S.config.tau)}" y2="${y(S.config.tau)}" stroke="var(--mute)" stroke-dasharray="6 4"/><text x="${W - m.r - 4}" y="${y(S.config.tau) - 5}" text-anchor="end" fill="var(--mute)" font-size="11">τ = ${S.config.tau.toFixed(2)}</text>`;
  if (!R.length) h += `<text x="${W / 2}" y="${H / 2}" text-anchor="middle" fill="var(--mute)">Start the stream or send flows to see gate decisions</text>`;
  const off = n - R.length;
  R.forEach((r, i) => {
    const cx = x(off + i), cy = y(r.conf);
    h += `<circle cx="${cx}" cy="${cy}" r="${r.truth && r.truth !== r.label ? 4 : 3}" fill="${r.fwd ? "var(--cloud)" : "var(--edge)"}" ${r.truth && r.truth !== r.label ? 'stroke="var(--alert)" stroke-width="1.8"' : 'opacity=".8"'} data-tip="#${r.id} · ${esc(r.label)}<br>c = ${r.conf.toFixed(3)} · ${r.fwd ? "→ cloud" : "edge"}${r.truth ? "<br>truth: " + esc(r.truth) : ""}"/>`;
  });
  svg($("#scatter"), h);
}
function bar(label, parts, txt) { return `<div class="barrow"><span>${label}</span><div class="bar">${parts}</div><span>${txt}</span></div>`; }
function drawBars() {
  const t = S.totals, mx = Math.max(t.cloud_only_latency_ms, t.mean_latency_ms, 1e-6);
  $("#latBars").innerHTML = bar("Cloud-Only", `<i class="c" style="width:100%"></i>`, f1(t.cloud_only_latency_ms, 1) + " ms") +
    bar("Edge-Cloud", `<i class="e" style="width:${100 * t.mean_latency_ms / mx}%"></i>`, f1(t.mean_latency_ms, 1) + " ms");
  $("#trafBar").innerHTML = bar("Cloud-Only", `<i class="c" style="width:100%"></i>`, "100%") +
    bar("Edge-Cloud", `<i class="c" style="width:${t.forwarded_pct}%"></i>`, f1(t.forwarded_pct) + "%");
}
function drawAlerts() {
  const tb = $("#alerts tbody");
  tb.innerHTML = S.alerts.length ? S.alerts.slice(0, 25).map(a => `<tr><td>${a.id}</td><td><span class="tag bad">${esc(a.label)}</span></td><td>${a.edge_label === a.label ? "same" : esc(a.edge_label)}</td><td class="num">${a.confidence.toFixed(2)}</td><td><span class="tag ${a.decided_by}">${a.decided_by}</span></td><td>${a.truth ? `<span class="tag ${a.correct ? "ok" : "bad"}">${esc(a.truth)}</span>` : "–"}</td><td class="num">${a.latency_ms.toFixed(2)}</td></tr>`).join("")
    : `<tr><td colspan="7" class="empty">No alerts yet.</td></tr>`;
}
function drawMix() {
  const e = Object.entries(S.by_label).sort((a, b) => b[1] - a[1]), tot = Math.max(S.totals.total, 1), mx = e.length ? e[0][1] : 1;
  const benign = S.datasets[S.config.dataset].benign;
  $("#mix").innerHTML = e.length ? e.map(([k, v]) => { const fw = S.fwd_by_label[k] || 0;
    return bar(`<span style="color:${k === benign ? "var(--ok)" : "var(--text)"}">${esc(k)}</span>`, `<i class="e" style="width:${100 * (v - fw) / mx}%"></i><i class="h" style="width:${100 * fw / mx}%"></i>`, f1(100 * v / tot, 0) + "%"); }).join("")
    : `<div class="empty">No traffic yet.</div>`;
}

/* ---------- polling ---------- */
async function poll() {
  if (busy) return; busy = true;
  try { S = await api("/api/state"); updateConnection(); syncControls(); render(); if (S.last_error) toast(S.last_error); }
  catch (e) { offlineMode = true; S = savedResponse("/api/state"); updateConnection(); syncControls(); render(); toast("Live API unavailable. Showing the included dataset snapshot."); }
  finally { busy = false; }
}
setInterval(() => { if (!offlineMode) poll(); }, 1000);

/* ---------- threshold lab ---------- */
let L = null;
async function loadLab() {
  if (!S) await poll();
  const ds = $("#labDataset"); if (!ds.value) ds.value = S.config.dataset;
  try { L = await api(`/api/curve?dataset=${ds.value}&delta_ms=${$("#labDelta").value}`); } catch (e) { $("#labNote").textContent = e.message; return; }
  drawLab();
}
$("#labZoom").onchange = () => L && drawLab(); $("#labDataset").onchange = loadLab; $("#labDelta").onchange = loadLab;
function lineChart(el, o) {
  const W = 560, H = 330, m = { l: 52, r: 16, t: 14, b: 40 };
  const x = scale(o.x0, o.x1, m.l, W - m.r), y = scale(o.y0, o.y1, H - m.b, m.t);
  let h = `<g class="axis">`;
  ticks(o.y0, o.y1, 5).forEach(v => h += `<line class="grid-l" x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}"/><text x="${m.l - 7}" y="${y(v) + 4}" text-anchor="end">${v.toFixed(o.yd ?? 0)}</text>`);
  ticks(o.x0, o.x1, 5).forEach(v => h += `<text x="${x(v)}" y="${H - m.b + 16}" text-anchor="middle">${v.toFixed(o.xd ?? 0)}</text>`);
  h += `<text x="${(m.l + W - m.r) / 2}" y="${H - 6}" text-anchor="middle">${o.xl}</text><text transform="translate(13 ${(m.t + H - m.b) / 2}) rotate(-90)" text-anchor="middle">${o.yl}</text></g>`;
  (o.hlines || []).forEach(l => h += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(l.v)}" y2="${y(l.v)}" stroke="${l.c}" stroke-dasharray="${l.d || "5 4"}"/><text x="${W - m.r - 3}" y="${y(l.v) - 4}" text-anchor="end" fill="${l.c}" font-size="11">${l.t}</text>`);
  o.series.forEach(s => {
    h += `<polyline fill="none" stroke="${s.c}" stroke-width="2.2" points="${s.pts.map(p => x(p[0]).toFixed(1) + "," + y(p[1]).toFixed(1)).join(" ")}"/>`;
    s.pts.forEach((p, i) => h += `<circle cx="${x(p[0])}" cy="${y(p[1])}" r="${s.pts.length > 14 ? 2.4 : 3.4}" fill="${s.c}" data-tip="${s.n}<br>${s.tips ? s.tips[i] : p[0].toFixed(1) + " , " + p[1].toFixed(2)}"/>`);
    if (s.mark != null) { const p = s.pts[s.mark]; h += `<circle cx="${x(p[0])}" cy="${y(p[1])}" r="7" fill="none" stroke="${s.c}" stroke-width="2"/>`; }
  });
  svg(el, h);
}
function nearest(arr, tau) { let b = 0; arr.forEach((p, i) => { if (Math.abs(p.tau - tau) < Math.abs(arr[b].tau - tau)) b = i; }); return b; }
function drawLab() {
  const d = S.datasets[L.dataset], le = d.le_ms, lc = d.lc_ms, A = L.curves.A, B = L.curves.B;
  const pol = S.config.policy, cur = S.config.dataset === L.dataset ? S.config.tau : L.tau_star[pol];
  const tips = a => a.map(p => `τ = ${p.tau}<br>F1 ${p.f1.toFixed(2)} · fwd ${p.fwd.toFixed(1)}%`);
  const lo = Math.min(L.edge_only_f1, ...A.map(p => p.f1), ...B.map(p => p.f1)), hi = Math.max(L.cloud_only_f1, ...A.map(p => p.f1)) + 0.5;
  const zoomMax = $("#labZoom")?.checked ? Math.min(100, Math.ceil(Math.max(...A.map(p => p.fwd)) / 5) * 5 + 5) : 100;
  lineChart($("#chF1"), { x0: 0, x1: zoomMax, y0: Math.floor(lo - 1), y1: Math.ceil(hi), xl: "flows forwarded to cloud (%)", yl: "macro-F1 (%)", yd: 1,
    hlines: [{ v: L.cloud_only_f1, c: "var(--cloud)", t: "Cloud-Only" }, { v: L.edge_only_f1, c: "var(--mute)", t: "Edge-Only (INT8 per-ch.)" }],
    series: [{ n: "Policy A", c: "var(--edge)", pts: A.map(p => [p.fwd, p.f1]), tips: tips(A), mark: pol === "A" ? nearest(A, cur) : null },
             { n: "Policy B", c: "var(--b)", pts: B.map(p => [p.fwd, p.f1]), tips: tips(B), mark: pol === "B" ? nearest(B, cur) : null }] });
  const lat = (p, dl) => le + p.fwd / 100 * (dl + lc), cols = { 20: "var(--accent)", 50: "var(--edge)", 100: "var(--cloud)" };
  const ds3 = [20, 50, 100], tmax = Math.max(...A.map(p => p.tau));
  lineChart($("#chLat"), { x0: 0.3, x1: 1, y0: 0, y1: 110, xl: "threshold τ  (Policy A)", yl: "mean latency per flow (ms)", xd: 1,
    hlines: ds3.map(dl => ({ v: dl + lc, c: cols[dl], t: `Cloud-Only Δ=${dl}`, d: "2 4" })),
    series: ds3.map(dl => ({ n: `Δ = ${dl} ms`, c: cols[dl], pts: A.map(p => [p.tau, lat(p, dl)]), tips: A.map(p => `τ = ${p.tau}<br>${lat(p, dl).toFixed(1)} ms vs ${(dl + lc).toFixed(1)} ms Cloud-Only`) })) });
  $("#labNote").innerHTML = `Break-even (Eq. 5): the gate beats Cloud-Only whenever f &lt; <b>${f1(L.breakeven_fraction_pct, 2)}%</b> at Δ = ${L.delta_ms} ms. τ* (validation): A = <b>${L.tau_star.A}</b>, B = <b>${L.tau_star.B}</b>.`;
  $("#labTable tbody").innerHTML = A.map((a, i) => { const b = B[i], sel = Math.abs(a.tau - cur) < 1e-9;
    return `<tr class="click ${sel ? "sel" : ""}" data-tau="${a.tau}"><td>${a.tau}</td><td class="num">${a.f1.toFixed(2)}</td><td class="num">${a.fwd.toFixed(1)}</td><td class="num">${a.latency_ms.toFixed(1)}</td><td class="num">${b.f1.toFixed(2)}</td><td class="num">${b.fwd.toFixed(1)}</td><td class="num">${b.latency_ms.toFixed(1)}</td></tr>`; }).join("");
  $$("#labTable tbody tr").forEach(r => r.onclick = async () => { if (offlineMode) { toast("Start the server to apply a live operating point."); return; } await pushConfig({ dataset: L.dataset, tau: +r.dataset.tau, delta_ms: +$("#labDelta").value }); loadLab(); });
}

/* ---------- benchmark ---------- */
const ms = (o, d = 2) => o == null ? "–" : (typeof o === "number" ? f1(o, d) : `${f1(o.mean, d)} <span class="muted">± ${f1(o.std, d)}</span>`);
async function loadBench() {
  const el = $("#benchBody"); el.innerHTML = `<div class="empty">Loading…</div>`;
  const bm = await api("/api/benchmark"), au = await api("/api/audit").catch(() => ({ available: false })), keys = Object.keys(S.datasets);
  const res = {}; for (const k of keys) res[k] = await api("/api/results/" + k);
  let h = "";
  if (!bm.available) h += `<div class="callout"><b>Single-seed results.</b> Run <code>python -m ids.train --benchmark --seeds 5</code> to generate real-data mean ± std tables across five seeds.</div>`;
  else h += `<div class="callout"><b>Five-seed study</b> (seeds ${(bm[keys[0]]?.seeds || []).join(", ")}), generated ${esc(bm.generated)}. Test-set mean ± std; Δ = 50 ms simulated; τ* chosen on validation only.</div>`;
  if (au.available) h += `<div class="card"><h3>Real-data integrity audit <small>seed-0 split · scripts/audit_real_data.py</small></h3>` + keys.filter(k => au[k]).map(k => { const a = au[k], n = a.metrics_novel_only;
    return `<p class="note" style="margin:8px 0 0"><b>${esc(res[k].title)}</b></p><div class="audit">
      <div class="${a.rows_match_paper ? "ok" : ""}"><span>De-duplicated rows</span><b>${a.unique_rows.toLocaleString()}</b><small>${a.rows_match_paper ? "matches" : "differs from"} paper (${a.paper_unique_rows.toLocaleString()}) · ${(a.raw_rows - a.unique_rows).toLocaleString()} duplicates removed</small></div>
      <div class="${a.label_columns_in_features.length ? "" : "ok"}"><span>Target leakage</span><b>${a.label_columns_in_features.length ? "FOUND" : "none"}</b><small>label columns excluded · split disjoint: ${a.split_disjoint}</small></div>
      <div><span>Test flows also in train</span><b>${f1(a.test_flows_with_identical_feature_vector_in_train_pct)}%</b><small>identical model-input vector; counted as exact duplicates</small></div>
      <div><span>Policy A F1 · all vs unseen</span><b>${f1(a.metrics_all_test.policy_a.f1)} / ${n ? f1(n.policy_a.f1) : "–"}</b><small>unseen = test flows with no identical train vector (${a.n_novel_test.toLocaleString()})</small></div>
      <div><span>Best single feature</span><b>${f1(a.top_single_feature_macro_f1[0].f1)}%</b><small>macro-F1 of ${esc(a.top_single_feature_macro_f1[0].feature)} alone (majority class: ${f1(a.majority_class_baseline_acc)}% acc)</small></div></div>`; }).join("") + `</div>`;
  for (const k of keys) {
    const r = res[k], b = bm.available ? bm[k] : null, src = b || r, M = src.methods;
    const order = ["edge_fp32", "edge_int8_pt", "edge_int8", "cloud_only", "policy_a", "policy_b"].filter(o => M[o]);
    const tauOf = p => b ? b.tau_star[p].mean : r.selection[p].tau;
    h += `<div class="sec"><h2>${esc(r.title)} <small class="muted">· ${r.n_flows.toLocaleString()} flows · ${r.n_classes} classes · ${r.n_edge_features} edge / ${r.n_full_features} cloud features · ${esc(r.source)} data</small></h2>
    <div class="card"><h3>Main results <small>Table II</small></h3><div class="tablewrap"><table><thead><tr><th>Method</th><th class="num">Acc %</th><th class="num">Macro-F1 %</th><th class="num">Prec %</th><th class="num">Recall %</th><th class="num">Fwd %</th><th class="num">Latency ms</th><th class="num">Size KB</th></tr></thead><tbody>` +
      order.map(o => { const m = M[o]; const nm = esc(m.method) + (o.startsWith("policy") ? ` <span class="muted">(τ*=${tauOf(o.slice(-1).toUpperCase()).toFixed(2)})</span>` : "");
        return `<tr><td>${nm}</td><td class="num">${ms(m.acc)}</td><td class="num"><b>${ms(m.f1)}</b></td><td class="num">${ms(m.precision)}</td><td class="num">${ms(m.recall)}</td><td class="num">${ms(m.fwd, 1)}</td><td class="num">${ms(m.latency_ms)}</td><td class="num">${m.size_kb == null ? "–" : ms(m.size_kb, 1)}</td></tr>`; }).join("") +
      `</tbody></table></div>${b ? `<p class="note">Policy B − Policy A macro-F1: ${ms(b.policy_b_minus_a_f1)} points (paired over seeds).</p>` : ""}</div>`;
    const sw = src.sweeps, g = x => b ? x : x;
    h += `<div class="grid2"><div class="card"><h3>Effect of τ <small>Table III · macro-F1 % / share forwarded %</small></h3><div class="tablewrap"><table><thead><tr><th>τ</th><th class="num">F1 (A)</th><th class="num">Fwd (A)</th><th class="num">F1 (B)</th><th class="num">Fwd (B)</th></tr></thead><tbody>` +
      sw.A.map((a, i) => { const bb = sw.B[i]; const v = x => b ? f1(x.mean, 2) : f1(x, 2); return `<tr><td>${a.tau}</td><td class="num">${v(a.f1)}</td><td class="num">${b ? f1(a.fwd.mean, 1) : f1(a.fwd, 1)}</td><td class="num">${v(bb.f1)}</td><td class="num">${b ? f1(bb.fwd.mean, 1) : f1(bb.fwd, 1)}</td></tr>`; }).join("") +
      `</tbody></table></div></div><div class="card"><h3>Per-class recall <small>% · test set, seed 0</small></h3>` +
      Object.keys(r.per_class_recall.cloud_only).map(c => { const e = r.per_class_recall.edge_int8[c], p = r.per_class_recall.policy_a[c], cl = r.per_class_recall.cloud_only[c];
        return `<div class="barrow" data-tip="${esc(c)} (n=${r.test_class_counts[c]})<br>edge ${e.toFixed(1)} · A ${p.toFixed(1)} · cloud ${cl.toFixed(1)}"><span>${esc(c)}</span><div class="bar" style="height:10px"><i class="e" style="width:${p}%"></i></div><span>${p.toFixed(0)}</span></div><div class="barrow" style="margin-top:-4px;grid-template-columns:130px 1fr 60px"><span></span><div class="bar" style="height:4px;opacity:.6"><i style="width:${e}%;background:var(--mute)"></i></div><span></span></div>`; }).join("") +
      `<p class="note">Teal: Policy A final decision · grey: edge INT8 alone. Rare classes can hide behind aggregate scores.</p></div></div></div>`;
  }
  el.innerHTML = h;
}

/* ---------- analyze ---------- */
function showResults(res, summary) {
  $("#anaSummary").textContent = summary || "";
  $("#schemaOut").hidden = true; $(".tablewrap", $("#tab-analyze")).hidden = false;
  $("#anaTable tbody").innerHTML = res.map((r, i) => `<tr><td>${i + 1}</td><td><span class="tag ${r.alert ? "bad" : "ok"}">${esc(r.label)}</span></td><td>${esc(r.edge_label)}</td><td class="num">${r.confidence.toFixed(3)}</td><td><span class="tag ${r.decided_by}">${r.decided_by}</span></td><td>${r.truth ? esc(r.truth) : "–"}</td><td>${r.truth ? (r.correct ? "✓" : "✗") : ""}</td></tr>`).join("");
}
$("#bSample").onclick = async () => { try { const s = await api("/api/sample?n=5&scenario=natural"); $("#jsonIn").value = JSON.stringify(s.flows.map(f => f.features), null, 1); $("#anaMsg").textContent = "Truth labels: " + s.flows.map(f => f.truth).join(", "); } catch (e) { $("#anaMsg").textContent = e.message; } };
$("#bSchema").onclick = async () => { const s = await api("/api/schema"); const p = $("#schemaOut"); p.hidden = false; p.textContent = `${s.dataset}: ${s.input_columns.length} input columns\nedge features: ${s.edge_features.join(", ")}\n\ncolumns: ${s.input_columns.join(", ")}`; };
$("#bClassify").onclick = async () => {
  let recs; try { recs = JSON.parse($("#jsonIn").value); if (!Array.isArray(recs)) recs = [recs]; } catch { $("#anaMsg").textContent = "Invalid JSON"; return; }
  try { const r = await api("/api/classify", { body: { records: recs } }); showResults(r.results, `${r.results.length} flows · ${r.results.filter(x => x.forwarded).length} forwarded · policy ${r.policy}, τ=${r.tau}`); $("#anaMsg").textContent = ""; poll(); } catch (e) { $("#anaMsg").textContent = e.message; }
};
$("#fCsv").onchange = async e => {
  const f = e.target.files[0]; if (!f) return; const fd = new FormData(); fd.append("file", f);
  $("#anaMsg").textContent = "Classifying…";
  try { const r = await api("/api/classify-csv", { body: fd });
    showResults(r.results, `${r.n} flows · ${r.forwarded_pct.toFixed(1)}% forwarded · ${r.alerts} alerts${r.accuracy != null ? " · accuracy " + r.accuracy.toFixed(1) + "%" : ""}${r.n > 500 ? " (first 500 shown)" : ""}`);
    $("#anaMsg").textContent = `${r.matched_columns}/${r.expected_columns} schema columns matched`; } catch (x) { $("#anaMsg").textContent = x.message; }
  e.target.value = "";
};

/* ---------- architecture ---------- */
function renderArch() {
  if (!S) return; const d = S.datasets[S.config.dataset];
  $("#archBody").innerHTML = `
  <div class="card"><h2>Four-layer pipeline</h2>
   <div class="flow">
    <div class="layer"><b>1 · Device layer</b><small>Sensors and actuators generate network flows. Here: recorded flows replayed from the test split.</small></div>
    <div class="layer e"><b>2 · Edge gateway</b><small>Drops identifiers, scales with training statistics, selects ${d.n_edge} features, runs the ${d.size_kb.toFixed(1)} KB INT8 MLP (64-32) and the confidence gate. Median ${d.le_ms.toFixed(3)} ms per flow.</small></div>
    <div class="layer c"><b>3 · Cloud</b><small>${esc(d.cloud)} on all ${d.n_full} features, only for forwarded flows. Median ${d.lc_ms.toFixed(2)} ms plus the round trip Δ.</small></div>
    <div class="layer"><b>4 · Application layer</b><small>Final label is logged; non-benign labels raise an alert (firewall rule / actuator command hook).</small></div>
   </div></div>
  <div class="grid2"><div class="card"><h3>Algorithm 1 — confidence-gated inference</h3>
<pre>x'  ← preprocess(x)
p   ← softmax(f_edge(x') / Tc)        Tc = ${d.temperature.toFixed(3)} (temperature scaling, validation)
c   ← max_k p_k ;  ŷ_edge ← argmax_k p_k
fwd ← c &lt; τ                            policy A: uncertain only   (Eq. 2)
if policy = B and ŷ_edge ≠ benign: fwd ← true     (Eq. 3)
ŷ   ← fwd ? f_cloud(x) : ŷ_edge       one WAN round trip when forwarded
log ŷ ; alert if ŷ ≠ benign</pre></div>
  <div class="card"><h3>Models and equations</h3>
   <div class="callout"><b>Latency</b> L(τ) = Le + f(τ)(Δ + Lc) <span class="mono">(4)</span><br><b>Break-even</b> f(τ) &lt; 1 − Le/(Δ + Lc) <span class="mono">(5)</span><br><b>Threshold</b> τ* = argmin f(τ) s.t. F1(τ) ≥ F1<sub>cloud</sub> − ε, ε = 0.005, validation only <span class="mono">(6)</span></div>
   <p class="muted">Edge: 64-32 ReLU MLP, Adam, early stopping, cross-entropy only (no distillation, as in the reported experiments), per-tensor symmetric INT8 weights (simulated: stored as integers, de-quantized for the matmul). Cloud: XGBoost (250 trees, depth 8) vs 256-128-64 DNN, best on validation. Cloud tier: <b>${esc(S.cloud_tier)}</b>.</p>
   <p class="muted">Run <code>python -m service.run --split</code> to put the cloud model behind a real HTTP hop; the gateway then measures the real HTTP + cloud-inference time and adds the simulated WAN delay Δ on top.</p></div></div>`;
}

/* ---------- boot ---------- */
(async () => { await poll(); const t = location.hash.slice(1) || "datasets"; const b = $$("#tabs button").find(b => b.dataset.tab === t) || $("#tabs button[data-tab=datasets]"); b.click(); })();
window.addEventListener("hashchange", () => { const b = $$("#tabs button").find(x => x.dataset.tab === location.hash.slice(1)); if (b && !b.classList.contains("on")) b.click(); });

// Theme changes only affect presentation; keep the current page and chart data.
(() => {
  const button = document.getElementById('themeToggle');
  function sync() {
    const dark = document.documentElement.dataset.theme === 'dark';
    button.textContent = dark ? 'Theme: Dark' : 'Theme: Light';
    button.setAttribute('aria-pressed', String(dark));
    button.title = dark ? 'Switch to light theme' : 'Switch to dark theme';
  }
  button.addEventListener('click', () => {
    const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem('ids-theme', next); } catch (_) {}
    sync();
  });
  sync();
})();
