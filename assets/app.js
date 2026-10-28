/* LSM-30M live dashboard
 * Fetches publisher-written files under live/ — first from raw.githubusercontent.com
 * (works even when the site itself is a cached deploy), falling back to same-origin.
 */
"use strict";

const GH_OWNER = "Vaskrokodile";
const GH_REPO = "lsm-live";
const GH_BRANCH = "main";
const RAW_BASE = `https://raw.githubusercontent.com/${GH_OWNER}/${GH_REPO}/${GH_BRANCH}/live`;
const LOCAL_BASE = "live";
const REFRESH_MS = 30_000;

const $ = (id) => document.getElementById(id);
const nf = new Intl.NumberFormat("en-US");

/* ---------- data loading ---------- */

async function fetchLive(name, kind) {
  const bust = `t=${Date.now()}`;
  const apiUrl = `https://api.github.com/repos/${GH_OWNER}/${GH_REPO}/contents/live/${name}?ref=${GH_BRANCH}&${bust}`;
  const tries = [
    { url: `${RAW_BASE}/${name}?${bust}`, src: "github:raw" },
    { url: apiUrl, src: "github:api", api: true },
    { url: `${LOCAL_BASE}/${name}?${bust}`, src: "local" },
  ];
  let lastErr = null;
  for (const t of tries) {
    try {
      const headers = t.api ? { Accept: "application/vnd.github.raw" } : {};
      const res = await fetch(t.url, { cache: "no-store", headers });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setSrcNote(t.src);
      return kind === "json" ? res.json() : res.text();
    } catch (e) {
      lastErr = e;
    }
  }
  throw lastErr;
}

function setSrcNote(src) {
  const el = $("src-note");
  if (el) el.textContent = `src:${src}`;
  const f = $("foot-src");
  const label = src === "local" ? "same-origin files"
    : src === "github:api" ? "github contents api" : "raw.githubusercontent.com";
  if (f) f.textContent = `serving ${label}`;
}

function parseJsonl(text) {
  const out = [];
  for (const line of text.split("\n")) {
    const s = line.trim();
    if (!s) continue;
    try { out.push(JSON.parse(s)); } catch { /* skip bad lines */ }
  }
  return out;
}

/* ---------- formatting ---------- */

const fmtNum = (n) => (n == null || isNaN(n) ? "—" : nf.format(Math.round(n)));
const fmtLoss = (v) => (v == null || isNaN(v) ? "—" : Number(v).toFixed(4));
const fmtToks = (n) => {
  if (n == null || isNaN(n)) return "—";
  if (n >= 1e9) return (n / 1e9).toFixed(2) + "B";
  if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "k";
  return String(n);
};
const basename = (p) => (p ? String(p).split(/[\\/]/).pop() : "—");

function ago(iso) {
  const t = Date.parse(iso);
  if (isNaN(t)) return iso || "—";
  const s = Math.max(0, (Date.now() - t) / 1000);
  if (s < 60) return `${Math.floor(s)}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${(s / 3600).toFixed(1)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

/* ---------- status / KPIs ---------- */

const STATE_CLS = { training: "pill-training", idle: "pill-idle", done: "pill-done" };

function renderStatus(st) {
  const pill = $("state-pill");
  const state = String(st.state || "idle").toLowerCase();
  pill.className = "pill " + (STATE_CLS[state] || "pill-idle");
  $("state-txt").textContent = state;

  $("run-name").textContent = st.run || "—";
  const meta = [];
  if (st.config && st.config.model) meta.push(st.config.model);
  if (st.config && st.config.hardware) meta.push(st.config.hardware);
  if (st.config && st.config.dataset) meta.push(st.config.dataset);
  $("run-meta").textContent = meta.join("  ·  ");

  $("k-step").textContent = fmtNum(st.step);
  $("k-target").textContent = st.target_step ? `/ ${nf.format(st.target_step)}` : "";
  const pct =
    st.target_step && st.step != null
      ? Math.min(100, (100 * st.step) / st.target_step)
      : 0;
  $("k-prog").style.width = pct.toFixed(1) + "%";
  $("k-prog").title = pct.toFixed(1) + "%";

  $("k-tokens").textContent = fmtToks(st.tokens_seen);
  $("k-tokens").title = fmtNum(st.tokens_seen) + " tokens";
  $("k-tps").textContent = fmtNum(st.tok_per_s);
  $("k-train").textContent = fmtLoss(st.train_loss);
  $("k-ema").textContent = fmtLoss(st.train_ema);
  $("k-val").textContent = fmtLoss(st.val_total);
  $("k-best").textContent = fmtLoss(st.best_val);
  $("k-best-step").textContent =
    st.best_step != null ? `@ step ${nf.format(st.best_step)}` : "";

  $("updated").textContent = st.updated_at ? `updated ${ago(st.updated_at)}` : "—";
  $("updated").title = st.updated_at || "";

  renderConfig(st.config);
}

function renderConfig(cfg) {
  const tbody = $("cfg-table").querySelector("tbody");
  tbody.innerHTML = "";
  if (!cfg || typeof cfg !== "object") {
    tbody.innerHTML = `<tr><td>config</td><td class="muted">not published yet</td></tr>`;
    return;
  }
  for (const [k, v] of Object.entries(cfg)) {
    const tr = document.createElement("tr");
    const td1 = document.createElement("td");
    const td2 = document.createElement("td");
    td1.textContent = k.replace(/_/g, " ");
    td2.textContent = typeof v === "object" ? JSON.stringify(v) : String(v);
    tr.append(td1, td2);
    tbody.appendChild(tr);
  }
}

/* ---------- checkpoints ---------- */

function renderCheckpoints(hist, st) {
  const rows = [];
  const seen = new Set();
  for (const r of hist) {
    if (r.event !== "checkpoint") continue;
    const key = `${r.step}|${r.kind || "?"}|${r.checkpoint || ""}`;
    if (seen.has(key)) continue;
    seen.add(key);
    const near = hist.find(
      (h) => h.event === "log" && h.step === r.step && h.val != null
    );
    rows.push({ step: r.step, kind: r.kind || "ckpt", file: r.checkpoint, val: near && near.val });
  }
  rows.sort((a, b) => b.step - a.step || String(b.kind).localeCompare(String(a.kind)));

  const latestPath = st && st.checkpoint ? basename(st.checkpoint) : null;
  const el = $("ckpt-list");
  $("ckpt-count").textContent = rows.length ? `${rows.length} recorded` : "";
  if (!rows.length && !latestPath) {
    el.innerHTML = `<div class="empty mono">no checkpoints yet</div>`;
    return;
  }
  el.innerHTML = "";
  for (const r of rows.slice(0, 60)) {
    const div = document.createElement("div");
    div.className = "ckpt-row";
    const file = basename(r.file) || latestPath || "checkpoint.pt";
    div.innerHTML =
      `<span class="ckpt-step">step ${nf.format(r.step)}</span>` +
      `<span class="ckpt-kind ${r.kind === "best" ? "k-best" : ""}">${r.kind}</span>` +
      `<span class="ckpt-file" title="${file}">${file}</span>` +
      `<span class="ckpt-val">${r.val != null ? "val " + fmtLoss(r.val) : ""}</span>`;
    el.appendChild(div);
  }
}

/* ---------- charts ---------- */

let lossChart = null;
let tpsChart = null;

const CHART_TXT = { color: "#7c8a96", fontFamily: "IBM Plex Mono, monospace", fontSize: 10.5 };
const GRID = { left: 52, right: 16, top: 18, bottom: 34 };
const AXIS_LINE = { lineStyle: { color: "#1e2630" } };
const SPLIT = { lineStyle: { color: "#161d25", type: "dashed" } };

function tooltipBase() {
  return {
    trigger: "axis",
    backgroundColor: "#11161d",
    borderColor: "#2a3540",
    textStyle: { color: "#dbe4ea", fontFamily: "IBM Plex Mono, monospace", fontSize: 11 },
    axisPointer: { lineStyle: { color: "#2a3540" } },
    valueFormatter: (v) => (v == null ? "—" : Number(v).toFixed(4)),
  };
}

function renderCharts(hist) {
  if (typeof echarts === "undefined") return;

  const logs = hist.filter((r) => r.event === "log" || r.event == null);
  const steps = logs.map((r) => r.step);
  const train = logs.map((r) => (r.train != null ? r.train : null));
  const ema = logs.map((r) => (r.ema != null ? r.ema : null));
  const val = logs.map((r) => (r.val != null ? r.val : null));
  const valLm = logs.map((r) => (r.val_lm != null ? r.val_lm : null));
  const tps = logs.map((r) => (r.tok_s != null ? r.tok_s : null));

  const bests = hist.filter((r) => r.event === "best" && r.val != null);
  const ckpts = hist.filter((r) => r.event === "checkpoint");
  const valMin = Math.min(...val.filter((v) => v != null), Infinity);
  const base = isFinite(valMin) ? valMin - 0.02 : 0;

  const bestPts = bests.map((b) => [b.step, b.val]);
  const ckptPts = ckpts.map((c) => [c.step, base]);

  if (!lossChart) lossChart = echarts.init($("loss-chart"), null, { renderer: "canvas" });
  lossChart.setOption(
    {
      animation: false,
      grid: GRID,
      tooltip: { ...tooltipBase(), trigger: "axis" },
      xAxis: {
        type: "value",
        min: "dataMin",
        max: "dataMax",
        name: "step",
        nameLocation: "middle",
        nameGap: 24,
        nameTextStyle: CHART_TXT,
        axisLabel: { ...CHART_TXT, formatter: (v) => (v >= 1000 ? v / 1000 + "k" : v) },
        axisLine: AXIS_LINE,
        splitLine: { show: false },
      },
      yAxis: {
        type: "value",
        scale: true,
        name: "loss",
        nameTextStyle: CHART_TXT,
        axisLabel: { ...CHART_TXT, formatter: (v) => v.toFixed(3) },
        axisLine: { show: false },
        splitLine: SPLIT,
      },
      dataZoom: [
        { type: "inside", xAxisIndex: 0 },
        { type: "slider", xAxisIndex: 0, height: 14, bottom: 4, borderColor: "#1e2630",
          backgroundColor: "#0d1218", fillerColor: "rgba(94,233,181,0.08)",
          handleStyle: { color: "#2a3540" }, moveHandleStyle: { color: "#2a3540" },
          textStyle: { color: "#49555f", fontSize: 9 } },
      ],
      series: [
        {
          name: "train",
          type: "line",
          data: steps.map((s, i) => [s, train[i]]),
          showSymbol: false,
          lineStyle: { width: 1, color: "#4b5866", opacity: 0.8 },
          itemStyle: { color: "#4b5866" },
          connectNulls: true,
          z: 1,
        },
        {
          name: "train EMA",
          type: "line",
          data: steps.map((s, i) => [s, ema[i]]),
          showSymbol: false,
          lineStyle: { width: 1.6, color: "#5ee9b5" },
          itemStyle: { color: "#5ee9b5" },
          connectNulls: true,
          z: 2,
        },
        {
          name: "val",
          type: "line",
          data: steps.map((s, i) => [s, val[i]]),
          showSymbol: true,
          symbol: "circle",
          symbolSize: 3.5,
          lineStyle: { width: 1.8, color: "#f0b429" },
          itemStyle: { color: "#f0b429" },
          connectNulls: true,
          z: 3,
        },
        {
          name: "val lm",
          type: "line",
          data: steps.map((s, i) => [s, valLm[i]]),
          showSymbol: false,
          lineStyle: { width: 1.2, color: "#c98ff0", type: "dashed" },
          itemStyle: { color: "#c98ff0" },
          connectNulls: true,
          z: 2,
        },
        {
          name: "best",
          type: "scatter",
          data: bestPts,
          symbol: "diamond",
          symbolSize: 8,
          itemStyle: { color: "#0a0d10", borderColor: "#5ee9b5", borderWidth: 1.6 },
          tooltip: { valueFormatter: (v) => Number(v).toFixed(4) },
          z: 4,
        },
        {
          name: "checkpoint",
          type: "scatter",
          data: ckptPts,
          symbol: "rect",
          symbolSize: [3, 9],
          itemStyle: { color: "#58a6c9", opacity: 0.65 },
          tooltip: { valueFormatter: () => "checkpoint" },
          z: 4,
        },
      ],
    },
    { notMerge: true }
  );

  if (!tpsChart) tpsChart = echarts.init($("tps-chart"), null, { renderer: "canvas" });
  const tpsVals = tps.filter((v) => v != null);
  const avg = tpsVals.length ? tpsVals.reduce((a, b) => a + b, 0) / tpsVals.length : null;
  $("tps-avg").textContent = avg ? `avg ${nf.format(Math.round(avg))} tok/s` : "";
  tpsChart.setOption(
    {
      animation: false,
      grid: { left: 56, right: 16, top: 14, bottom: 28 },
      tooltip: { ...tooltipBase(), valueFormatter: (v) => (v == null ? "—" : nf.format(Math.round(v)) + " tok/s") },
      xAxis: {
        type: "value",
        min: "dataMin",
        max: "dataMax",
        axisLabel: { ...CHART_TXT, formatter: (v) => (v >= 1000 ? v / 1000 + "k" : v) },
        axisLine: AXIS_LINE,
        splitLine: { show: false },
      },
      yAxis: {
        type: "value",
        scale: true,
        axisLabel: { ...CHART_TXT, formatter: (v) => (v / 1000).toFixed(1) + "k" },
        axisLine: { show: false },
        splitLine: SPLIT,
      },
      series: [
        {
          name: "tok/s",
          type: "line",
          data: steps.map((s, i) => [s, tps[i]]),
          showSymbol: false,
          lineStyle: { width: 1.4, color: "#58a6c9" },
          itemStyle: { color: "#58a6c9" },
          areaStyle: { color: "rgba(88,166,201,0.10)" },
          connectNulls: true,
          markLine: avg
            ? {
                silent: true,
                symbol: "none",
                data: [{ yAxis: avg }],
                lineStyle: { color: "#49555f", type: "dashed", width: 1 },
                label: { show: false },
              }
            : undefined,
        },
      ],
    },
    { notMerge: true }
  );
}

window.addEventListener("resize", () => {
  lossChart && lossChart.resize();
  tpsChart && tpsChart.resize();
});

/* ---------- report ---------- */

async function renderReport() {
  const el = $("report");
  try {
    const md = await fetchLive("report.md", "text");
    if (window.marked && marked.parse) {
      el.innerHTML = marked.parse(md, { gfm: true, breaks: false });
    } else {
      const pre = document.createElement("pre");
      pre.textContent = md;
      el.innerHTML = "";
      el.appendChild(pre);
    }
  } catch {
    el.innerHTML = `<div class="empty mono">no report published yet — expecting live/report.md</div>`;
  }
}

/* ---------- log tail ---------- */

function esc(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function hlLine(line) {
  let cls = "";
  if (/new best/i.test(line)) cls = "hl-best";
  else if (/^checkpoint/i.test(line)) cls = "hl-ckpt";
  else if (/^complete/i.test(line)) cls = "hl-complete";
  else if (/^step \d+/i.test(line)) cls = "hl-step";
  else if (/error|traceback|nan\b/i.test(line) && /error|traceback/i.test(line)) cls = "hl-err";
  const e = esc(line);
  return cls ? `<span class="${cls}">${e}</span>` : e;
}

async function renderLog() {
  const el = $("logtail");
  try {
    const txt = await fetchLive("logtail.txt", "text");
    const lines = txt.split("\n");
    const tail = lines.slice(-200);
    el.innerHTML = tail.map(hlLine).join("\n");
    $("log-meta").textContent = `${tail.filter((l) => l.length).length} lines`;
    if ($("autoscroll").checked) el.scrollTop = el.scrollHeight;
  } catch {
    el.textContent = "no log tail published yet — expecting live/logtail.txt";
  }
}

/* ---------- refresh loop ---------- */

async function refresh() {
  try {
    const [st, histTxt] = await Promise.all([
      fetchLive("status.json", "json"),
      fetchLive("history.jsonl", "text"),
    ]);
    const hist = parseJsonl(histTxt);
    renderStatus(st);
    renderCharts(hist);
    renderCheckpoints(hist, st);
    removeBanner();
  } catch (e) {
    showBanner(`awaiting data — could not fetch live files (${e && e.message}). retrying…`);
  }
}

function showBanner(msg) {
  if ($("err-banner")) { $("err-banner").textContent = msg; return; }
  const d = document.createElement("div");
  d.id = "err-banner";
  d.className = "banner";
  d.textContent = msg;
  document.querySelector("main.wrap").prepend(d);
  const pill = $("state-pill");
  pill.className = "pill pill-error";
  $("state-txt").textContent = "no data";
}
function removeBanner() {
  const d = $("err-banner");
  if (d) d.remove();
}

let remaining = REFRESH_MS / 1000;
setInterval(() => {
  remaining -= 1;
  if (remaining <= 0) {
    remaining = REFRESH_MS / 1000;
    refresh();
    renderLog();
  }
  $("countdown").textContent = remaining + "s";
}, 1000);

/* ---------- boot ---------- */

refresh();
renderReport();
renderLog();
