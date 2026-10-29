/* LSM-30M live dashboard
 * Fetches publisher-written files under live/ — first from raw.githubusercontent.com
 * (works even when the site itself is a cached deploy), falling back to same-origin.
 */
"use strict";

const GH_OWNER = "Vaskrokodile";
const GH_REPO = "lsm-live";
const GH_BRANCH = "main";
const RAW_BASE = `https://raw.githubusercontent.com/${GH_OWNER}/${GH_REPO}/${GH_BRANCH}/live/lsm30`;
const LOCAL_BASE = "live/lsm30";
const REFRESH_MS = 30_000;

const $ = (id) => document.getElementById(id);
const nf = new Intl.NumberFormat("en-US");

/* ---------- data loading ---------- */

async function fetchLive(name, kind) {
  const bust = `t=${Date.now()}`;
  const apiUrl = `https://api.github.com/repos/${GH_OWNER}/${GH_REPO}/contents/live/lsm30/${name}?ref=${GH_BRANCH}&${bust}`;
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
const fmtPct = (v) => (v == null || isNaN(v) ? "—" : (100 * Number(v)).toFixed(1) + "%");
const fmtRateCount = (rate, total) => rate == null || total == null ? "—" : `${Math.round(rate * total)} / ${total}`;
const fmtCount = (count, total) => count == null || total == null ? "—" : `${count} / ${total}`;
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
  if ($("best-ckpt")) {
    $("best-ckpt").textContent = st.best_checkpoint
      ? `best ckpt ${basename(st.best_checkpoint)}` : "";
  }
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
  $("k-val-source").textContent = st.val_total_source ? `source ${st.val_total_source}` : "";
  $("k-best").textContent = fmtLoss(st.best_val);
  $("k-best-step").textContent =
    st.best_step != null ? `@ step ${nf.format(st.best_step)}` : "";
  $("k-best-source").textContent = st.best_val_source ? `source ${st.best_val_source}` : "";

  $("updated").textContent = st.updated_at ? `updated ${ago(st.updated_at)}` : "—";
  $("updated").title = st.updated_at || "";

  renderConfig(st.config);
  renderHtmlQuality(st.generation_eval, st.generation_comparison);
}

function renderHtmlQuality(evalRow, comparison) {
  const meta = $("html-eval-meta");
  const note = $("html-eval-note");
  const paired = $("html-comparison");
  if (!evalRow || typeof evalRow !== "object") {
    $("html-balance").textContent = "—";
    $("html-balanced").textContent = "—";
    $("html-doctype").textContent = "—";
    $("html-pairs").textContent = "—";
    meta.textContent = "no generated-page evaluation";
    note.textContent = "No samples evaluated for this run.";
  } else {
    const n = Number.isInteger(evalRow.samples) && evalRow.samples > 0 ? evalRow.samples : null;
    const settings = evalRow.settings && typeof evalRow.settings === "object" ? evalRow.settings : null;
    $("html-balance").textContent = fmtPct(evalRow.balance);
    $("html-balanced").textContent = fmtRateCount(evalRow.balanced, n);
    $("html-doctype").textContent = fmtRateCount(evalRow.doctype, n);
    $("html-pairs").textContent = `${fmtRateCount(evalRow.html_pair, n)} HTML · ${fmtRateCount(evalRow.body_pair, n)} body`;
    meta.textContent = `step ${fmtNum(evalRow.step)} · ${n == null ? "sample count unavailable" : `${nf.format(n)} samples`}`;
    const temperature = settings?.temperature;
    const topK = settings?.top_k;
    const cap = settings?.max_new_bytes ?? settings?.max_new;
    note.textContent = temperature != null && topK != null && cap != null
      ? `temperature ${temperature}; top-k ${topK}; cap ${cap} bytes. Closure and pair rates are proportions of generated continuations.`
      : "Metrics describe generated samples; decode settings or sample count may be unavailable.";
  }
  const before = comparison && comparison.before && typeof comparison.before === "object"
    ? comparison.before : null;
  const after = comparison && comparison.after && typeof comparison.after === "object"
    ? comparison.after : null;
  if (before && after) {
    const beforeStep = before.checkpoint_step;
    const afterStep = after.checkpoint_step;
    const provenance = beforeStep != null && afterStep != null
      ? ` Checkpoints ${fmtNum(beforeStep)} → ${fmtNum(afterStep)}.` : "";
    const beforeSamples = Number.isInteger(before.samples) && before.samples > 0 ? before.samples : null;
    const afterSamples = Number.isInteger(after.samples) && after.samples > 0 ? after.samples : null;
    paired.textContent = `Matched pre-run → final: tag closure ${fmtPct(before.balance)} → ${fmtPct(after.balance)}; balanced ${fmtRateCount(before.balanced, beforeSamples)} → ${fmtRateCount(after.balanced, afterSamples)}; HTML pairs ${fmtRateCount(before.html_pair, beforeSamples)} → ${fmtRateCount(after.html_pair, afterSamples)}.${provenance} ${comparison.caveat || "Small comparison; interpret cautiously."}`;
  } else {
    paired.textContent = "";
  }
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
    const near = hist.find((h) => h.event === "log" && h.step === r.step &&
      (h.val_total != null || h.val != null || h.val_v2 != null || h.val_v3a != null || h.val_legacy != null));
    const val = near
      ? near.val_total ?? near.val_v3a ?? near.val_v2 ?? near.val_legacy ?? near.val
      : r.val_total ?? r.val_v3a ?? r.val_v2 ?? r.val_legacy ?? r.val ?? null;
    const valLabel = near
      ? (near.val_total_source ?? (near.val_v3a != null ? "v3a" : near.val_v2 != null ? "v2" : near.val_legacy != null || near.val != null ? "legacy" : ""))
      : (r.val_total_source ?? (r.val_v3a != null ? "v3a" : r.val_v2 != null ? "v2" : r.val_legacy != null || r.val != null ? "legacy" : ""));
    rows.push({ step: r.step, kind: r.kind || "ckpt", file: r.checkpoint, val, valLabel });
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
    const stepEl = document.createElement("span");
    stepEl.className = "ckpt-step";
    stepEl.textContent = `step ${nf.format(r.step)}`;
    const kindEl = document.createElement("span");
    kindEl.className = `ckpt-kind${r.kind === "best" ? " k-best" : ""}`;
    kindEl.textContent = r.kind;
    const fileEl = document.createElement("span");
    fileEl.className = "ckpt-file";
    fileEl.title = file;
    fileEl.textContent = file;
    const valEl = document.createElement("span");
    valEl.className = "ckpt-val";
    valEl.textContent = r.val != null
      ? `${r.valLabel ? r.valLabel + " " : "val "}${fmtLoss(r.val)}` : "";
    div.append(stepEl, kindEl, fileEl, valEl);
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

  const logs = hist.filter((r) => r.event === "log" || r.event === "eval" || r.event == null);
  const steps = logs.map((r) => r.step);
  const train = logs.map((r) => (r.train != null ? r.train : null));
  const ema = logs.map((r) => (r.ema != null ? r.ema : null));
  const valV2 = logs.map((r) => (r.val_v2 != null ? r.val_v2 : null));
  const valV3a = logs.map((r) => (r.val_v3a != null ? r.val_v3a : null));
  const valTotal = logs.map((r) => (r.val_total != null ? r.val_total : null));
  const valLmV2 = logs.map((r) => (r.val_lm_v2 != null ? r.val_lm_v2 : null));
  const valLmV3a = logs.map((r) => (r.val_lm_v3a != null ? r.val_lm_v3a : null));
  const valLegacy = logs.map((r) => (r.val_legacy != null ? r.val_legacy : r.val != null ? r.val : null));
  const tps = logs.map((r) => (r.tok_s != null ? r.tok_s : null));

  const bests = hist.filter((r) => r.event === "best" && (r.val_total != null || r.val_v2 != null || r.val_v3a != null || r.val_legacy != null || r.val != null));
  const ckpts = hist.filter((r) => r.event === "checkpoint");
  const valMin = Math.min(...valTotal.concat(valV2, valV3a).filter((v) => v != null), Infinity);
  const ckptBase = isFinite(valMin) ? valMin - 0.02 : 0;

  const bestPtsV2 = bests.filter((b) => b.val_v2 != null).map((b) => [b.step, b.val_v2]);
  const bestPtsV3a = bests.filter((b) => b.val_v3a != null).map((b) => [b.step, b.val_v3a]);
  const bestPtsTotal = bests.filter((b) => b.val_total != null).map((b) => [b.step, b.val_total]);
  const bestPtsLegacy = bests.filter((b) => b.val_legacy != null || b.val != null)
    .map((b) => [b.step, b.val_legacy ?? b.val]);
  const ckptPts = ckpts.map((c) => [c.step, ckptBase]);

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
          name: "val total v2",
          type: "line",
          data: steps.map((s, i) => [s, valV2[i]]),
          showSymbol: true,
          symbol: "circle",
          symbolSize: 3.5,
          lineStyle: { width: 1.8, color: "#f0b429" },
          itemStyle: { color: "#f0b429" },
          connectNulls: false,
          z: 3,
        },
        {
          name: "val total v3a",
          type: "line",
          data: steps.map((s, i) => [s, valV3a[i]]),
          showSymbol: true,
          symbol: "circle",
          symbolSize: 3.5,
          lineStyle: { width: 1.8, color: "#ff8f5b" },
          itemStyle: { color: "#ff8f5b" },
          connectNulls: false,
          z: 3,
        },
        {
          name: "val byte CE v2",
          type: "line",
          data: steps.map((s, i) => [s, valLmV2[i]]),
          showSymbol: false,
          lineStyle: { width: 1.2, color: "#c98ff0", type: "dashed" },
          itemStyle: { color: "#c98ff0" },
          connectNulls: false,
          z: 2,
        },
        {
          name: "val byte CE v3a",
          type: "line",
          data: steps.map((s, i) => [s, valLmV3a[i]]),
          showSymbol: false,
          lineStyle: { width: 1.2, color: "#84b9ff", type: "dashed" },
          itemStyle: { color: "#84b9ff" },
          connectNulls: false,
          z: 2,
        },
        { name: "val total (dataset-tagged)", type: "line",
          data: steps.map((s, i) => [s, valTotal[i]]), showSymbol: true,
          symbol: "circle", symbolSize: 3.5,
          lineStyle: { width: 1.5, color: "#d7c6a2", type: "dotted" },
          itemStyle: { color: "#d7c6a2" }, connectNulls: false, z: 2 },
        { name: "val total (unlabeled corpus)", type: "line",
          data: steps.map((s, i) => [s, valLegacy[i]]), showSymbol: true,
          symbol: "circle", symbolSize: 3.5,
          lineStyle: { width: 1.5, color: "#9aa6b2", type: "dotted" },
          itemStyle: { color: "#9aa6b2" }, connectNulls: false, z: 2 },
        { name: "best v2", type: "scatter", data: bestPtsV2,
          symbol: "diamond", symbolSize: 8,
          itemStyle: { color: "#0a0d10", borderColor: "#f0b429", borderWidth: 1.6 },
          tooltip: { valueFormatter: (v) => Number(v).toFixed(4) }, z: 4 },
        { name: "best v3a", type: "scatter", data: bestPtsV3a,
          symbol: "diamond", symbolSize: 8,
          itemStyle: { color: "#0a0d10", borderColor: "#ff8f5b", borderWidth: 1.6 },
          tooltip: { valueFormatter: (v) => Number(v).toFixed(4) }, z: 4 },
        { name: "best (dataset-tagged)", type: "scatter", data: bestPtsTotal,
          symbol: "diamond", symbolSize: 8,
          itemStyle: { color: "#0a0d10", borderColor: "#d7c6a2", borderWidth: 1.6 }, z: 4 },
        { name: "best (unlabeled corpus)", type: "scatter", data: bestPtsLegacy,
          symbol: "diamond", symbolSize: 8,
          itemStyle: { color: "#0a0d10", borderColor: "#9aa6b2", borderWidth: 1.6 }, z: 4 },
        { name: "checkpoint", type: "scatter", data: ckptPts,
          symbol: "rect", symbolSize: [3, 9],
          itemStyle: { color: "#58a6c9", opacity: 0.65 },
          tooltip: { valueFormatter: () => "checkpoint" }, z: 4 },
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

let lastReportText = null;
function renderSafeMarkdown(markdown) {
  const escaped = markdown.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const rendered = marked.parse(escaped, { gfm: true, breaks: false });
  const parsed = new DOMParser().parseFromString(rendered, "text/html");
  const allowed = new Set([
    "H1", "H2", "H3", "H4", "H5", "H6", "P", "UL", "OL", "LI",
    "STRONG", "EM", "DEL", "CODE", "PRE", "BLOCKQUOTE", "HR", "TABLE",
    "THEAD", "TBODY", "TR", "TH", "TD", "A", "BR",
  ]);
  for (const node of parsed.body.querySelectorAll("*")) {
    if (!allowed.has(node.tagName)) {
      node.replaceWith(document.createTextNode(node.textContent || ""));
      continue;
    }
    const href = node.tagName === "A" ? node.getAttribute("href") : null;
    for (const attr of [...node.attributes]) node.removeAttribute(attr.name);
    if (href) {
      try {
        const url = new URL(href, location.href);
        if (url.protocol === "http:" || url.protocol === "https:") {
          node.setAttribute("href", url.href);
          node.setAttribute("rel", "noopener noreferrer");
          node.setAttribute("target", "_blank");
        }
        else node.replaceWith(document.createTextNode(node.textContent || ""));
      } catch {
        node.replaceWith(document.createTextNode(node.textContent || ""));
      }
    }
  }
  return [...parsed.body.childNodes];
}

async function renderReport() {
  const el = $("report");
  try {
    const md = await fetchLive("report.md", "text");
    if (md === lastReportText) return;
    lastReportText = md;
    if (window.marked && marked.parse) {
      el.replaceChildren(...renderSafeMarkdown(md));
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
    await renderReport();
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
