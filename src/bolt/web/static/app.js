/* ChainGuard Terminal
 *
 * A view over the pipeline, never a second implementation of it. Every number
 * rendered here arrived from an endpoint that read a real artefact. Nothing is
 * computed client-side except formatting and chart geometry, so the console
 * cannot drift away from what `bolt evaluate` actually produced.
 */
"use strict";

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
};

const state = {
  status: null,
  universe: null,
  asset: "BTC",
  asOf: null,
  stream: null,
  loaded: new Set(),
  ledger: null,
};

/* ------------------------------------------------------------- formatting */

const NA = "--"; // terminal convention for "no value", never "0"

function num(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) return NA;
  return Number(value).toFixed(digits);
}

function pct(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return NA;
  return (Number(value) * 100).toFixed(digits) + "%";
}

function money(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return NA;
  const n = Number(value);
  if (n >= 1000) return n.toLocaleString("en-US", { maximumFractionDigits: 0 });
  if (n >= 1) return n.toFixed(2);
  return n.toFixed(4);
}

const short = (hash, head = 10, tail = 6) => {
  if (!hash) return NA;
  // slice(-0) returns the entire string, not an empty one, so a zero tail has
  // to short-circuit. Without this the masthead printed the hash twice over.
  if (tail <= 0) return hash.length <= head ? hash : hash.slice(0, head);
  return hash.length <= head + tail + 1
    ? hash : `${hash.slice(0, head)}…${hash.slice(-tail)}`;
};

function setBusy(busy, message) {
  const live = $("s-live");
  live.textContent = busy ? "WORKING" : "READY";
  if (busy) live.setAttribute("data-busy", ""); else live.removeAttribute("data-busy");
  $("s-msg").textContent = message || "";
}

/* -------------------------------------------------------------- transport */

async function get(path, params) {
  const url = new URL(path, window.location.origin);
  Object.entries(params || {}).forEach(([k, v]) => {
    if (v !== null && v !== undefined && v !== "") url.searchParams.set(k, v);
  });
  const response = await fetch(url);
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try { detail = (await response.json()).detail || detail; } catch { /* keep status */ }
    const error = new Error(detail);
    error.status = response.status;
    throw error;
  }
  return response.json();
}

/** Render a failure where the data would have been. Never leave a stale table. */
function fail(container, error) {
  container.replaceChildren();
  const box = el("div", "err");
  box.append(el("div", null, error.message));
  if (error.status === 404 || error.status === 503) {
    box.append(el("div", "muted", "The pipeline step that produces this has not been run."));
  }
  container.append(box);
}

function table(columns, rows, options = {}) {
  const t = el("table");
  const thead = el("thead");
  const hr = el("tr");
  columns.forEach((c) => {
    const th = el("th", c.numeric ? "num" : null, c.label);
    if (c.title) th.title = c.title;
    hr.append(th);
  });
  thead.append(hr);
  t.append(thead);

  const tbody = el("tbody");
  if (!rows.length) {
    const tr = el("tr");
    const td = el("td", "muted");
    td.colSpan = columns.length;
    td.textContent = options.empty || "no rows";
    tr.append(td);
    tbody.append(tr);
  }
  rows.forEach((row, index) => {
    const tr = el("tr");
    if (options.emphasise && options.emphasise(row, index)) tr.dataset.emphasis = "";
    if (options.onSelect) {
      tr.style.cursor = "pointer";
      tr.tabIndex = 0;
      tr.addEventListener("click", () => options.onSelect(row));
      tr.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); options.onSelect(row); }
      });
    }
    columns.forEach((c) => {
      const cell = c.render ? c.render(row, index) : row[c.key];
      const td = el("td", c.numeric ? "num" : null);
      if (cell instanceof Node) td.append(cell);
      else td.textContent = cell === null || cell === undefined ? NA : String(cell);
      if (c.cls) td.classList.add(...c.cls(row).split(" ").filter(Boolean));
      tr.append(td);
    });
    tbody.append(tr);
  });
  t.append(tbody);
  return t;
}

/* ------------------------------------------------------------------- boot */

async function boot() {
  tickClock();
  setInterval(tickClock, 1000);

  try {
    state.status = await get("/api/status");
  } catch (error) {
    document.querySelector(".viewport").prepend(
      Object.assign(el("div", "err"), { textContent: `Cannot reach the API: ${error.message}` })
    );
    return;
  }

  const s = state.status;
  $("m-hash").textContent = short(s.dataset_sha256, 8, 6);
  $("m-hash").title = s.dataset_sha256 || "no frozen dataset";
  $("m-build").textContent = short(s.code_version, 7, 0);
  $("m-build").title = s.code_version;
  $("m-threshold").textContent = pct(s.config.threshold, 0);
  $("m-horizon").textContent = `${s.config.horizon_days}D`;
  $("s-features").textContent = s.config.features;
  $("s-models").textContent = s.artefacts.models.length
    ? s.artefacts.models.join(" ") : "none";

  if (!s.ready) {
    const missing = s.missing.join(", ");
    $("chain").replaceChildren(
      Object.assign(el("p", "empty"), {
        innerHTML: `Pipeline incomplete. Missing: <code>${missing}</code>. ` +
          `Run <code>bolt build</code> then <code>bolt train</code>.`,
      })
    );
  }

  try {
    state.universe = await get("/api/universe");
    populateCommand();
  } catch (error) {
    console.error(error);
  }

  get("/api/ledger", { limit: 1 })
    .then((data) => { $("s-ledger").textContent = data.track_record.predictions; })
    .catch(() => { $("s-ledger").textContent = "0"; });

  wireNav();
  wireCommand();
  loadView("monitor");
}

function tickClock() {
  $("m-clock").textContent =
    new Date().toISOString().slice(0, 19).replace("T", " ") + "Z";
}

function populateCommand() {
  const assets = state.universe.assets.filter((a) => a.role === "target" && a.rows > 0);
  const select = $("f-asset");
  select.replaceChildren();
  assets.forEach((a) => {
    const option = el("option", null, a.symbol);
    option.value = a.symbol;
    select.append(option);
  });

  const primary = state.status.config.primary;
  state.asset = assets.some((a) => a.symbol === primary) ? primary : assets[0]?.symbol;
  select.value = state.asset;
  $("s-assets").textContent = `${assets.length}/${state.universe.assets.length}`;

  const btc = state.universe.assets.find((a) => a.symbol === state.asset);
  const last = btc ? btc.last : null;
  const dateField = $("f-date");
  if (last) { dateField.max = last; dateField.value = last; state.asOf = last; }
  const first = btc ? btc.first : null;
  if (first) dateField.min = first;

  // Episode jumps: the dates a reviewer will actually want to inspect. The day
  // BEFORE each episode is the interesting one - that is where a warning would
  // have had to fire to be useful.
  const preset = $("f-preset");
  state.universe.episodes.forEach((episode) => {
    const eve = new Date(episode.start);
    eve.setUTCDate(eve.getUTCDate() - 1);
    const value = eve.toISOString().slice(0, 10);
    const option = el("option", null, `${episode.name} — eve of ${value}`);
    option.value = value;
    preset.append(option);
  });
}

/* -------------------------------------------------------------- navigation */

function wireNav() {
  document.querySelectorAll(".nav button").forEach((button) => {
    button.addEventListener("click", () => activate(button.dataset.view));
  });

  document.addEventListener("keydown", (event) => {
    const typing = /^(INPUT|SELECT|TEXTAREA)$/.test(event.target.tagName);
    if (event.key === "/" && !typing) { event.preventDefault(); $("f-asset").focus(); return; }
    if (typing || event.metaKey || event.ctrlKey || event.altKey) return;
    const views = ["monitor", "market", "models", "drivers", "ledger", "system"];
    const index = Number(event.key) - 1;
    if (index >= 0 && index < views.length) { event.preventDefault(); activate(views[index]); }
  });
}

function activate(view) {
  document.querySelectorAll(".nav button").forEach((b) =>
    b.setAttribute("aria-selected", String(b.dataset.view === view))
  );
  document.querySelectorAll(".view").forEach((section) => {
    if (section.id === `view-${view}`) section.setAttribute("data-active", "");
    else section.removeAttribute("data-active");
  });
  loadView(view);
}

function loadView(view) {
  const loaders = {
    market: loadMarket, models: loadModels, drivers: loadDrivers,
    ledger: loadLedger, system: loadSystem,
  };
  const loader = loaders[view];
  if (!loader || state.loaded.has(view)) return;
  state.loaded.add(view);
  loader().catch((error) => console.error(view, error));
}

function wireCommand() {
  $("command").addEventListener("submit", (event) => {
    event.preventDefault();
    runChain();
  });
  $("f-asset").addEventListener("change", (event) => {
    state.asset = event.target.value;
    state.loaded.delete("market");
    if (document.querySelector("#view-market[data-active]")) {
      state.loaded.add("market");
      loadMarket().catch(console.error);
    }
  });
  $("f-date").addEventListener("change", (event) => { state.asOf = event.target.value; });
  $("f-preset").addEventListener("change", (event) => {
    if (!event.target.value) return;
    $("f-date").value = event.target.value;
    state.asOf = event.target.value;
    activate("monitor");
    runChain();
  });
}

/* ------------------------------------------------------- the agent chain */

const STAGE_LABEL = {
  start: null, agent: null, orchestrator: null, skeptic: null,
  decision: null, explanation: null,
};

function runChain() {
  if (state.stream) { state.stream.close(); state.stream = null; }

  const chain = $("chain");
  chain.replaceChildren();
  $("d-band").textContent = NA;
  $("d-score").textContent = NA;
  $("d-conf").textContent = NA;
  $("d-action").textContent = NA;
  $("d-band").className = "";
  $("d-detail").hidden = true;
  $("c-body").replaceChildren(el("p", "muted", "awaiting decision"));
  $("o-body").replaceChildren(el("p", "muted", "awaiting decision"));

  const asset = $("f-asset").value;
  const asOf = $("f-date").value;
  state.asset = asset; state.asOf = asOf;
  $("d-context").textContent = `${asset} as of ${asOf}`;

  const waiting = el("p", "awaiting", "connecting to the agent chain");
  chain.append(waiting);
  setBusy(true, `running chain: ${asset} ${asOf}`);
  $("f-run").disabled = true;

  const url = new URL("/api/run", window.location.origin);
  url.searchParams.set("asset", asset);
  if (asOf) url.searchParams.set("as_of", asOf);

  const stream = new EventSource(url);
  state.stream = stream;
  let analogues = [];

  const finish = (message) => {
    stream.close();
    state.stream = null;
    $("f-run").disabled = false;
    setBusy(false, message || "");
  };

  stream.addEventListener("start", (event) => {
    waiting.remove();
    const data = JSON.parse(event.data);
    $("chain-sub").textContent =
      `models: ${data.models.length ? data.models.join(", ") : "none loaded"}`;
  });

  ["agent", "orchestrator", "skeptic", "decision", "explanation"].forEach((name) => {
    stream.addEventListener(name, (event) => {
      const report = JSON.parse(event.data);
      chain.append(renderStage(report, name));
      chain.scrollTop = chain.scrollHeight;
      if (name === "decision") renderDecision(report);
      if (name === "explanation") renderExplanation(report);
    });
  });

  stream.addEventListener("analogues", (event) => {
    analogues = JSON.parse(event.data);
    if (analogues.length) chain.append(renderAnalogues(analogues));
  });

  stream.addEventListener("commitment", (event) => {
    renderCommitment(JSON.parse(event.data));
  });

  stream.addEventListener("outcome", (event) => {
    renderOutcome(JSON.parse(event.data));
  });

  stream.addEventListener("done", () => finish("chain complete"));

  stream.onerror = () => {
    if (state.stream !== stream) return;
    waiting.remove();
    if (!chain.children.length) {
      chain.append(Object.assign(el("p", "err"), {
        textContent: "The agent chain stream failed. Check the server log.",
      }));
    }
    finish("stream ended");
  };
}

function renderStage(report, kind) {
  const stage = el("div", "stage");

  const head = el("div", "stage-head");
  head.append(el("span", "stage-name", report.agent.toUpperCase()));

  const verdict = el("div", "stage-verdict");
  if (report.status && report.status !== "OK") {
    verdict.append(el("span", `tag st-${report.status}`, report.status));
  }
  verdict.append(el("span", "muted", `conf ${pct(report.confidence, 0)}`));

  const score = el("span", `stage-score ${report.risk}`,
    report.score === null ? NA : Number(report.score).toFixed(0));
  verdict.append(score);
  verdict.append(el("span", report.risk, report.risk.replace(/_/g, " ")));
  head.append(verdict);
  stage.append(head);

  if (report.score !== null && report.score !== undefined) {
    const meter = el("div", `meter ${report.risk}`);
    const fill = el("i");
    fill.style.transform = `scaleX(${Math.max(0, Math.min(1, report.score / 100))})`;
    meter.append(fill);
    stage.append(meter);
  }

  if (report.evidence && report.evidence.length) {
    const list = el("ul", "evidence");
    report.evidence.forEach((item, index) => {
      const li = el("li");
      li.style.animationDelay = `${index * 34}ms`;
      const dir = el("span", "dir", item.direction > 0 ? "▲" : item.direction < 0 ? "▼" : "·");
      dir.dataset.d = String(item.direction);
      li.append(dir, el("span", null, item.detail));
      list.append(li);
    });
    stage.append(list);
  }

  (report.notes || []).forEach((note) => {
    const flagged = /NOT IMPLEMENTED|PROXIES|DISAGREE|declines|below the/i.test(note);
    stage.append(el("p", `stage-note${flagged ? " flag" : ""}`, note));
  });

  if (kind === "skeptic" && !(report.evidence || []).length) {
    stage.append(el("p", "stage-note", "No material counter-evidence found."));
  }
  return stage;
}

function renderAnalogues(analogues) {
  const stage = el("div", "stage");
  const head = el("div", "stage-head");
  head.append(el("span", "stage-name", "HISTORICAL ANALOGUES"));
  head.append(Object.assign(el("div", "stage-verdict"), {}));
  head.lastChild.append(el("span", "muted", `${analogues.length} precedent(s)`));
  stage.append(head);

  const list = el("ul", "evidence");
  analogues.forEach((item, index) => {
    const li = el("li");
    li.style.animationDelay = `${index * 34}ms`;
    const outcome = item.what_happened_next_30d || {};
    const text = outcome.available
      ? `${item.date} (similarity ${num(item.similarity, 2)}) was followed by a ${pct(outcome.max_drawdown)} drawdown`
      : `${item.date} (similarity ${num(item.similarity, 2)}); outcome unavailable`;
    const dot = el("span", "dir", "·");
    dot.dataset.d = "0";
    li.append(dot, el("span", null, text));
    list.append(li);
  });
  stage.append(list);
  return stage;
}

function renderDecision(report) {
  const band = $("d-band");
  band.textContent = report.risk.replace(/_/g, " ");
  band.className = report.risk;
  band.parentElement.classList.remove("flash");
  void band.offsetWidth;
  band.parentElement.classList.add("flash");

  $("d-score").textContent = report.score === null ? NA : `${Number(report.score).toFixed(0)}/100`;
  $("d-score").className = report.risk;
  $("d-conf").textContent = pct(report.confidence, 0);
  $("d-action").textContent = (report.extra && report.extra.action) || (report.notes || [])[0] || NA;
}

function renderExplanation(report) {
  const detail = $("d-detail");
  detail.replaceChildren();
  const narrative = (report.notes || [])[0];
  if (narrative) detail.append(el("p", null, narrative));

  const drivers = (report.extra && report.extra.top_drivers) || [];
  if (drivers.length) {
    const bars = el("div", "bars");
    bars.style.marginTop = "10px";
    const max = Math.max(...drivers.map((d) => Math.abs(d.share || 0)), 1e-9);
    drivers.forEach((driver, index) => {
      const row = el("div", "bar-row");
      row.append(el("span", "lbl", driver.driver));
      const track = el("div", "bar-track");
      const fill = el("i", "bar-fill");
      fill.style.width = `${(Math.abs(driver.share) / max) * 100}%`;
      fill.style.animationDelay = `${index * 40}ms`;
      track.append(fill);
      row.append(track, el("span", "num", pct(driver.share, 0)));
      bars.append(row);
    });
    detail.append(bars);
  }
  detail.hidden = !detail.children.length;
}

function renderCommitment(data) {
  const body = $("c-body");
  body.replaceChildren();

  const rows = [
    ["PREDICTION ID", short(data.prediction_id, 14, 8), data.prediction_id],
    ["SHA-256", short(data.digest, 14, 8), data.digest],
    ["PAYLOAD", data.payload_path.split(/[\\/]/).pop(), data.payload_path],
  ];
  const list = el("dl", "readout");
  list.style.borderTop = "0";
  rows.forEach(([label, value, title]) => {
    const cell = el("div");
    cell.append(el("dt", null, label));
    const dd = el("dd", "sm", value);
    dd.title = title;
    cell.append(dd);
    list.append(cell);
  });
  body.append(list);

  const committed = data.committed;
  const note = el("p", `note ${committed ? "" : "warn"}`);
  note.textContent = committed
    ? `Committed on-chain at block time ${data.block_time}. ${data.reason}`
    : "Not committed on-chain. The digest proves the payload is unaltered, but only a "
      + "chain timestamp can prove the prediction existed before the outcome.";
  body.append(note);

  if (data.explorer_url) {
    const link = el("a", null, "view transaction");
    link.href = data.explorer_url;
    link.target = "_blank";
    link.rel = "noreferrer";
    body.append(link);
  }

  const severity = data.expected_severity || {};
  if (severity.available) {
    body.append(el("p", "note",
      `Expected decline ${pct(severity.expected_decline_low)} to `
      + `${pct(severity.expected_decline_high)} over ${severity.horizon_days} days, `
      + `from ${severity.basis}.`));
  }
}

function renderOutcome(data) {
  const body = $("o-body");
  body.replaceChildren();

  if (!data.available) {
    body.append(el("p", "muted", data.reason));
    return;
  }

  const list = el("dl", "readout");
  list.style.borderTop = "0";
  const cells = [
    ["MAX DRAWDOWN", pct(data.max_drawdown), data.crashed ? "neg" : "pos"],
    ["THRESHOLD", pct(data.threshold, 0), "muted"],
    ["OUTCOME", data.crashed ? "CRASH" : "NO CRASH", data.crashed ? "neg" : "pos"],
  ];
  cells.forEach(([label, value, cls]) => {
    const cell = el("div");
    cell.append(el("dt", null, label));
    cell.append(el("dd", cls, value));
    list.append(cell);
  });
  body.append(list);

  body.append(el("p", "note",
    `Over the ${data.horizon_days} days after the as-of date, ${state.asset} fell from `
    + `${money(data.start_price)} to a low of ${money(data.min_price)}. This is hindsight: `
    + `the system never sees it at prediction time.`));
}

/* ------------------------------------------------------------ MARKET view */

async function loadMarket() {
  const chart = $("t-chart");
  try {
    const data = await get("/api/timeline", { asset: state.asset });
    $("t-sub").textContent =
      `${data.asset} · ${data.dates[0]} to ${data.dates[data.dates.length - 1]} · ${data.dates.length} sessions`;
    chart.replaceChildren(priceChart(data));
  } catch (error) { fail(chart, error); }

  const fv = $("fv-body");
  try {
    const data = await get("/api/features", { asset: state.asset, as_of: state.asOf });
    $("fv-sub").textContent = `${data.asset} as of ${data.as_of}`;
    fv.replaceChildren(table([
      { label: "FEATURE", key: "feature" },
      { label: "FAMILY", key: "family", cls: () => "muted" },
      { label: "SRC", key: "provenance", cls: (r) => r.provenance },
      { label: "VALUE", numeric: true, render: (r) => num(r.value, 4) },
      { label: "PCTILE", numeric: true, render: (r) => pct(r.percentile, 0),
        cls: (r) => r.percentile !== null && r.percentile >= 0.9 ? "neg"
          : r.percentile !== null && r.percentile <= 0.1 ? "pos" : "" },
      { label: "", render: (r) => percentileBar(r.percentile) },
    ], data.features, {
      emphasise: (r) => r.percentile !== null && (r.percentile >= 0.9 || r.percentile <= 0.1),
    }));
  } catch (error) { fail(fv, error); }

  const u = $("u-body");
  try {
    const rows = state.universe.assets;
    u.replaceChildren(table([
      { label: "SYM", key: "symbol" },
      { label: "ROLE", key: "role",
        cls: (r) => r.role === "context" ? "muted" : "" },
      { label: "ROWS", numeric: true, key: "rows",
        cls: (r) => r.rows === 0 ? "neg" : "" },
      { label: "FIRST", key: "first", cls: () => "muted" },
      { label: "LAST", key: "last", cls: () => "muted" },
      { label: "POS", numeric: true, key: "positive_labels",
        render: (r) => r.role === "context" ? NA : r.positive_labels },
      { label: "LAST CLOSE", numeric: true, render: (r) => money(r.last_close) },
    ], rows, { emphasise: (r) => r.rows === 0 }));

    const thin = rows.filter((r) => r.rows > 0 && r.rows < 400);
    if (thin.length) {
      u.append(Object.assign(el("p", "note warn"), {
        textContent: `${thin.map((r) => r.symbol).join(", ")} carry fewer than 400 sessions `
          + `and post-date every crisis episode. They cannot contribute to the `
          + `cross-episode analysis, and the data card says so.`,
      }));
    }
  } catch (error) { fail(u, error); }
}

function percentileBar(value) {
  const track = el("div", "bar-track");
  track.style.width = "90px";
  if (value === null || value === undefined) return track;
  const fill = el("i", "bar-fill");
  fill.style.width = `${value * 100}%`;
  if (value >= 0.9) fill.style.background = "var(--red)";
  else if (value <= 0.1) fill.style.background = "var(--green)";
  track.append(fill);
  return track;
}

/** Log-scale price line with crisis bands and positive-label ticks. */
function priceChart(data) {
  const W = 1200, H = 320, padL = 56, padR = 12, padT = 12, padB = 34, labelH = 16;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("class", "chart");
  svg.setAttribute("preserveAspectRatio", "none");
  svg.style.height = "320px";

  const make = (tag, attrs) => {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v));
    return node;
  };

  const prices = data.close.filter((v) => v !== null && v > 0);
  if (!prices.length) return el("p", "empty", "no price data");

  const lo = Math.log(Math.min(...prices)), hi = Math.log(Math.max(...prices));
  const plotH = H - padT - padB - labelH;
  const x = (i) => padL + (i / (data.dates.length - 1)) * (W - padL - padR);
  const y = (v) => padT + plotH - ((Math.log(v) - lo) / (hi - lo || 1)) * plotH;

  const index = new Map(data.dates.map((d, i) => [d, i]));
  const nearest = (iso) => {
    if (index.has(iso)) return index.get(iso);
    const target = Date.parse(iso);
    let best = 0, distance = Infinity;
    data.dates.forEach((d, i) => {
      const gap = Math.abs(Date.parse(d) - target);
      if (gap < distance) { distance = gap; best = i; }
    });
    return best;
  };

  // Crisis bands sit behind everything: context, not content.
  data.episodes.forEach((episode) => {
    const x0 = x(nearest(episode.start)), x1 = x(nearest(episode.end));
    svg.append(make("rect", {
      class: "band", x: x0, y: padT, width: Math.max(x1 - x0, 2), height: plotH,
    }));
    const label = make("text", { x: x0 + 3, y: padT + 10, transform: `rotate(90 ${x0 + 3} ${padT + 10})` });
    label.textContent = episode.name;
    label.setAttribute("font-size", "8");
    svg.append(label);
  });

  // Price axis: four gridlines, labelled in real units.
  for (let i = 0; i <= 3; i++) {
    const value = Math.exp(lo + ((hi - lo) * i) / 3);
    const yy = y(value);
    svg.append(make("line", { class: "axis", x1: padL, x2: W - padR, y1: yy, y2: yy }));
    const text = make("text", { x: padL - 6, y: yy + 3, "text-anchor": "end" });
    text.textContent = money(value);
    svg.append(text);
  }

  const path = data.close
    .map((v, i) => (v === null ? null : `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(v).toFixed(1)}`))
    .filter(Boolean).join(" ");
  const line = make("path", { class: "series draw", d: path });
  svg.append(line);

  // Positive crash labels as a tick rail under the price.
  const railY = padT + plotH + 8;
  data.label.forEach((v, i) => {
    if (v === 1) svg.append(make("rect", { class: "mark", x: x(i), y: railY, width: 1.4, height: labelH }));
  });
  const railLabel = make("text", { x: padL - 6, y: railY + 12, "text-anchor": "end" });
  railLabel.textContent = "LABEL";
  svg.append(railLabel);

  // Year ticks.
  let lastYear = null;
  data.dates.forEach((d, i) => {
    const year = d.slice(0, 4);
    if (year !== lastYear) {
      lastYear = year;
      svg.append(make("line", { class: "axis", x1: x(i), x2: x(i), y1: padT, y2: padT + plotH }));
      const text = make("text", { x: x(i) + 3, y: H - 8 });
      text.textContent = year;
      svg.append(text);
    }
  });

  requestAnimationFrame(() => {
    try {
      const length = line.getTotalLength();
      line.style.setProperty("--len", length);
    } catch { line.classList.remove("draw"); }
  });
  return svg;
}

/* ------------------------------------------------------------ MODELS view */

async function loadModels() {
  const mc = $("mc-body");
  let data;
  try {
    data = await get("/api/models");
  } catch (error) { fail(mc, error); return; }

  const baseline = data.baseline_pr_auc;
  $("mc-sub").textContent =
    `walk-forward, embargoed · random-classifier PR-AUC ${num(baseline, 3)}`;

  // The one thing a reader can most easily misread, stated BEFORE the table
  // rather than in a footnote under it.
  const best = data.comparison[0];
  const lift = baseline ? (parseFloat(best.pr_auc) / baseline).toFixed(1) : NA;
  const caveat = el("div", "caveat");
  caveat.append(el("b", null, "Read the ranking, not the magnitude. "));
  caveat.append(document.createTextNode(
    `The best model reaches roughly ${lift}x the random baseline of ${num(baseline, 3)}, `
    + `not a high absolute score. A 20% drawdown within 14 days is a rare, hard target. `
    + `The base paper's F1 of 0.703 is on pin-bar reversals, a far more frequent pattern, `
    + `and the two numbers are not comparable. Any PR-AUC near 0.95 here would be `
    + `evidence of leakage, not skill.`
  ));
  $("models-caveat").replaceChildren(caveat);

  const columns = Object.keys(data.comparison[0]).filter((k) => k !== "model");
  mc.replaceChildren(table([
    { label: "MODEL", key: "model" },
    ...columns.map((c) => ({
      label: c.replace(/_/g, " ").toUpperCase(), key: c, numeric: true,
    })),
  ], data.comparison, { emphasise: (row, index) => index === 0 }));

  mc.append(el("p", "note",
    "Random Forest leads; the from-scratch LSTM does not, and the one-line volatility "
    + "rule ties it. Spec Rule 12.7 requires reporting that rather than tuning the deep "
    + "model until it wins, so it is the headline here and in the report."));

  renderPerFold(data);
  renderLeadTime(data);
  await renderSensitivity();
}

function renderPerFold(data) {
  const pf = $("pf-body");
  if (!data.per_fold.length) {
    pf.replaceChildren(el("p", "empty", "no per-fold table"));
    return;
  }
  const folds = [...new Set(data.per_fold.map((r) => r.fold))].sort();
  const models = [...new Set(data.per_fold.map((r) => r.model))];
  const lookup = new Map(data.per_fold.map((r) => [`${r.model}|${r.fold}`, r]));
  const hi = Math.max(...data.per_fold.map((r) => r.pr_auc).filter((v) => v !== null));

  pf.replaceChildren(table([
    { label: "MODEL", key: "model" },
    ...folds.map((fold) => ({
      label: fold.replace(/^fold\d_/, ""), numeric: true,
      render: (row) => {
        const cell = lookup.get(`${row.model}|${fold}`);
        if (!cell || cell.pr_auc === null) return NA;
        const beat = cell.pr_auc > cell.pr_auc_baseline;
        const span = el("span", beat ? "pos" : "neg", num(cell.pr_auc, 3));
        span.title = `baseline ${num(cell.pr_auc_baseline, 3)} - `
          + `${beat ? "beats" : "fails to beat"} random`;
        span.style.opacity = String(0.5 + 0.5 * (cell.pr_auc / hi));
        return span;
      },
    })),
  ], models.map((model) => ({ model }))));

  pf.append(el("p", "note",
    "Green beats that fold's random baseline; red does not. Every model fails at least "
    + "one fold, which is why the aggregate alone would be misleading."));
}

function renderLeadTime(data) {
  const lt = $("lt-body");
  if (!data.lead_time.length) {
    lt.replaceChildren(el("p", "empty", "no lead-time table"));
    return;
  }
  const episodes = [...new Set(data.lead_time.map((r) => r.episode))];
  const models = [...new Set(data.lead_time.map((r) => r.model))];
  const lookup = new Map(data.lead_time.map((r) => [`${r.model}|${r.episode}`, r]));

  lt.replaceChildren(table([
    { label: "EPISODE", render: (row) => row.episode },
    ...models.map((model) => ({
      label: model.toUpperCase(), numeric: true,
      render: (row) => {
        const cell = lookup.get(`${model}|${row.episode}`);
        if (!cell || cell.lead_time_days === null) {
          const span = el("span", "muted", "none");
          span.title = "no sustained warning fired before this episode";
          return span;
        }
        const days = Number(cell.lead_time_days);
        const span = el("span", days > 0 ? "pos" : "neg", `${days}d`);
        span.title = days > 0
          ? `${days} days of warning`
          : "fired late, after the episode had started";
        return span;
      },
    })),
  ], episodes.map((episode) => ({ episode }))));

  lt.append(el("p", "note",
    'A blank reading of "none" means no sustained crossing - two or more consecutive '
    + "days above threshold - fired before that episode. Failures are shown per episode "
    + "and never averaged away."));
}

async function renderSensitivity() {
  const ls = $("ls-body");
  try {
    const labels = await get("/api/labels");
    ls.replaceChildren(table([
      { label: "THRESHOLD", numeric: true, render: (r) => pct(r.threshold, 0) },
      { label: "HORIZON", numeric: true, render: (r) => `${r.horizon_days}d` },
      { label: "POSITIVE RATE", numeric: true, render: (r) => pct(r.positive_rate, 2) },
      { label: "POSITIVES", numeric: true, key: "n_positive" },
      { label: "LABELLED", numeric: true, key: "n_labelled" },
      { label: "EPISODES COVERED", numeric: true,
        render: (r) => el("span",
          r.episodes_covered === r.episodes_total ? "pos" : "neg",
          `${r.episodes_covered}/${r.episodes_total}`) },
      { label: "MISSED", key: "uncovered",
        cls: (r) => (r.uncovered === "-" ? "muted" : "neg") },
    ], labels.sensitivity, {
      emphasise: (r) => r.threshold === labels.default.threshold
        && r.horizon_days === labels.default.horizon_days,
    }));

    ls.append(el("p", "note",
      `The highlighted row is the configured default: ${pct(labels.default.threshold, 0)} `
      + `within ${labels.default.horizon_days} days, the tightest setting that still covers `
      + `all four crisis episodes. Tighter variants miss the USDC de-peg. This table is `
      + `what makes the crash definition defensible rather than arbitrary.`));
  } catch (error) { fail(ls, error); }
}

/* ----------------------------------------------------------- DRIVERS view */

const FAMILY = {
  technical: ["realized_vol_7", "realized_vol_14", "realized_vol_30", "volume_z_20",
    "drawdown_depth", "drawdown_duration", "rsi_14", "momentum_10", "atr_14"],
  onchain: ["exchange_netflow_7", "whale_tx_count", "active_addresses",
    "stablecoin_netflow", "nvt_ratio"],
  sentiment: ["headline_polarity_1d", "headline_polarity_7d", "headline_count_z",
    "negative_ratio_7d"],
  contagion: ["mean_pairwise_corr_30", "corr_dispersion_30", "btc_lead_lag_5",
    "eigen_centrality_30"],
};

function familyOf(feature) {
  const base = String(feature).replace(/\[t-\d+\]$/, "");
  for (const [family, names] of Object.entries(FAMILY)) {
    if (names.includes(base)) return family;
  }
  return "technical";
}

const FAMILY_VAR = {
  technical: "amber", onchain: "cyan", sentiment: "yellow", contagion: "violet",
};

async function loadDrivers() {
  const head = $("g4-head");
  let data;
  try {
    data = await get("/api/consistency");
  } catch (error) { fail(head, error); return; }

  const summary = data.summary;
  const readout = el("dl", "readout");
  readout.style.borderTop = "0";
  [
    ["STABILITY INDEX", num(summary.stability_index, 3)],
    ["EPISODES", summary.n_episodes],
    ["PAIRS COMPARED", summary.n_pairs],
    ["SOURCE", `${summary.model} / ${summary.method}`],
  ].forEach(([label, value]) => {
    const cell = el("div");
    cell.append(el("dt", null, label), el("dd", null, value));
    readout.append(cell);
  });

  head.replaceChildren(readout);
  head.append(el("p", "note", summary.interpretation));
  head.append(el("p", "note",
    "The interpretation bands are fixed in interpret_stability() before any result is "
    + "computed, so the conclusion cannot be retrofitted to whatever number came out. "
    + "Low consistency would be a negative result, and reportable as one."));

  renderMatrix(data);
  await renderAttribution(summary.model || "xgb");
  renderPerEpisode(data);
}

function renderMatrix(data) {
  const host = $("g4-matrix");
  const t = el("table", "matrix");
  const thead = el("thead");
  const hr = el("tr");
  hr.append(el("th"));
  data.episodes.forEach((name) => hr.append(el("th", null, name)));
  thead.append(hr);
  t.append(thead);

  const tbody = el("tbody");
  data.spearman.forEach((row, i) => {
    const tr = el("tr");
    tr.append(el("th", null, data.episodes[i]));
    row.forEach((value, j) => {
      const td = el("td", null, num(value, 2));
      td.style.background = rankColour(value);
      td.title = `${data.episodes[i]} vs ${data.episodes[j]}: `
        + `rank correlation ${num(value, 3)}`;
      if (i === j) {
        // An episode compared with itself is always 1.00 and carries no
        // information. Mute it structurally rather than with opacity, which
        // would dim the black text along with the fill and fail contrast.
        td.style.background = "var(--bg-raised)";
        td.style.color = "var(--fg-faint)";
      }
      tr.append(td);
    });
    tbody.append(tr);
  });
  t.append(tbody);

  host.replaceChildren(t);
  host.append(el("p", "note",
    "1.00 would mean two crises are explained by the same drivers in the same order; "
    + "0.00 would mean their explanations are unrelated."));
}

function rankColour(value) {
  const v = Math.max(-1, Math.min(1, Number(value) || 0));
  if (v >= 0) {
    return `rgb(${Math.round(214 - 130 * v)},${Math.round(150 + 72 * v)},${Math.round(88 + 8 * v)})`;
  }
  const t = -v;
  return `rgb(${Math.round(214 + 34 * t)},${Math.round(150 - 78 * t)},${Math.round(88 - 18 * t)})`;
}

async function renderAttribution(model) {
  const host = $("attr-body");
  try {
    const data = await get("/api/attribution", { model });
    $("attr-sub").textContent = `${data.model} / ${data.features.length} features`;

    const top = data.features.slice(0, 12);
    const max = Math.max(...top.map((f) => f.share_of_total), 1e-9);
    const bars = el("div", "bars");
    top.forEach((feature, index) => {
      const row = el("div", "bar-row");
      row.append(el("span", "lbl", feature.feature));
      const track = el("div", "bar-track");
      const fill = el("i", "bar-fill");
      fill.style.width = `${(feature.share_of_total / max) * 100}%`;
      fill.style.animationDelay = `${index * 28}ms`;
      fill.dataset.family = familyOf(feature.feature);
      track.append(fill);
      row.append(track, el("span", "num", pct(feature.share_of_total, 1)));
      bars.append(row);
    });
    host.replaceChildren(bars);

    const contagion = top.filter((f) => familyOf(f.feature) === "contagion");
    if (contagion.length) {
      const share = contagion.reduce((sum, f) => sum + f.share_of_total, 0);
      host.append(el("p", "note",
        `Cross-asset contagion features account for ${pct(share, 1)} of attribution across `
        + `the top ${top.length}, led by ${contagion[0].feature}. The base paper models a `
        + `single asset and has none of them. That is G2 - and this is evidence for it, `
        + `not a claim about it.`));
    }
  } catch (error) { fail(host, error); }
}

function renderPerEpisode(data) {
  const host = $("g4-episodes");
  if (!data.top_features.length) {
    host.replaceChildren(el("p", "empty", "no per-episode table"));
    return;
  }
  const episodes = [...new Set(data.top_features.map((r) => r.episode))];
  const ranks = [...new Set(data.top_features.map((r) => r.rank))].sort((a, b) => a - b);
  const lookup = new Map(data.top_features.map((r) => [`${r.episode}|${r.rank}`, r]));

  host.replaceChildren(table([
    { label: "RANK", numeric: true, render: (row) => row.rank },
    ...episodes.map((episode) => ({
      label: episode.toUpperCase(),
      render: (row) => {
        const cell = lookup.get(`${episode}|${row.rank}`);
        if (!cell) return NA;
        const span = el("span", null, cell.feature);
        span.title = `${pct(cell.share_of_total, 1)} of this episode's attribution`;
        span.style.color = `var(--${FAMILY_VAR[familyOf(cell.feature)] || "fg"})`;
        return span;
      },
    })),
  ], ranks.map((rank) => ({ rank }))));

  host.append(el("p", "note",
    "Where a row holds the same feature across every column, that driver recurs across "
    + "crises. Where it does not, that crisis carried a signature of its own. Colour marks "
    + "the feature family: amber technical, cyan on-chain, yellow sentiment, violet contagion."));
}

/* ------------------------------------------------------------ LEDGER view */

async function loadLedger() {
  const body = $("lg-body");
  let data;
  try {
    data = await get("/api/ledger");
  } catch (error) { fail(body, error); return; }

  state.ledger = data;
  const record = data.track_record;
  $("s-ledger").textContent = `${record.predictions}`;

  const readout = $("tr-readout");
  readout.replaceChildren();
  // Deliberately NOT a combined precision here. A headline 0.947 sitting above a
  // table that says "0 alerts out of sample" would be the single most misleading
  // thing this console could display, so the headline counts only what happened
  // and the split below carries the performance claim.
  const counts = record.counts || {};
  [
    ["ISSUED", record.predictions, null],
    ["COMMITTED ON-CHAIN", record.committed, record.committed ? "pos" : "neg"],
    ["RESOLVED", record.resolved, null],
    ["HORIZON OPEN", record.pending, "muted"],
    ["WARNINGS ISSUED",
      (counts.TRUE_POSITIVE || 0) + (counts.FALSE_POSITIVE || 0), null],
    ["CRASHES MISSED", counts.FALSE_NEGATIVE || 0,
      (counts.FALSE_NEGATIVE || 0) ? "neg" : null],
  ].forEach(([label, value, cls]) => {
    const cell = el("div");
    cell.append(el("dt", null, label), el("dd", cls, value));
    readout.append(cell);
  });

  const note = $("tr-note");
  note.replaceChildren();

  // The split is the whole point. A combined precision mixes calls the models
  // were trained on with calls they were not, and the two mean different things.
  if (record.in_sample && record.out_of_sample) {
    note.append(splitTable(record));
  }

  const entries = Object.entries(record.counts || {});
  if (entries.length) {
    const line = el("p", "note");
    line.append(document.createTextNode("All outcomes: "));
    entries.sort().forEach(([key, value], index) => {
      if (index) line.append(document.createTextNode("   "));
      line.append(el("span", key, `${key.replace(/_/g, " ")} ${value}`));
    });
    note.append(line);
  }

  if (!record.predictions) {
    note.append(el("p", "note",
      "The ledger is empty. Run `bolt monitor --cycles 12 --every-days 30` to build a "
      + "track record from the frozen dataset, or run the chain from the MONITOR view."));
  } else if (!record.committed) {
    note.append(el("p", "note warn",
      "No prediction has been committed on-chain yet. The local digests prove the "
      + "payloads are unaltered, but only a chain timestamp proves a prediction existed "
      + "BEFORE its outcome was known - which is the whole of G5. Deploy the registry "
      + "and re-run with --commit."));
  }

  body.replaceChildren(table([
    { label: "ASSET", key: "asset" },
    { label: "AS OF", key: "as_of_date" },
    { label: "BAND", key: "severity_band", cls: (r) => r.severity_band },
    { label: "SCORE", numeric: true, render: (r) => num(r.risk_score, 0) },
    { label: "P(CRASH)", numeric: true, render: (r) => pct(r.probability, 1) },
    { label: "OUTCOME", key: "classification", cls: (r) => r.classification },
    { label: "ACTUAL DD", numeric: true,
      render: (r) => r.actual_max_drawdown === null ? NA : pct(r.actual_max_drawdown, 1),
      cls: (r) => (r.actual_max_drawdown >= 0.2 ? "neg" : "") },
    { label: "CHAIN", render: (r) => el("span", r.committed ? "pos" : "muted",
      r.committed ? "committed" : "local") },
    { label: "DIGEST", render: (r) => {
      const span = el("span", "muted", short(r.digest, 10, 6));
      span.title = r.digest;
      return span;
    } },
  ], data.rows, {
    empty: "no predictions recorded",
    onSelect: (row) => verifyPrediction(row.prediction_id),
    emphasise: (r) => r.severity_band === "HIGH" || r.severity_band === "CRITICAL",
  }));

  if (data.rows.length) {
    body.append(el("p", "note",
      "Select any row to recompute its SHA-256 from the published payload."));
  }
}

/** Side-by-side in-sample vs out-of-sample, with the honest reading stated. */
function splitTable(record) {
  const host = el("div");
  const rows = [
    { period: `IN-SAMPLE (to ${record.training_boundary})`, ...record.in_sample,
      warn: true },
    { period: `OUT OF SAMPLE (after ${record.training_boundary})`,
      ...record.out_of_sample },
  ];

  host.append(table([
    { label: "PERIOD", key: "period" },
    { label: "RESOLVED", numeric: true, key: "resolved" },
    { label: "ALERTS", numeric: true, key: "alerts",
      cls: (r) => (r.alerts === 0 ? "neg" : "") },
    { label: "TP", numeric: true, render: (r) => r.counts.TRUE_POSITIVE || 0 },
    { label: "FP", numeric: true, render: (r) => r.counts.FALSE_POSITIVE || 0 },
    { label: "FN", numeric: true, render: (r) => r.counts.FALSE_NEGATIVE || 0,
      cls: (r) => ((r.counts.FALSE_NEGATIVE || 0) > 0 ? "neg" : "") },
    { label: "PRECISION", numeric: true,
      render: (r) => (r.precision === null ? "undefined" : num(r.precision, 3)),
      cls: (r) => (r.precision === null ? "muted" : "") },
    { label: "RECALL", numeric: true,
      render: (r) => (r.recall === null ? "undefined" : num(r.recall, 3)),
      cls: (r) => (r.recall !== null && r.recall < 0.1 ? "neg" : "") },
  ], rows));

  const out = record.out_of_sample;
  const inn = record.in_sample;
  const caveat = el("div", "caveat");
  caveat.style.marginTop = "14px";
  caveat.append(el("b", null, "The in-sample figures are not evidence of foresight. "));
  caveat.append(document.createTextNode(
    `The deployment models were fitted on everything up to ${record.training_boundary}, `
    + `so a hit before that date shows the agent chain works, not that the system saw `
    + `anything coming. `
  ));

  if (out.resolved && out.alerts === 0) {
    caveat.append(el("b", null,
      `Out of sample the system issued NO warnings across ${out.resolved} resolved `
      + `predictions and missed ${out.counts.FALSE_NEGATIVE || 0} real crashes. `));
    caveat.append(document.createTextNode(
      `In-sample precision of ${num(inn.precision, 3)} therefore describes the training `
      + `period and nothing else. This is the finding, and it is exactly the kind of gap `
      + `an on-chain commitment record exists to make impossible to hide.`
    ));
  } else if (out.resolved) {
    caveat.append(document.createTextNode(
      `Out of sample: ${out.alerts} alert(s), precision `
      + `${out.precision === null ? "undefined" : num(out.precision, 3)}, recall `
      + `${out.recall === null ? "undefined" : num(out.recall, 3)}. That is the number to `
      + `defend.`
    ));
  } else {
    caveat.append(document.createTextNode(
      "No out-of-sample predictions have resolved yet, so there is no honest "
      + "performance claim to make."
    ));
  }
  host.append(caveat);
  return host;
}

async function verifyPrediction(predictionId) {
  const host = $("vf-body");
  host.replaceChildren(el("p", "loading", "recomputing digest"));
  try {
    const data = await get("/api/verify", { prediction_id: predictionId });
    host.replaceChildren();

    const readout = el("dl", "readout");
    readout.style.borderTop = "0";
    [
      ["PREDICTION", short(data.prediction_id, 12, 8), data.prediction_id],
      ["RECOMPUTED", short(data.recomputed_digest, 12, 8), data.recomputed_digest],
      ["RECORDED", short(data.ledger_digest, 12, 8), data.ledger_digest],
      ["AS OF", data.as_of_date, data.as_of_date],
    ].forEach(([label, value, title]) => {
      const cell = el("div");
      cell.append(el("dt", null, label));
      const dd = el("dd", "sm", value);
      dd.title = title || "";
      cell.append(dd);
      readout.append(cell);
    });
    host.append(readout);

    const verdict = el("p", `note ${data.match ? "" : "bad"}`);
    verdict.append(el("span", data.match ? "pos" : "neg",
      data.match ? "DIGEST MATCH. " : "DIGEST MISMATCH. "));
    verdict.append(document.createTextNode(data.note));
    host.append(verdict);
    host.append(el("p", `note ${data.committed ? "" : "warn"}`, data.chain_note));

    host.append(el("p", "note",
      "This recomputation runs through chain/verify.py, which re-implements canonical "
      + "serialisation from the standard library and is tested to import nothing from "
      + "the pipeline it verifies."));
  } catch (error) { fail(host, error); }
}

/* ------------------------------------------------------------ SYSTEM view */

const ARCHITECTURE = [
  ["Market Intelligence", "rules + analytics", "IMPLEMENTED", ""],
  ["On-Chain Intelligence", "on-chain analytics", "IMPLEMENTED",
    "four of five inputs are documented proxies; no Etherscan key configured"],
  ["News & Sentiment", "NLP", "PARTIAL",
    "reports sentiment level only; event identification is not built"],
  ["Quantitative", "LSTM + XGBoost", "IMPLEMENTED", ""],
  ["Risk Orchestrator", "weighted evidence", "IMPLEMENTED", ""],
  ["Explanation", "LLM narrative", "PARTIAL",
    "driver ranking is real attribution; the prose is a deterministic template"],
  ["Skeptic / Red-Team", "falsifiable rules", "IMPLEMENTED", ""],
  ["Decision", "thresholds + abstention", "IMPLEMENTED", ""],
  ["Blockchain Audit", "deterministic code", "IMPLEMENTED", ""],
  ["Evaluation", "outcome analytics", "IMPLEMENTED", ""],
  ["Health", "drift monitoring", "IMPLEMENTED", ""],
];

async function loadSystem() {
  const sys = $("sys-body");
  const s = state.status;
  const artefacts = s.artefacts;

  sys.replaceChildren(table([
    { label: "ARTEFACT", key: "name" },
    { label: "STATE", render: (r) => el("span", r.ok ? "pos" : "neg",
      r.ok ? "present" : "missing") },
    { label: "DETAIL", key: "detail", cls: () => "muted" },
  ], [
    { name: "frozen dataset", ok: artefacts.panel, detail: "data/processed/panel.parquet" },
    { name: "data card", ok: artefacts.data_card, detail: "DATA_CARD.md" },
    { name: "fitted models", ok: artefacts.models.length > 0,
      detail: artefacts.models.join(" ") || "run bolt train" },
    { name: "model comparison", ok: artefacts.model_comparison,
      detail: "run bolt evaluate --all --report" },
    { name: "consistency (G4)", ok: artefacts.consistency,
      detail: "run bolt explain --consistency" },
  ]));

  const config = el("dl", "readout");
  [
    ["SHA-256", short(s.dataset_sha256, 10, 6), s.dataset_sha256],
    ["CODE", short(s.code_version, 10, 0), s.code_version],
    ["SEED", s.config.seed],
    ["LOOKBACK", `${s.config.lookback_days}d`],
    ["EMBARGO", `${s.config.embargo_days}d`],
    ["FEATURES", s.config.features],
  ].forEach(([label, value, title]) => {
    const cell = el("div");
    cell.append(el("dt", null, label));
    const dd = el("dd", "sm", value);
    if (title) dd.title = title;
    cell.append(dd);
    config.append(cell);
  });
  sys.append(config);

  sys.append(el("p", "note",
    `The embargo is ${s.config.embargo_days} days, at least lookback + horizon `
    + `(${s.config.lookback_days} + ${s.config.horizon_days}). A shorter value is rejected `
    + `at config load: a training window's label period could otherwise resolve inside the `
    + `test range.`));

  const arch = $("arch-body");
  arch.replaceChildren(table([
    { label: "AGENT", key: "name" },
    { label: "INTELLIGENCE", key: "kind", cls: () => "muted" },
    { label: "STATUS", key: "status",
      cls: (r) => (r.status === "IMPLEMENTED" ? "pos" : "MODERATE") },
    { label: "LIMITATION", key: "limit", cls: () => "muted" },
  ], ARCHITECTURE.map(([name, kind, status, limit]) => ({ name, kind, status, limit }))));

  arch.append(el("p", "note",
    "Nine of eleven agents are implemented and fully deterministic. The split is not "
    + "arbitrary: every agent whose output enters the hashed on-chain payload must be "
    + "byte-reproducible, or a verifier recomputes a different digest and the commitment "
    + "proves nothing. The two that remain partial are the two whose only job is prose - "
    + "the one thing a cryptographic commitment does not need. Both declare their own "
    + "limitation in every report they emit."));

  await loadHealth();
}

async function loadHealth() {
  const host = $("hl-body");
  const drift = $("dr-body");
  let data;
  try {
    data = await get("/api/health");
  } catch (error) { fail(host, error); return; }

  host.replaceChildren();
  if (data.report) {
    const readout = el("dl", "readout");
    readout.style.borderTop = "0";
    [
      ["CONCERNS", data.report.extra.concerns],
      ["REFERENCE WINDOWS", data.reference_windows],
      ["RECENT WINDOWS", data.current_windows],
    ].forEach(([label, value]) => {
      const cell = el("div");
      cell.append(el("dt", null, label), el("dd", null, value));
      readout.append(cell);
    });
    host.append(readout);

    const recommendation = data.report.extra.recommendation || "";
    const severe = /RETRAINING/.test(recommendation);
    host.append(el("p", `note ${severe ? "warn" : ""}`, recommendation));
    // The recommendation is also the final note; printing both duplicates it.
    (data.report.notes || [])
      .filter((n) => n !== recommendation)
      .forEach((n) => host.append(el("p", "note", n)));
  } else {
    host.replaceChildren(el("p", "empty",
      "Not enough history to compare distributions yet."));
  }

  const readings = data.drift || [];
  if (!readings.length) {
    drift.replaceChildren(el("p", "empty", "no drift readings"));
    return;
  }
  drift.replaceChildren(table([
    { label: "FEATURE", key: "feature" },
    { label: "PSI", numeric: true, render: (r) => num(r.psi, 4) },
    { label: "SEVERITY", key: "severity",
      cls: (r) => (r.severity === "major" ? "neg"
        : r.severity === "minor" ? "MODERATE" : "muted") },
    { label: "", render: (r) => {
      const track = el("div", "bar-track");
      track.style.width = "160px";
      const fill = el("i", "bar-fill");
      const scaled = Math.min(1, (r.psi || 0) / 0.5);
      fill.style.width = `${scaled * 100}%`;
      fill.style.background = r.severity === "major" ? "var(--red)"
        : r.severity === "minor" ? "var(--yellow)" : "var(--green)";
      track.append(fill);
      return track;
    } },
  ], readings, { emphasise: (r) => r.severity === "major" }));

  const structural = readings.filter((r) => (r.psi || 0) > 1.0);
  if (structural.length) {
    drift.append(el("p", "note warn",
      `${structural.map((r) => r.feature).join(", ")} show PSI above 1.0, which is far `
      + `beyond a regime change. These are non-stationary BY CONSTRUCTION - `
      + `drawdown_duration counts days since the running peak and grows without bound in `
      + `a long uptrend, so its distribution must shift. The monitor is right to flag it; `
      + `the finding is about the feature's definition, not about the market.`));
  }

  drift.append(el("p", "note",
    "Population Stability Index between the training era and recent windows. Conventional "
    + "cut-points, fixed in advance: 0.10 minor, 0.25 major. A model scoring inputs unlike "
    + "anything it was trained on is extrapolating, and its probabilities stop meaning "
    + "what they meant."));
}

/* ------------------------------------------------------------------- init */

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot);
} else {
  boot();
}
