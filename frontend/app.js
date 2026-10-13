/* ═══════════════════════════════════════════════════════════
   SpringScan — app.js  (v3 — real API, clean result card)
═══════════════════════════════════════════════════════════ */
"use strict";

const API_URL = "/api/camera1/inference";
const SUPPORTED = ["image/png", "image/jpeg", "image/webp", "image/bmp"];

/* ── Spring ID counter (session) ── */
let springCounter = 1;
function nextSpringId() {
  return "SPR-" + String(springCounter++).padStart(4, "0");
}

/* ══════════════════════════════════════════
   1. THEME
══════════════════════════════════════════ */
const html = document.documentElement;
const themeToggle = document.getElementById("themeToggle");
const themeIcon = document.getElementById("themeIcon");
let currentTheme = localStorage.getItem("sds-theme") || "dark";
applyTheme(currentTheme);
themeToggle.addEventListener("click", () => {
  currentTheme = currentTheme === "dark" ? "light" : "dark";
  applyTheme(currentTheme);
  localStorage.setItem("sds-theme", currentTheme);
});
function applyTheme(t) {
  html.setAttribute("data-theme", t);
  themeIcon.className = t === "dark" ? "fa-solid fa-sun" : "fa-solid fa-moon";
  themeToggle.title =
    t === "dark" ? "Switch to light mode" : "Switch to dark mode";
}

/* ══════════════════════════════════════════
   2. MODE TABS
══════════════════════════════════════════ */
document.querySelectorAll(".mode-tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    const tgt = tab.dataset.mode;
    document.querySelectorAll(".mode-tab").forEach((t) => {
      t.classList.remove("active");
      t.setAttribute("aria-selected", "false");
    });
    tab.classList.add("active");
    tab.setAttribute("aria-selected", "true");
    document
      .querySelectorAll(".panel")
      .forEach((p) => p.classList.toggle("hidden", p.id !== `panel-${tgt}`));
    if (tgt !== "live" && liveStream) stopLive();
  });
});

/* ══════════════════════════════════════════
   3. HELPERS
══════════════════════════════════════════ */
function formatBytes(b) {
  if (b < 1024) return b + " B";
  if (b < 1048576) return (b / 1024).toFixed(1) + " KB";
  return (b / 1048576).toFixed(1) + " MB";
}
function readFileAsDataURL(file) {
  return new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = () => res(r.result);
    r.onerror = rej;
    r.readAsDataURL(file);
  });
}
function setStatus(text, state = "ok") {
  document.getElementById("statusText").textContent = text;
  const dot = document.querySelector(".status-dot");
  dot.className =
    "status-dot" +
    (state === "busy" ? " busy" : state === "error" ? " error" : "");
}

/* ── Real API call ── */
async function runInference(file) {
  const fd = new FormData();
  fd.append("image", file);
  const res = await fetch(API_URL, { method: "POST", body: fd });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Server error ${res.status}`);
  }
  return res.json();
  // Returns: { prediction, confidence, decision, probabilities, inference_ms }
}

/* ── Result card builder ── */
// Renders a clean card: image + spring ID + confidence + verdict
function buildResultCard(dataURL, apiResult, springId) {
  const isUncertain = apiResult.decision === "UNCERTAIN";
  const isGood = apiResult.prediction === "GOOD";

  const pct = (apiResult.confidence * 100).toFixed(1);

  const verdictCls = isUncertain ? "uncertain" : isGood ? "good" : "bad";
  const icon = isUncertain
    ? "fa-question-circle"
    : isGood
      ? "fa-check-circle"
      : "fa-times-circle";

  const label = isUncertain ? "UNCERTAIN" : isGood ? "GOOD" : "DEFECTIVE";
  const decision = apiResult.decision;

  return `
    <div class="result-card">
      <div class="result-img-wrap">
        <img src="${dataURL}" alt="Spring image" class="result-img" />
        <div class="result-verdict-overlay ${verdictCls}">
          <i class="fa-solid ${icon}"></i> ${label}
        </div>
      </div>
      <div class="result-info">
        <div class="result-row">
          <span class="result-label">Spring ID</span>
          <span class="result-value mono">${springId}</span>
        </div>
        <div class="result-row">
          <span class="result-label">Detection Result</span>
          <span class="result-value ${verdictCls}">${label}</span>
        </div>
        <div class="result-row">
          <span class="result-label">Confidence</span>
          <span class="result-value">${pct}%</span>
        </div>
        <div class="result-row">
          <span class="result-label">Decision</span>
          <span class="result-value ${verdictCls}">${decision}</span>
        </div>
      </div>
    </div>`;
}

/* ══════════════════════════════════════════
   4. SINGLE IMAGE
══════════════════════════════════════════ */
const singleDropzone = document.getElementById("singleDropzone");
const singleFileInput = document.getElementById("singleFileInput");
const singleIdle = document.getElementById("singleIdle");
const singlePreview = document.getElementById("singlePreview");
const singlePreviewImg = document.getElementById("singlePreviewImg");
const singleFileMeta = document.getElementById("singleFileMeta");
const singleRemoveBtn = document.getElementById("singleRemoveBtn");
const singleError = document.getElementById("singleError");
const singleErrorMsg = document.getElementById("singleErrorMsg");
const singleInspectBtn = document.getElementById("singleInspectBtn");
const singleUploadSec = document.getElementById("singleUploadSection");
const singleResultsSec = document.getElementById("singleResults");

let singleFile = null,
  singleDataURL = null;

singleDropzone.addEventListener("click", (e) => {
  if (singleRemoveBtn.contains(e.target)) return;
  singleFileInput.click();
});
singleDropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") singleFileInput.click();
});
["dragenter", "dragover"].forEach((ev) =>
  singleDropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    singleDropzone.classList.add("drag-over");
  }),
);
["dragleave", "drop"].forEach((ev) =>
  singleDropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    singleDropzone.classList.remove("drag-over");
  }),
);
singleDropzone.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files[0];
  if (f) loadSingleFile(f);
});
singleFileInput.addEventListener("change", () => {
  if (singleFileInput.files[0]) loadSingleFile(singleFileInput.files[0]);
  singleFileInput.value = "";
});
singleRemoveBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  clearSingle();
});

async function loadSingleFile(file) {
  singleError.classList.add("hidden");
  if (!SUPPORTED.includes(file.type)) {
    singleErrorMsg.textContent = `Unsupported: ${file.name.split(".").pop().toUpperCase()}. Use PNG, JPG, WEBP, or BMP.`;
    singleError.classList.remove("hidden");
    return;
  }
  singleFile = file;
  singleDataURL = await readFileAsDataURL(file);
  singlePreviewImg.src = singleDataURL;
  singleFileMeta.textContent = `${file.name}  ·  ${formatBytes(file.size)}`;
  singleIdle.classList.add("hidden");
  singlePreview.classList.remove("hidden");
  singleInspectBtn.disabled = false;
}

function clearSingle() {
  singleFile = null;
  singleDataURL = null;
  singleIdle.classList.remove("hidden");
  singlePreview.classList.add("hidden");
  singleInspectBtn.disabled = true;
  singleError.classList.add("hidden");
}

singleInspectBtn.addEventListener("click", async () => {
  if (!singleFile) return;
  setStatus("Inspecting…", "busy");
  singleInspectBtn.disabled = true;
  singleInspectBtn.innerHTML =
    '<i class="fa-solid fa-circle-notch fa-spin"></i> Analysing…';
  singleError.classList.add("hidden");

  try {
    const apiResult = await runInference(singleFile);
    const springId = nextSpringId();

    // Replace result section body with clean card
    // Keep the back/another buttons; replace everything else
    const resultBody = document.getElementById("singleResults");
    // Inject card into the images card area — simplest approach: replace dash-main-row and below
    resultBody.innerHTML = `
      <div class="dash-line-header">
        <div class="dash-line-left">
          <span class="dash-live-dot"></span>
          <span class="dash-line-title">Inspection Result</span>
        </div>
        <div class="dash-line-right">
          <button class="back-btn" id="singleBackBtn2">
            <i class="fa-solid fa-arrow-left"></i> New Inspection
          </button>
        </div>
      </div>
      ${buildResultCard(singleDataURL, apiResult, springId)}
      <div style="display:flex;justify-content:center;margin-top:24px;">
        <button class="inspect-btn inspect-btn-outline" id="singleAnotherBtn2">
          <i class="fa-solid fa-rotate-left"></i> Inspect Another
        </button>
      </div>`;

    document
      .getElementById("singleBackBtn2")
      .addEventListener("click", resetSingle);
    document
      .getElementById("singleAnotherBtn2")
      .addEventListener("click", resetSingle);

    singleUploadSec.classList.add("hidden");
    singleResultsSec.classList.remove("hidden");
    singleResultsSec.classList.add("fade-in");
    setStatus("Inspection complete", "ok");
  } catch (err) {
    singleErrorMsg.textContent = "API error: " + err.message;
    singleError.classList.remove("hidden");
    setStatus("Error", "error");
  }

  singleInspectBtn.disabled = false;
  singleInspectBtn.innerHTML =
    '<i class="fa-solid fa-magnifying-glass-chart"></i><span>Run Inspection</span><i class="fa-solid fa-chevron-right"></i>';
});

function resetSingle() {
  clearSingle();
  singleResultsSec.classList.add("hidden");
  singleResultsSec.classList.remove("fade-in");
  singleUploadSec.classList.remove("hidden");
  setStatus("System Ready", "ok");
}

/* ══════════════════════════════════════════
   5. BATCH
══════════════════════════════════════════ */
const batchDropzone = document.getElementById("batchDropzone");
const batchFileInput = document.getElementById("batchFileInput");
const batchAddInput = document.getElementById("batchAddInput");
const batchQueue = document.getElementById("batchQueue");
const batchQueueCount = document.getElementById("batchQueueCount");
const batchFileList = document.getElementById("batchFileList");
const batchInspectBtn = document.getElementById("batchInspectBtn");
const batchUploadSec = document.getElementById("batchUploadSection");
const batchResultsSec = document.getElementById("batchResults");
const batchProgressFill = document.getElementById("batchProgressFill");
const batchProgressLabel = document.getElementById("batchProgressLabel");
const batchKpiStrip = document.getElementById("batchKpiStrip");
const sidebarList = document.getElementById("sidebarList");
const detailPlaceholder = document.getElementById("detailPlaceholder");
const detailContent = document.getElementById("detailContent");
const batchLiveDot = document.getElementById("batchLiveDot");

let batchFiles = [],
  batchData = [],
  activeBatchIdx = -1;

batchDropzone.addEventListener("click", () => batchFileInput.click());
batchDropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") batchFileInput.click();
});
["dragenter", "dragover"].forEach((ev) =>
  batchDropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    batchDropzone.classList.add("drag-over");
  }),
);
["dragleave", "drop"].forEach((ev) =>
  batchDropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    batchDropzone.classList.remove("drag-over");
  }),
);
batchDropzone.addEventListener("drop", (e) =>
  addBatchFiles([...e.dataTransfer.files]),
);
batchFileInput.addEventListener("change", () => {
  addBatchFiles([...batchFileInput.files]);
  batchFileInput.value = "";
});
document
  .getElementById("batchAddMoreBtn")
  .addEventListener("click", () => batchAddInput.click());
batchAddInput.addEventListener("change", () => {
  addBatchFiles([...batchAddInput.files]);
  batchAddInput.value = "";
});
document.getElementById("batchClearAll").addEventListener("click", () => {
  batchFiles = [];
  renderBatchQueue();
});

function addBatchFiles(files) {
  batchFiles.push(...files.filter((f) => SUPPORTED.includes(f.type)));
  renderBatchQueue();
}
function renderBatchQueue() {
  batchFileList.innerHTML = "";
  if (!batchFiles.length) {
    batchQueue.classList.add("hidden");
    batchInspectBtn.disabled = true;
    return;
  }
  batchQueue.classList.remove("hidden");
  batchQueueCount.textContent = `${batchFiles.length} image${batchFiles.length !== 1 ? "s" : ""} selected`;
  batchInspectBtn.disabled = false;
  batchFiles.forEach((f, idx) => {
    const li = document.createElement("li");
    li.className = "batch-file-item";
    const url = URL.createObjectURL(f);
    li.innerHTML = `<img class="bfi-thumb" src="${url}" /><span class="bfi-name">${f.name}</span><span class="bfi-size">${formatBytes(f.size)}</span><button class="bfi-remove" title="Remove"><i class="fa-solid fa-xmark"></i></button>`;
    li.querySelector(".bfi-remove").addEventListener("click", () => {
      batchFiles.splice(idx, 1);
      URL.revokeObjectURL(url);
      renderBatchQueue();
    });
    batchFileList.appendChild(li);
  });
}

batchInspectBtn.addEventListener("click", runBatch);
async function runBatch() {
  if (!batchFiles.length) return;
  setStatus("Batch processing…", "busy");
  batchUploadSec.classList.add("hidden");
  batchResultsSec.classList.remove("hidden");
  batchData = [];
  sidebarList.innerHTML = "";
  detailPlaceholder.classList.remove("hidden");
  detailContent.classList.add("hidden");
  document.getElementById("csvExportBtn").disabled = true;
  batchLiveDot.className = "dash-live-dot busy";

  const total = batchFiles.length;
  let done = 0,
    good = 0,
    bad = 0,
    uncertain = 0;

  const dataURLs = [];
  for (let i = 0; i < total; i++) {
    const url = await readFileAsDataURL(batchFiles[i]);
    dataURLs.push(url);
    addSidebarItem(i, batchFiles[i].name, url);
  }

  for (let i = 0; i < total; i++) {
    batchProgressLabel.textContent = `Processing ${i + 1} / ${total}`;
    try {
      const apiResult = await runInference(batchFiles[i]);
      const springId = nextSpringId();
      const r = {
        apiResult,
        springId,
        dataURL: dataURLs[i],
        filename: batchFiles[i].name,
        isGood: apiResult.prediction === "GOOD",
      };
      batchData.push(r);
      if (r.isGood) {
        good++;
      } else if (r.isUncertain) {
        uncertain++;
      } else {
        bad++;
      }
      updateSidebarItem(i, r);
    } catch (err) {
      const r = {
        apiResult,
        springId,
        dataURL: dataURLs[i],
        filename: batchFiles[i].name,
        isGood: apiResult.decision === "ACCEPT",
        isUncertain: apiResult.decision === "UNCERTAIN",
      };
      batchData.push(r);
      bad++;
      updateSidebarItem(i, r);
    }
    done++;
    batchProgressFill.style.width = `${(done / total) * 100}%`;
  }

  batchProgressLabel.textContent = `Complete — ${total} image${total !== 1 ? "s" : ""}`;
  batchLiveDot.className = "dash-live-dot";

  // Simple KPI strip: just counts
  batchKpiStrip.innerHTML = `
  <div class="kpi-card">
    <div class="kpi-label">Total</div>
    <div class="kpi-value">${total}</div>
  </div>

  <div class="kpi-card">
    <div class="kpi-label">Good</div>
    <div class="kpi-value good">${good}</div>
  </div>

  <div class="kpi-card">
    <div class="kpi-label">Defective</div>
    <div class="kpi-value bad">${bad}</div>
  </div>

  <div class="kpi-card">
    <div class="kpi-label">Uncertain</div>
    <div class="kpi-value uncertain">${uncertain}</div>
  </div>

  <div class="kpi-card">
    <div class="kpi-label">Pass rate</div>
    <div class="kpi-value">${((good / total) * 100).toFixed(1)}%</div>
  </div>
`;

  document.getElementById("csvExportBtn").disabled = false;
  setStatus("Batch complete", "ok");
  if (batchData.length) selectBatchItem(0);
}

function addSidebarItem(idx, name, thumb) {
  const el = document.createElement("div");

  el.className = "sidebar-item";
  el.dataset.idx = idx;

  const img = document.createElement("img");
  img.className = "sidebar-thumb";
  img.src = thumb;

  const info = document.createElement("div");
  info.className = "sidebar-info";

  const fileName = document.createElement("div");
  fileName.className = "sidebar-fname";
  fileName.textContent = name;

  const status = document.createElement("div");
  status.className = "sidebar-status";
  status.textContent = "Queued";

  info.appendChild(fileName);
  info.appendChild(status);

  const badge = document.createElement("span");
  badge.className = "sidebar-badge pending";
  badge.id = "sbadge-" + idx;
  badge.textContent = "…";

  el.appendChild(img);
  el.appendChild(info);
  el.appendChild(badge);

  el.addEventListener("click", () => selectBatchItem(idx));

  sidebarList.appendChild(el);
}
function updateSidebarItem(idx, r) {
  const el = sidebarList.children[idx];
  if (!el) return;

  const status = r.error
    ? "Error"
    : r.isUncertain
      ? "Uncertain"
      : r.isGood
        ? "Good"
        : "Defective";

  const badgeClass = r.error
    ? "bad"
    : r.isUncertain
      ? "uncertain"
      : r.isGood
        ? "good"
        : "bad";

  const badgeText = r.error
    ? "ERR"
    : r.isUncertain
      ? "UNCERTAIN"
      : r.isGood
        ? "GOOD"
        : "BAD";

  el.querySelector(".sidebar-status").textContent = status;

  const b = document.getElementById("sbadge-" + idx);

  b.className = "sidebar-badge " + badgeClass;
  b.textContent = badgeText;
}
function selectBatchItem(idx) {
  if (!batchData[idx]) return;
  activeBatchIdx = idx;
  [...sidebarList.children].forEach((c, i) =>
    c.classList.toggle("active", i === idx),
  );
  // Scroll active sidebar item into view
  const activeEl = sidebarList.children[idx];
  if (activeEl)
    activeEl.scrollIntoView({ block: "nearest", behavior: "smooth" });

  const r = batchData[idx];
  const total = batchData.length;
  detailPlaceholder.classList.add("hidden");
  detailContent.classList.remove("hidden");

  const navBar =
    '<div class="batch-nav-bar">' +
    '<button class="batch-nav-btn" id="batchPrevBtn" ' +
    (idx === 0 ? "disabled" : "") +
    ">" +
    '<i class="fa-solid fa-chevron-left"></i> Prev' +
    "</button>" +
    '<span class="batch-nav-counter">' +
    (idx + 1) +
    " / " +
    total +
    "</span>" +
    '<button class="batch-nav-btn" id="batchNextBtn" ' +
    (idx === total - 1 ? "disabled" : "") +
    ">" +
    'Next <i class="fa-solid fa-chevron-right"></i>' +
    "</button>" +
    "</div>";

  if (r.error) {
    detailContent.innerHTML =
      navBar +
      '<div style="padding:24px;color:var(--bad)">' +
      '<i class="fa-solid fa-triangle-exclamation"></i> Error: ' +
      r.error +
      "</div>";
  } else {
    detailContent.innerHTML =
      navBar + buildResultCard(r.dataURL, r.apiResult, r.springId);
  }

  document
    .getElementById("batchPrevBtn")
    .addEventListener("click", () => selectBatchItem(activeBatchIdx - 1));
  document
    .getElementById("batchNextBtn")
    .addEventListener("click", () => selectBatchItem(activeBatchIdx + 1));
}

// CSV Export
// CSV Export
document.getElementById("csvExportBtn").addEventListener("click", () => {
  if (!batchData.length) return;

  const rows = ["Spring ID,Filename,Result,Confidence,Decision"];

  batchData.forEach((r) => {
    if (r.error) {
      rows.push('"' + r.springId + '","' + r.filename + '",ERROR,—,—');
    } else {
      const result = r.isUncertain
        ? "UNCERTAIN"
        : r.isGood
          ? "GOOD"
          : "DEFECTIVE";

      rows.push(
        '"' +
          r.springId +
          '","' +
          r.filename +
          '",' +
          result +
          "," +
          (r.apiResult.confidence * 100).toFixed(1) +
          "%," +
          r.apiResult.decision,
      );
    }
  });

  const a = document.createElement("a");

  a.href = URL.createObjectURL(
    new Blob([rows.join("\n")], {
      type: "text/csv",
    }),
  );

  a.download = "spring-batch-" + Date.now() + ".csv";

  a.click();
});

document.getElementById("batchBackBtn").addEventListener("click", () => {
  batchFiles = [];
  batchData = [];
  renderBatchQueue();
  batchResultsSec.classList.add("hidden");
  batchUploadSec.classList.remove("hidden");
  batchProgressFill.style.width = "0%";
  setStatus("System Ready", "ok");
});

/* ══════════════════════════════════════════
   6. LIVE FEED
══════════════════════════════════════════ */
const liveStartBtn = document.getElementById("liveStartBtn");
const liveVideo = document.getElementById("liveVideo");
const liveCanvas = document.getElementById("liveCanvas");
const liveSnapBtn = document.getElementById("liveSnapBtn");
const liveStopBtn = document.getElementById("liveStopBtn");
const camOverlayIdle = document.getElementById("camOverlayIdle");
const camLiveBadge = document.getElementById("camLiveBadge");
const camScanLine = document.getElementById("camScanLine");
const liveFps = document.getElementById("liveFps");
const fpsTick = document.getElementById("fpsTick");
const liveHistory = document.getElementById("liveHistory");
const liveEmptyState = document.getElementById("liveEmptyState");
const liveSnapDash = document.getElementById("liveSnapDash");
const cameraSelect = document.getElementById("cameraSelect");

let liveStream = null,
  fpsInterval = null,
  frameCount = 0,
  liveSnaps = [];

(async () => {
  try {
    await navigator.mediaDevices
      .getUserMedia({ video: true })
      .then((s) => s.getTracks().forEach((t) => t.stop()));
    const devs = await navigator.mediaDevices.enumerateDevices();
    const cams = devs.filter((d) => d.kind === "videoinput");
    cameraSelect.innerHTML = "";
    if (!cams.length) {
      cameraSelect.innerHTML = "<option>No cameras found</option>";
      return;
    }
    cams.forEach((c, i) => {
      const o = document.createElement("option");

      o.value = c.deviceId;
      o.textContent = c.label || "Camera " + (i + 1);

      cameraSelect.appendChild(o);
    });
  } catch {
    cameraSelect.innerHTML = "<option>Camera access denied</option>";
  }
})();

liveStartBtn.addEventListener("click", () => {
  if (liveStream) stopLive();
  else startLive();
});

async function startLive() {
  const did = cameraSelect.value;
  try {
    liveStream = await navigator.mediaDevices.getUserMedia({
      video: did ? { deviceId: { exact: did } } : true,
    });
    liveVideo.srcObject = liveStream;
    liveVideo.play();
    camOverlayIdle.classList.add("hidden");
    camLiveBadge.classList.remove("hidden");
    camScanLine.classList.remove("hidden");
    liveFps.classList.remove("hidden");
    liveSnapBtn.disabled = false;
    liveStopBtn.disabled = false;
    liveStartBtn.innerHTML = '<i class="fa-solid fa-stop"></i> Stop Camera';
    liveStartBtn.classList.add("stop-mode");
    cameraSelect.disabled = true;
    setStatus("Live feed active", "busy");
    fpsInterval = setInterval(() => {
      fpsTick.textContent = frameCount;
      frameCount = 0;
    }, 1000);
    liveVideo.addEventListener("timeupdate", () => frameCount++);
  } catch {
    setStatus("Camera error", "error");
  }
}
function stopLive() {
  if (liveStream) {
    liveStream.getTracks().forEach((t) => t.stop());
    liveStream = null;
  }
  clearInterval(fpsInterval);
  liveVideo.srcObject = null;
  camOverlayIdle.classList.remove("hidden");
  camLiveBadge.classList.add("hidden");
  camScanLine.classList.add("hidden");
  liveFps.classList.add("hidden");
  liveSnapBtn.disabled = true;
  liveStopBtn.disabled = true;
  liveStartBtn.innerHTML = '<i class="fa-solid fa-play"></i> Start Camera';
  liveStartBtn.classList.remove("stop-mode");
  cameraSelect.disabled = false;
  setStatus("System Ready", "ok");
  frameCount = 0;
  fpsTick.textContent = "—";
}
document.getElementById("liveStopBtn").addEventListener("click", stopLive);

liveSnapBtn.addEventListener("click", async () => {
  if (!liveStream) return;
  liveSnapBtn.disabled = true;
  setStatus("Inspecting snapshot…", "busy");

  const ctx = liveCanvas.getContext("2d");
  liveCanvas.width = liveVideo.videoWidth || 640;
  liveCanvas.height = liveVideo.videoHeight || 480;
  ctx.drawImage(liveVideo, 0, 0);
  const dataURL = liveCanvas.toDataURL("image/png");

  // Convert canvas to Blob for API
  liveCanvas.toBlob(async (blob) => {
    try {
      const file = new File([blob], "snap-" + Date.now() + ".png", {
        type: "image/png",
      });
      const apiResult = await runInference(file);
      const springId = nextSpringId();

      liveSnaps.unshift({
        dataURL,
        apiResult,
        springId,
        ts: new Date().toLocaleTimeString(),
      });
      renderLiveHistory();

      // Show latest result as clean card
      liveSnapDash.classList.remove("hidden");
      liveSnapDash.innerHTML =
        '<div class="dash-card-header" style="padding:12px 4px 10px;font-size:.9rem;font-weight:700;color:var(--text-1);">Latest Snapshot Result</div>' +
        buildResultCard(dataURL, apiResult, springId);

      setStatus("Inspection complete", "ok");
    } catch (err) {
      setStatus("API error", "error");
      liveSnapDash.classList.remove("hidden");
      liveSnapDash.innerHTML =
        '<div style="padding:16px;color:var(--bad)">' +
        '<i class="fa-solid fa-triangle-exclamation"></i> Error: ' +
        err.message +
        "</div>";
    }
    liveSnapBtn.disabled = false;
  }, "image/png");
});

function renderLiveHistory() {
  liveEmptyState.classList.add("hidden");
  liveHistory.innerHTML = "";

  liveSnaps.forEach((snap) => {
    const isUncertain = snap.apiResult?.decision === "UNCERTAIN";
    const isGood = snap.apiResult?.decision === "ACCEPT";

    const label = isUncertain ? "UNCERTAIN" : isGood ? "GOOD" : "DEFECTIVE";

    const labelClass = isUncertain ? "uncertain" : isGood ? "good" : "bad";

    const pct = snap.apiResult
      ? (snap.apiResult.confidence * 100).toFixed(1) + "%"
      : "—";

    const el = document.createElement("div");
    el.className = "live-history-item";

    el.innerHTML =
      '<img class="lhi-thumb" src="' +
      snap.dataURL +
      '" />' +
      '<div class="lhi-info">' +
      '<div class="lhi-time">' +
      snap.ts +
      " · " +
      snap.springId +
      "</div>" +
      '<div class="lhi-label ' +
      labelClass +
      '">' +
      label +
      "</div>" +
      '<div class="lhi-conf">' +
      pct +
      " confidence</div>" +
      "</div>";

    el.addEventListener("click", () => openLightbox(snap.dataURL));

    liveHistory.appendChild(el);
  });
}

const lightbox = document.getElementById("lightbox");
const lbImg = document.getElementById("lbImg");

document.getElementById("lbClose").addEventListener("click", closeLightbox);

lightbox.addEventListener("click", (e) => {
  if (e.target === lightbox) closeLightbox();
});

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeLightbox();
});

function openLightbox(src) {
  lbImg.src = src;
  lightbox.classList.add("open");
  document.body.style.overflow = "hidden";
}

function closeLightbox() {
  lightbox.classList.remove("open");
  document.body.style.overflow = "";
}
