// BrainLYNX web frontend. No build step — plain JS.

const API = "";
let sessionId = null;
let axis = 1;
let axisLen = 0;
let currentIdx = 0;
let mode = "point"; // "point" | "box"
let objId = 0;
let opacity = 0.5;
let startPage = null;
let endPage = null;
let imgW = 0, imgH = 0;
let scale = 1;
let sourcePath = null;
let lastMaskPath = null;

// per-frame mask cache: {frameIdx: {objId: Image}}
const maskCache = {};

// tab10-ish palette matching the original matplotlib "tab10" colormap used in show_mask
const PALETTE = [
  [31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40],
  [148, 103, 189], [140, 86, 75], [227, 119, 194], [127, 127, 127],
  [188, 189, 34], [23, 190, 207],
];

const $ = (id) => document.getElementById(id);
const baseCanvas = $("baseCanvas");
const maskCanvas = $("maskCanvas");
const overlayCanvas = $("overlayCanvas");
const baseCtx = baseCanvas.getContext("2d");
const maskCtx = maskCanvas.getContext("2d");
const overlayCtx = overlayCanvas.getContext("2d");

function setStatus(text) { $("statusText").textContent = text; }

function appendLog(el, text) {
  const ts = new Date().toLocaleTimeString();
  el.textContent += (el.textContent ? "\n" : "") + `[${ts}] ${text}`;
  el.scrollTop = el.scrollHeight;
}
function clearLog(el) { el.textContent = ""; }

function b64ToImage(b64) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.src = "data:image/png;base64," + b64;
  });
}

function sizeCanvases(w, h) {
  imgW = w; imgH = h;
  const wrap = $("canvasWrap");
  const maxW = wrap.parentElement.clientWidth - 28;
  const maxH = wrap.parentElement.clientHeight - 60;
  scale = Math.min(maxW / w, maxH / h, 4);
  const cw = Math.round(w * scale), ch = Math.round(h * scale);
  for (const c of [baseCanvas, maskCanvas, overlayCanvas]) {
    c.width = w; c.height = h;
    c.style.width = cw + "px";
    c.style.height = ch + "px";
  }
  wrap.style.width = cw + "px";
  wrap.style.height = ch + "px";
}

async function drawBaseSlice(pngB64) {
  const img = await b64ToImage(pngB64);
  sizeCanvases(img.width, img.height);
  baseCtx.clearRect(0, 0, imgW, imgH);
  baseCtx.drawImage(img, 0, 0);
}

async function drawMasks(masksByObj) {
  maskCtx.clearRect(0, 0, imgW, imgH);
  for (const [objIdStr, pngB64] of Object.entries(masksByObj || {})) {
    const oid = parseInt(objIdStr, 10);
    const color = PALETTE[oid % PALETTE.length];
    const img = await b64ToImage(pngB64);

    // draw grayscale mask offscreen, then recolor via compositing
    const tmp = document.createElement("canvas");
    tmp.width = imgW; tmp.height = imgH;
    const tctx = tmp.getContext("2d");
    tctx.drawImage(img, 0, 0);
    const data = tctx.getImageData(0, 0, imgW, imgH);
    for (let i = 0; i < data.data.length; i += 4) {
      const on = data.data[i] > 0; // grayscale value in R channel
      data.data[i] = color[0];
      data.data[i + 1] = color[1];
      data.data[i + 2] = color[2];
      data.data[i + 3] = on ? Math.round(opacity * 255) : 0;
    }
    tctx.putImageData(data, 0, 0);
    maskCtx.drawImage(tmp, 0, 0);
  }
}

function clearOverlay() { overlayCtx.clearRect(0, 0, imgW, imgH); }

function drawPointMarker(x, y, positive) {
  overlayCtx.beginPath();
  overlayCtx.arc(x, y, 4, 0, Math.PI * 2);
  overlayCtx.fillStyle = positive ? "#2ecc71" : "#e74c3c";
  overlayCtx.fill();
}

function drawBoxMarker(x1, y1, x2, y2) {
  overlayCtx.strokeStyle = "#2ecc71";
  overlayCtx.lineWidth = 1;
  overlayCtx.strokeRect(Math.min(x1, x2), Math.min(y1, y2), Math.abs(x2 - x1), Math.abs(y2 - y1));
}

// track markers so re-rendering a frame still shows them (best-effort, client-side only)
const markerState = {}; // frameIdx -> {points: [{x,y,label}], box: [x1,y1,x2,y2]}

function redrawMarkersForCurrentFrame() {
  clearOverlay();
  const m = markerState[currentIdx];
  if (!m) return;
  for (const p of m.points || []) drawPointMarker(p.x, p.y, p.label === 1);
  if (m.box) drawBoxMarker(...m.box);
}

// =============================== API calls ===================================
async function apiPost(path, body, qs) {
  const url = qs ? `${path}?${qs}` : path;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || res.statusText);
  }
  return res.json();
}

async function ensureSession() {
  if (sessionId) return sessionId;
  const r = await apiPost("/api/session", {});
  sessionId = r.session_id;
  return sessionId;
}

async function loadFile(path) {
  if (!path) return;
  await ensureSession();
  setStatus("加载中...");
  try {
    const r = await apiPost("/api/load", { file_path: path, axis }, `session_id=${sessionId}`);
    axisLen = r.axis_len;
    currentIdx = r.idx;
    $("sliceSlider").min = 0;
    $("sliceSlider").max = axisLen - 1;
    $("sliceSlider").value = currentIdx;
    $("sliceLabel").textContent = `${currentIdx + 1} / ${axisLen}`;
    await drawBaseSlice(r.slice_png);
    maskCache[currentIdx] = {};
    clearOverlay();
    setStatus(`已加载: ${path}`);
    sourcePath = path;
    lastMaskPath = null;
    $("savePath").value = deriveSavePath(path);
    $("segPath").value = "";
  } catch (e) {
    setStatus("加载失败: " + e.message);
  }
}

function deriveSavePath(path) {
  // mirrors the original Rename convention: <filename>_mask.nii.gz next to the source file
  const lower = path.toLowerCase();
  let base;
  if (lower.endsWith(".nii.gz")) base = path.slice(0, -7);
  else if (lower.endsWith(".nii")) base = path.slice(0, -4);
  else base = path;
  return base + "_mask.nii.gz";
}

function deriveSegPath(path) {
  const lower = path.toLowerCase();
  let base;
  if (lower.endsWith(".nii.gz")) base = path.slice(0, -7);
  else if (lower.endsWith(".nii")) base = path.slice(0, -4);
  else base = path;
  return base + "_seg.nii.gz";
}

async function setAxis(a) {
  if (!sessionId) return;
  axis = a;
  document.querySelectorAll(".axis-btn").forEach((b) => b.classList.toggle("active", parseInt(b.dataset.axis) === a));
  const r = await apiPost("/api/set_axis", {}, `session_id=${sessionId}&axis=${a}`);
  axisLen = r.axis_len;
  currentIdx = r.idx;
  $("sliceSlider").max = axisLen - 1;
  $("sliceSlider").value = currentIdx;
  $("sliceLabel").textContent = `${currentIdx + 1} / ${axisLen}`;
  await drawBaseSlice(r.slice_png);
  clearOverlay();
}

async function goToSlice(idx) {
  if (!sessionId) return;
  const r = await apiPost("/api/slice", { session_id: sessionId, idx });
  currentIdx = r.idx;
  $("sliceSlider").value = currentIdx;
  $("sliceLabel").textContent = `${currentIdx + 1} / ${axisLen}`;
  await drawBaseSlice(r.slice_png);
  await drawMasks(r.masks);
  redrawMarkersForCurrentFrame();
}

function canvasCoords(evt) {
  const rect = baseCanvas.getBoundingClientRect();
  const x = (evt.clientX - rect.left) * (imgW / rect.width);
  const y = (evt.clientY - rect.top) * (imgH / rect.height);
  return { x, y };
}

async function onPoint(x, y, label) {
  if (!markerState[currentIdx]) markerState[currentIdx] = { points: [] };
  if (!markerState[currentIdx].points) markerState[currentIdx].points = [];
  markerState[currentIdx].points.push({ x, y, label });
  drawPointMarker(x, y, label === 1);
  try {
    const r = await apiPost("/api/point", { session_id: sessionId, obj_id: objId, x, y, label });
    await drawMasks(r.masks);
  } catch (e) {
    setStatus("推理失败: " + e.message);
  }
}

let boxStart = null;
async function onBoxDrag(x1, y1, x2, y2) {
  markerState[currentIdx] = { ...(markerState[currentIdx] || {}), box: [x1, y1, x2, y2] };
  redrawMarkersForCurrentFrame();
  try {
    const r = await apiPost("/api/box", { session_id: sessionId, obj_id: objId, x1, y1, x2, y2 });
    await drawMasks(r.masks);
  } catch (e) {
    setStatus("推理失败: " + e.message);
  }
}

// =============================== events ===================================
$("folderPath").addEventListener("keydown", (e) => {
  if (e.key === "Enter") $("btnListFiles").click();
});

$("btnListFiles").onclick = async () => {
  const dir = $("folderPath").value.trim();
  if (!dir) return;
  const box = $("fileListBox");
  box.textContent = "列表中...";
  box.classList.remove("file-list");
  box.className = "log-box file-list";
  try {
    const res = await fetch(`/api/list_nifti?dir_path=${encodeURIComponent(dir)}`);
    const data = await res.json();
    if (data.error) {
      box.textContent = "错误: " + data.error;
      return;
    }
    if (data.files.length === 0) {
      box.textContent = "该目录下没有找到 .nii / .nii.gz 文件";
      return;
    }
    box.innerHTML = "";
    if (data.truncated) {
      const note = document.createElement("div");
      note.className = "hint";
      note.textContent = `结果过多,只显示前 ${data.files.length} 个`;
      box.appendChild(note);
    }
    for (const f of data.files) {
      const item = document.createElement("div");
      item.className = "file-item";
      item.textContent = f.rel;
      item.dataset.path = f.path;
      item.onclick = () => {
        document.querySelectorAll(".file-item").forEach((el) => el.classList.remove("active"));
        item.classList.add("active");
        loadFile(f.path);
      };
      box.appendChild(item);
    }
  } catch (e) {
    box.textContent = "列出失败: " + e.message;
  }
};

document.querySelectorAll(".axis-btn").forEach((b) => {
  b.onclick = () => setAxis(parseInt(b.dataset.axis));
});

$("modePoint").onclick = () => {
  mode = "point";
  $("modePoint").classList.add("active");
  $("modeBox").classList.remove("active");
};
$("modeBox").onclick = () => {
  mode = "box";
  $("modeBox").classList.add("active");
  $("modePoint").classList.remove("active");
};

$("btnAddObj").onclick = () => {
  const sel = $("objSelect");
  const n = sel.options.length;
  const opt = document.createElement("option");
  opt.value = n;
  opt.textContent = `Object ${n + 1}`;
  sel.appendChild(opt);
  sel.value = n;
  objId = n;
};
$("objSelect").onchange = (e) => { objId = parseInt(e.target.value); };

$("btnClearPrompts").onclick = async () => {
  if (!sessionId) return;
  await apiPost("/api/clear_prompts", { session_id: sessionId, obj_id: objId });
  delete markerState[currentIdx];
  clearOverlay();
};

$("opacitySlider").oninput = async (e) => {
  opacity = parseInt(e.target.value) / 100;
  const r = await apiPost("/api/slice", { session_id: sessionId, idx: currentIdx });
  await drawMasks(r.masks);
};

$("sliceSlider").oninput = (e) => goToSlice(parseInt(e.target.value));

$("canvasWrap").addEventListener("wheel", (e) => {
  e.preventDefault();
  if (!sessionId) return;
  const delta = e.deltaY > 0 ? 1 : -1;
  const next = Math.max(0, Math.min(axisLen - 1, currentIdx + delta));
  goToSlice(next);
}, { passive: false });

overlayCanvas.addEventListener("contextmenu", (e) => e.preventDefault());

overlayCanvas.addEventListener("mousedown", (e) => {
  if (!sessionId) return;
  const { x, y } = canvasCoords(e);
  if (mode === "point") {
    const label = e.button === 2 ? 0 : 1; // right click = background
    onPoint(x, y, label);
  } else {
    boxStart = { x, y };
  }
});

overlayCanvas.addEventListener("mousemove", (e) => {
  if (mode === "box" && boxStart) {
    const { x, y } = canvasCoords(e);
    redrawMarkersForCurrentFrame();
    drawBoxMarker(boxStart.x, boxStart.y, x, y);
  }
});

overlayCanvas.addEventListener("mouseup", (e) => {
  if (mode === "box" && boxStart) {
    const { x, y } = canvasCoords(e);
    onBoxDrag(boxStart.x, boxStart.y, x, y);
    boxStart = null;
  }
});

$("btnSetStart").onclick = async () => {
  if (!sessionId) return;
  const r = await apiPost("/api/set_start_page", {}, `session_id=${sessionId}`);
  startPage = r.start_page;
  $("startPageLabel").textContent = startPage + 1;
};
$("btnSetEnd").onclick = async () => {
  if (!sessionId) return;
  const r = await apiPost("/api/set_end_page", {}, `session_id=${sessionId}`);
  endPage = r.end_page;
  $("endPageLabel").textContent = endPage + 1;
};

$("btnPropagate").onclick = () => {
  if (!sessionId) return;
  if (startPage === null) { setStatus("请先设置起始帧"); return; }
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/propagate/${sessionId}`);
  const progEl = $("propagateProgress");
  clearLog(progEl);
  ws.onopen = () => {
    ws.send(JSON.stringify({
      start_frame: startPage,
      end_frame: endPage !== null ? endPage : axisLen - 1,
    }));
    appendLog(progEl, "已连接,开始传播...");
  };
  ws.onmessage = async (evt) => {
    const msg = JSON.parse(evt.data);
    if (msg.type === "status") {
      appendLog(progEl, msg.message);
    } else if (msg.type === "frame") {
      appendLog(progEl, `已处理帧 ${msg.frame_idx + 1}`);
      maskCache[msg.frame_idx] = msg.masks;
      if (msg.frame_idx === currentIdx) {
        await drawMasks(msg.masks);
      }
      $("sliceSlider").value = msg.frame_idx;
      currentIdx = msg.frame_idx;
      $("sliceLabel").textContent = `${currentIdx + 1} / ${axisLen}`;
      const sr = await apiPost("/api/slice", { session_id: sessionId, idx: msg.frame_idx });
      await drawBaseSlice(sr.slice_png);
      await drawMasks(msg.masks);
    } else if (msg.type === "done") {
      appendLog(progEl, "传播完成。");
    } else if (msg.type === "error") {
      appendLog(progEl, "错误: " + msg.message);
    }
  };
  ws.onerror = () => { appendLog(progEl, "WebSocket 连接错误"); };
};

$("btnSave").onclick = async () => {
  if (!sessionId) return;
  const path = $("savePath").value.trim();
  if (!path) return;
  const force = $("forceSave").checked;
  const resEl = $("saveResult");
  appendLog(resEl, force ? "保存中(强制,缺失帧将填充为空白)..." : "保存中...");
  try {
    const r = await apiPost("/api/save", { session_id: sessionId, file_path: path, force });
    if (r.saved) {
      appendLog(resEl, "已保存: " + r.path);
      if (r.warning) appendLog(resEl, r.warning);
      lastMaskPath = r.path;
      if (!$("segPath").value.trim()) {
        $("segPath").value = deriveSegPath(sourcePath || path);
      }
    } else {
      appendLog(resEl, r.warning || "保存未完成");
    }
  } catch (e) {
    appendLog(resEl, "保存失败: " + e.message);
  }
};

$("btnSaveSeg").onclick = async () => {
  if (!sessionId) return;
  const outPath = $("segPath").value.trim();
  if (!outPath) return;
  const resEl = $("segResult");
  if (!lastMaskPath) {
    appendLog(resEl, "还没有保存过mask,请先完成第7步'保存 Mask'。");
    return;
  }
  appendLog(resEl, "抠除中(原图 × mask)...");
  try {
    const r = await apiPost("/api/save_segmentation", {
      session_id: sessionId,
      output_path: outPath,
      mask_path: lastMaskPath,
    });
    if (r.saved) {
      appendLog(resEl, "已保存: " + r.path);
      appendLog(resEl, "使用的mask: " + r.mask_path);
    } else {
      appendLog(resEl, r.error || "保存未完成");
    }
  } catch (e) {
    appendLog(resEl, "保存失败: " + e.message);
  }
};

window.addEventListener("resize", () => {
  if (imgW && imgH) sizeCanvases(imgW, imgH);
});