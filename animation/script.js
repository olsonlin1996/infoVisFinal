const colorPalette = [
  "#00d2ff", "#f5c542", "#9ae6b4", "#f30505ff", "#7f5af0",
  "#00ff22ff", "#d36d00ff", "#4dabf7", "#ff8fab", "#2a6e33ff",
  "#e97b7bff", "#3ea330ff", "#c0a0ff", "#0004ffff", "#ffe600ff",
  "#fab3b3ff", "#8c00ffff", "#ff2882ff", "#837c7dff", "#ffffffff"
];
const playbackSpeed = 4;

const races = [
  { slug: "2024_bahrain", label: "巴林" },
  { slug: "2024_saudi_arabia", label: "沙烏地阿拉伯" },
  { slug: "2024_australia", label: "澳洲" },
  { slug: "2024_japan", label: "日本" },
  { slug: "2024_china", label: "中國" },
  { slug: "2024_united_states", label: "美國" },
  { slug: "2024_italy", label: "義大利" },
  { slug: "2024_monaco", label: "摩納哥" },
  { slug: "2024_canada", label: "加拿大" },
  { slug: "2024_spain", label: "西班牙" },
  { slug: "2024_austria", label: "奧地利" },
  { slug: "2024_united_kingdom", label: "英國" },
  { slug: "2024_hungary", label: "匈牙利" },
  { slug: "2024_belgium", label: "比利時" },
  { slug: "2024_netherlands", label: "荷蘭" },
  { slug: "2024_azerbaijan", label: "亞塞拜然" },
  { slug: "2024_singapore", label: "新加坡" },
  { slug: "2024_mexico", label: "墨西哥" },
  { slug: "2024_brazil", label: "巴西" },
  { slug: "2024_qatar", label: "卡達" },
  { slug: "2024_united_arab_emirates", label: "阿聯" },
];

const listEl = document.getElementById("driverList");
const canvas = document.getElementById("trackCanvas");
const ctx = canvas.getContext("2d");
const codeEl = document.getElementById("driverCode");
const nameEl = document.getElementById("driverName");
const noteEl = document.getElementById("driverNote");
const statusEl = document.getElementById("statusText");
const progressEl = document.getElementById("progressList");
const progressSlider = document.getElementById("progressSlider");
const progressTime = document.getElementById("progressTime");
const goalImage = new Image();
goalImage.src = "goal.png";
const defaultMarkerImage = new Image();
defaultMarkerImage.src = "racing-car.png";
const markerCache = new Map(); // hex -> Image

const state = {
  drivers: [],
  scaleMeta: null,
  totalDuration: 0,
  maxFinishedDuration: 0,
  startStamp: null,
  rafId: null,
  currentRace: null,
  progressBars: new Map(),
  isScrubbing: false,
  rankData: new Map(), // code -> [{t, pos, ...}]
  hover: { active: false, x: 0, y: 0 },
  lastElapsed: 0,
};

function preloadMarkerImages() {
  colorPalette.forEach((col) => {
    const key = col.replace("#", "").toLowerCase();
    if (markerCache.has(key)) return;
    const img = new Image();
    img.src = `marker/${key}.png`;
    img.onload = () => {
      markerCache.set(key, img);
      if (!state.rafId && state.drivers.length) renderScene(state.lastElapsed, false);
    };
    img.onerror = () => markerCache.set(key, defaultMarkerImage);
  });
}

function renderRaceButtons() {
  listEl.innerHTML = "";
  races.forEach((race) => {
    const btn = document.createElement("button");
    btn.className = "driver-btn";
    btn.dataset.slug = race.slug;
    btn.textContent = race.label;
    btn.addEventListener("click", () => loadRace(race));
    listEl.appendChild(btn);
  });
}

function setActiveRace(slug) {
  document.querySelectorAll(".driver-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.slug === slug);
  });
}

function computeScaleMeta(drivers) {
  if (!drivers.length) return null;
  const rect = canvas.getBoundingClientRect();
  const padding = 40;
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  drivers.forEach((d) => {
    d.points.forEach((p) => {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    });
  });
  const scale = Math.min(
    (rect.width - padding * 2) / Math.max(1, maxX - minX),
    (rect.height - padding * 2) / Math.max(1, maxY - minY)
  );
  return { minX, minY, maxY, scale, padding, width: rect.width, height: rect.height };
}

function buildScaled(points, meta) {
  if (!meta) return [];
  return points.map((p) => ({
    x: (p.x - meta.minX) * meta.scale + meta.padding,
    y: (meta.maxY - p.y) * meta.scale + meta.padding,
    t: p.t,
  }));
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * window.devicePixelRatio;
  canvas.height = rect.height * window.devicePixelRatio;
  ctx.setTransform(window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0);

  state.scaleMeta = computeScaleMeta(state.drivers);
  state.drivers.forEach((d) => {
    d.scaled = buildScaled(d.points, state.scaleMeta);
  });
  redrawStatic();
}

function redrawStatic() {
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  if (!state.scaleMeta) return;

  ctx.save();
  ctx.translate(0.5, 0.5);
  ctx.strokeStyle = "rgba(255,255,255,0.12)";
  ctx.lineWidth = 1.5;
  let goalPos = null;
  let bestFinish = null; // track driver with最短完賽時間
  state.drivers.forEach((driver) => {
    if (!driver.scaled.length) return;
    ctx.beginPath();
    driver.scaled.forEach((p, i) => {
      if (i === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    });
    ctx.stroke();
    // 以最早完賽者的終點作為 goal
    if (
      driver.finished &&
      driver.totalDuration != null &&
      (bestFinish === null || driver.totalDuration < bestFinish.totalDuration)
    ) {
      bestFinish = driver;
      goalPos = driver.scaled[driver.scaled.length - 1];
    }
  });
  ctx.restore();
  if (goalPos) drawGoalMarker(goalPos.x, goalPos.y);
}

function updateDriverProgress(driver, elapsed) {
  while (
    driver.currentIndex + 1 < driver.points.length &&
    driver.points[driver.currentIndex + 1].t <= elapsed
  ) {
    driver.currentIndex += 1;
  }

  const nextIndex = Math.min(driver.currentIndex + 1, driver.points.length - 1);
  const pA = driver.scaled[driver.currentIndex];
  const pB = driver.scaled[nextIndex];
  const tA = driver.points[driver.currentIndex].t;
  const tB = driver.points[nextIndex].t;
  let cx = pA.x, cy = pA.y;
  let angle = Math.atan2(pB.y - pA.y, pB.x - pA.x);

  if (elapsed >= tB) {
    cx = pB.x;
    cy = pB.y;
    driver.currentIndex = nextIndex;
  } else if (tB > tA && elapsed > tA) {
    const alpha = Math.min(1, Math.max(0, (elapsed - tA) / (tB - tA)));
    cx = pA.x + (pB.x - pA.x) * alpha;
    cy = pA.y + (pB.y - pA.y) * alpha;
  }

  return { cx, cy, angle };
}

function drawFrame(timestamp) {
  if (!state.drivers.length) return;
  if (!state.startStamp) state.startStamp = timestamp;
  const elapsed = (timestamp - state.startStamp) * playbackSpeed;
  renderScene(elapsed, true);
}

function renderScene(elapsed, scheduleNext = false) {
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  redrawStatic();

  let finished = true;
  const markerDraws = [];
  state.drivers.forEach((driver) => {
    if (!driver.scaled.length) return;
    const { cx, cy, angle } = updateDriverProgress(driver, elapsed);
    const isHover =
      state.hover.active &&
      Math.hypot(state.hover.x - cx, state.hover.y - cy) <= 16;

    ctx.save();
    ctx.strokeStyle = driver.color;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    driver.scaled.forEach((p, i) => {
      if (i === 0) ctx.moveTo(p.x, p.y);
      else if (i <= driver.currentIndex) ctx.lineTo(p.x, p.y);
    });
    ctx.lineTo(cx, cy);
    ctx.stroke();
    ctx.restore();

    markerDraws.push({
      cx,
      cy,
      angle,
      color: driver.color,
      img: resolveMarkerImage(driver.color),
      label: isHover ? driver.name : "",
    });

    if (elapsed < driver.totalDuration) finished = false;
  });

  // 確保車圖層在路徑之上
  markerDraws.forEach((m) => drawMarker(m.cx, m.cy, m.color, m.img, m.angle, m.label));

  const clampedElapsed = Math.max(0, Math.min(elapsed, state.totalDuration));
  const totalSeconds = Math.floor(clampedElapsed / 1000);
  const m = String(Math.floor(totalSeconds / 60)).padStart(2, "0");
  const s = String(totalSeconds % 60).padStart(2, "0");
  ctx.fillStyle = "rgba(0,0,0,0.7)";
  ctx.fillRect(12, 12, 110, 30);
  ctx.fillStyle = "#fff";
  ctx.font = "14px monospace";
  ctx.fillText(`T+${m}:${s}`, 20, 32);
  progressTime.textContent = `T+${m}:${s}`;
  state.lastElapsed = clampedElapsed;

  if (!state.isScrubbing) setSliderFromElapsed(clampedElapsed);
  updateProgressBars(clampedElapsed);
  if (finished) {
    statusEl.textContent = "全部播放完畢";
    return;
  }
  if (scheduleNext) state.rafId = requestAnimationFrame(drawFrame);
}

function drawMarker(x, y, color, img, angle = 0, label = "") {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(angle);
  const size = 40;
  const useImg = img && img.complete && img.naturalWidth;
  if (useImg) {
    ctx.translate(-size / 2, -size / 2);
    ctx.drawImage(img, 0, 0, size, size);
  } else {
    ctx.fillStyle = color;
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(-8, -5);
    ctx.lineTo(6, -5);
    ctx.lineTo(6, -9);
    ctx.lineTo(16, 0);
    ctx.lineTo(6, 9);
    ctx.lineTo(6, 5);
    ctx.lineTo(-8, 5);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }
  if (label) {
    ctx.rotate(-angle);
    ctx.translate(size / 2 + 6, -4);
    const padding = 4;
    ctx.font = "12px sans-serif";
    const textWidth = ctx.measureText(label).width;
    ctx.fillStyle = "rgba(0,0,0,0.7)";
    ctx.fillRect(-padding, -12, textWidth + padding * 2, 18);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, 0, 0);
  }
  ctx.restore();
}

function drawGoalMarker(x, y) {
  const size = 22;
  if (goalImage.complete && goalImage.naturalWidth) {
    ctx.save();
    ctx.translate(x - size / 2, y - size / 2);
    ctx.drawImage(goalImage, 0, 0, size, size/2);
    ctx.restore();
  } else {
    ctx.save();
    ctx.translate(x, y);
    ctx.fillStyle = "#fff";
    ctx.strokeStyle = "rgba(0,0,0,0.5)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(-6, -6);
    ctx.lineTo(6, 0);
    ctx.lineTo(-6, 6);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }
}

function stopAnimation() {
  if (state.rafId) cancelAnimationFrame(state.rafId);
  state.rafId = null;
  state.startStamp = null;
  state.drivers.forEach((d) => (d.currentIndex = 0));
  updateProgressBars();
}

async function loadRace(race) {
  stopAnimation();
  setActiveRace(race.slug);
  state.currentRace = race;
  statusEl.textContent = `載入 ${race.label} 資料...`;
  codeEl.textContent = race.slug.toUpperCase();
  nameEl.textContent = `${race.label} 正賽`;
  noteEl.textContent = "20 位車手同步播放";

  try {
    const base = `driver_json/${race.slug}`;
    const posBase = `driver_positions/${race.slug}`;
    const resDrivers = await fetch(`${base}/drivers.json`);
    if (!resDrivers.ok) throw new Error(`drivers.json HTTP ${resDrivers.status}`);
    const driversJson = await resDrivers.json();

    const configs = driversJson.drivers.map((d, idx) => ({
      code: d.code,
      name: d.name,
      color: colorPalette[idx % colorPalette.length],
      note: `#${d.number}`,
    }));

    const found = [];
    const missingCodes = [];
    state.rankData = new Map();
    for (const cfg of configs) {
      try {
        const r = await fetch(`${base}/${cfg.code}.json`);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const json = await r.json();
        if (!json.points || !json.points.length) throw new Error("缺少 points");
        found.push({
          ...cfg,
          points: json.points,
          scaled: [],
          currentIndex: 0,
          totalDuration: json.points[json.points.length - 1].t,
        });
      } catch (err) {
        console.warn(`[SKIP] ${cfg.code}: ${err.message || err}`);
        missingCodes.push(cfg.code);
      }

      // 嘗試載入即時排名資料
      try {
        const rpos = await fetch(`${posBase}/${cfg.code}.json`);
        if (rpos.ok) {
          const recs = await rpos.json();
          if (Array.isArray(recs) && recs.length) {
            state.rankData.set(cfg.code, recs);
          }
        }
      } catch (err) {
        console.warn(`[SKIP rank] ${cfg.code}: ${err.message || err}`);
      }
    }

    if (!found.length) throw new Error("沒有可用的車手資料");

    state.drivers = found;
    const durations = found.map((d) => d.totalDuration);
    state.totalDuration = Math.max(...durations);
    const finishToleranceMs = 120000; // 慢於最晚完賽 120s 內仍視為完賽
    const finishRatioThreshold = 0.8; // 至少 80% 賽程，視為完賽候選
    state.drivers.forEach((d) => {
      d.finished =
        d.totalDuration >= state.totalDuration * finishRatioThreshold ||
        (state.totalDuration - d.totalDuration) <= finishToleranceMs;
    });
    const finishedDurations = state.drivers
      .filter((d) => d.finished)
      .map((d) => d.totalDuration);
    state.maxFinishedDuration = finishedDurations.length
      ? Math.max(...finishedDurations)
      : state.totalDuration;
    state.scaleMeta = computeScaleMeta(state.drivers);
    state.drivers.forEach((d) => {
      d.scaled = buildScaled(d.points, state.scaleMeta);
    });

    renderProgressBars(state.drivers);
    resizeCanvas();
    setSliderFromElapsed(0);
    const skipMsg = missingCodes.length ? `（略過 ${missingCodes.join(", ")}）` : "";
    statusEl.textContent = `播放中... ${skipMsg}`;
    state.startStamp = null;
    state.rafId = requestAnimationFrame(drawFrame);
  } catch (err) {
    statusEl.textContent = `載入失敗：${err.message}`;
    console.error(err);
  }
}

window.addEventListener("resize", resizeCanvas);
renderRaceButtons();
preloadMarkerImages();
if (races.length) loadRace(races[0]);

progressSlider.addEventListener("pointerdown", () => { state.isScrubbing = true; stopAnimation(); });
progressSlider.addEventListener("pointerup", () => {
  state.isScrubbing = false;
  applySliderElapsed();
});
progressSlider.addEventListener("input", () => {
  state.isScrubbing = true;
  applySliderElapsed(true);
});

canvas.addEventListener("mousemove", (e) => {
  const rect = canvas.getBoundingClientRect();
  state.hover = { active: true, x: e.clientX - rect.left, y: e.clientY - rect.top };
  if (!state.rafId && state.drivers.length) renderScene(state.lastElapsed, false);
});
canvas.addEventListener("mouseleave", () => {
  state.hover = { active: false, x: 0, y: 0 };
  if (!state.rafId && state.drivers.length) renderScene(state.lastElapsed, false);
});

function renderProgressBars(drivers) {
  progressEl.innerHTML = "";
  state.progressBars = new Map();
  drivers.forEach((d) => {
    const row = document.createElement("div");
    row.className = "progress-row";
    const dot = document.createElement("span");
    dot.className = "progress-dot";
    dot.style.background = d.color;
    const label = document.createElement("div");
    label.className = "progress-label";
    label.textContent = `${d.code} ${d.name}`;
    const bar = document.createElement("div");
    bar.className = "progress-bar";
    const fill = document.createElement("div");
    fill.className = "progress-fill";
    fill.style.background = d.color;
    bar.appendChild(fill);
    row.appendChild(dot);
    row.appendChild(label);
    row.appendChild(bar);
    progressEl.appendChild(row);
    state.progressBars.set(d.code, { fill, row });
  });
  updateProgressBars();
}

function getCurrentRank(code, elapsedMs) {
  const recs = state.rankData.get(code);
  if (!recs || !recs.length) return null;
  let rank = recs[recs.length - 1].pos;
  for (let i = 0; i < recs.length; i += 1) {
    if (recs[i].t > elapsedMs) {
      rank = i === 0 ? recs[0].pos : recs[i - 1].pos;
      break;
    }
  }
  return rank;
}

function updateProgressBars(elapsedMs = 0) {
  const rows = [];
  state.drivers.forEach((d) => {
    const entry = state.progressBars.get(d.code);
    if (!entry) return;
    const idx = Math.min(d.currentIndex, d.points.length - 1);
    const driverElapsed = d.points[idx]?.t ?? 0;
    const denom = d.finished ? (d.totalDuration || 1) : (state.maxFinishedDuration || 1);
    const effectiveElapsed = Math.min(driverElapsed, denom);
    const pct = Math.min(1, Math.max(0, effectiveElapsed / denom));
    entry.fill.style.width = `${(pct * 100).toFixed(1)}%`;
    const rank = getCurrentRank(d.code, elapsedMs);
    rows.push({
      row: entry.row,
      pct,
      code: d.code,
      driverElapsed,
      finished: d.finished,
      finishTime: d.totalDuration,
      rank: rank ?? 999,
    });
  });
  rows
    .sort((a, b) => {
      // 0) 即時排名：小的在上
      if (a.rank !== b.rank) return a.rank - b.rank;
      // 1) 進度高的在上
      if (b.pct !== a.pct) return b.pct - a.pct;
      // 2) 兩者都已完賽，時間短者在上
      if (a.finished && b.finished) return a.finishTime - b.finishTime || a.code.localeCompare(b.code);
      // 3) 都未完賽且進度相同，已跑時間多者在上
      if (!a.finished && !b.finished && a.driverElapsed !== b.driverElapsed) {
        return b.driverElapsed - a.driverElapsed;
      }
      return a.code.localeCompare(b.code);
    })
    .forEach(({ row }) => progressEl.appendChild(row));
}

function setSliderFromElapsed(elapsedMs) {
  if (!state.totalDuration) return;
  const pct = Math.min(100, Math.max(0, (elapsedMs / state.totalDuration) * 100));
  progressSlider.value = pct;
}

function applySliderElapsed(previewOnly = false) {
  if (!state.totalDuration || !state.drivers.length) return;
  const pct = Number(progressSlider.value) / 100;
  const elapsed = pct * state.totalDuration;
  state.drivers.forEach((d) => (d.currentIndex = 0));
  renderScene(elapsed, false);
  if (!previewOnly) {
    state.startStamp = performance.now() - elapsed / playbackSpeed;
    state.rafId = requestAnimationFrame(drawFrame);
  }
}

function resolveMarkerImage(color) {
  const key = color.replace("#", "").toLowerCase();
  if (markerCache.has(key)) return markerCache.get(key);
  const img = new Image();
  img.src = `marker/${key}.png`;
  markerCache.set(key, img);
  return img;
}
