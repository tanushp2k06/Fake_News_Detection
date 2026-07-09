/* ── State ─────────────────────────────────── */
let inferenceHistory = [];
let totalCount = 0;
let fakeCount   = 0;
let isAnalyzing = false;

const PIPELINE_MSGS = [
  "Tokenizing input with WordPiece tokenizer…",
  "Generating contextual embeddings [CLS]…",
  "Running transformer encoder layers…",
  "Computing multi-head self-attention…",
  "Classifying with softmax output layer…"
];

/* ── Init ──────────────────────────────────── */
document.addEventListener("DOMContentLoaded", () => {
  drawNeuralBg();
  const ta = document.getElementById("newsInput");
  ta.addEventListener("input", () => {
    const len = ta.value.length;
    document.getElementById("charCount").textContent = len;
    document.getElementById("tokenCount").textContent = "~" + Math.round(len / 4);
  });
});

/* ── Analyze ───────────────────────────────── */
async function analyzeNews() {
  const text = document.getElementById("newsInput").value.trim();
  if (!text) { shakeTextarea(); return; }
  if (isAnalyzing) return;

  isAnalyzing = true;
  document.getElementById("analyzeBtn").disabled = true;

  showLoading();
  animatePipeline();

  try {
    const result = await callModelAPI(text);
    displayResult(result, text);
  } catch (err) {
    console.error("API Error:", err);
    showError("Detection failed: " + (err.message || "Please try again."));
  } finally {
    isAnalyzing = false;
    document.getElementById("analyzeBtn").disabled = false;
    hideLoading();
  }
}

/* ── Claude API ────────────────────────────── */
async function callModelAPI(text) {
  const response = await fetch("http://127.0.0.1:8000/analyze", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ text: text })
  });

  if (!response.ok) {
    let errMsg = `HTTP ${response.status}`;
    try {
      const err = await response.json();
      errMsg = err.detail || err.message || errMsg;
    } catch (_) {}
    throw new Error(errMsg);
  }

  return await response.json();
}
/* ── Display Result ────────────────────────── */
function displayResult(data, originalText) {
  const panel = document.getElementById("resultPanel");
  panel.classList.remove("hidden");

  // Verdict
  const icons = { REAL: "✅", FAKE: "❌", UNCERTAIN: "⚠️" };
  const labels = { REAL: "Credible", FAKE: "Fake News", UNCERTAIN: "Unverified" };
  const subs   = {
    REAL:      "BERT classifier: HIGH CREDIBILITY",
    FAKE:      "BERT classifier: MISINFORMATION DETECTED",
    UNCERTAIN: "BERT classifier: INSUFFICIENT SIGNAL"
  };

  const vKey = data.verdict || "UNCERTAIN";
  document.getElementById("verdictIcon").textContent = icons[vKey];
  const lbl = document.getElementById("verdictLabel");
  lbl.textContent = labels[vKey];
  lbl.className = "verdict-label " + vKey.toLowerCase();
  document.getElementById("verdictSub").textContent = subs[vKey];

  // Score ring
  const arc = document.getElementById("scoreArc");
  const circumference = 226.2;
  const colorMap = { REAL: "#00e676", FAKE: "#ff4f6a", UNCERTAIN: "#ffd54f" };
  arc.style.stroke = colorMap[vKey];
  const numEl = document.getElementById("scoreNum");
  numEl.textContent = "0";
  panel.style.display = "";

  setTimeout(() => {
    const offset = circumference - (data.confidence / 100) * circumference;
    arc.style.transition = "stroke-dashoffset 1s cubic-bezier(0.16,1,0.3,1)";
    arc.style.strokeDashoffset = offset;
    animateNum(numEl, 0, data.confidence, 1000);
  }, 80);

  // Flags
  const flagsRow = document.getElementById("flagsRow");
  flagsRow.innerHTML = "";
  (data.flags || []).forEach(f => {
    const el = document.createElement("span");
    el.className = "flag " + (f.severity || "yellow");
    el.textContent = f.label;
    flagsRow.appendChild(el);
  });

  // Analysis
  document.getElementById("analysisText").textContent = data.summary || "";

  // Features
  const grid = document.getElementById("featuresGrid");
  grid.innerHTML = "";
  (data.features || []).forEach((feat, i) => {
    const level = feat.score >= 70 ? "high" : feat.score >= 40 ? "medium" : "low";
    const barColor = feat.score >= 70 ? "var(--fake)" : feat.score >= 40 ? "var(--uncertain)" : "var(--real)";
    const card = document.createElement("div");
    card.className = "feat-card";
    card.style.animationDelay = i * 0.06 + "s";
    card.innerHTML = `
      <div class="feat-name">${feat.name}</div>
      <div class="feat-score ${level}">${feat.score}</div>
      <div class="feat-bar-bg">
        <div class="feat-bar-fill" style="width:0;background:${barColor}" data-w="${feat.score}"></div>
      </div>
    `;
    grid.appendChild(card);
  });
  setTimeout(() => {
    grid.querySelectorAll(".feat-bar-fill").forEach(el => {
      el.style.width = el.dataset.w + "%";
    });
  }, 200);

  // Token highlights
  const suspicious = (data.suspicious_tokens || []).map(t => t.toLowerCase());
  const words = originalText.split(/(\s+)/);
  const highlighted = words.map(w => {
    const clean = w.replace(/[^a-z0-9]/gi, "").toLowerCase();
    if (suspicious.includes(clean)) {
      const level = vKey === "FAKE" ? "high" : "medium";
      return `<span class="tok ${level}">${escapeHTML(w)}</span>`;
    }
    return escapeHTML(w);
  }).join("");
  document.getElementById("tokenHighlight").innerHTML = highlighted || "<em>No tokens extracted.</em>";

  // Stats
  totalCount++;
  if (vKey === "FAKE") fakeCount++;
  document.getElementById("totalCount").textContent = totalCount;
  document.getElementById("fakeCount").textContent = fakeCount;

  // History
  addToHistory(data, originalText);

  // Scroll to result
  setTimeout(() => panel.scrollIntoView({ behavior: "smooth", block: "start" }), 100);
}

/* ── History ───────────────────────────────── */
function addToHistory(data, text) {
  inferenceHistory.unshift({ data, text });
  renderHistory();
}

function renderHistory() {
  const list = document.getElementById("historyList");
  if (!inferenceHistory.length) {
    list.innerHTML = `<p class="empty-state">No inferences yet — submit text above to begin.</p>`;
    return;
  }
  list.innerHTML = "";
  inferenceHistory.slice(0, 10).forEach((entry, i) => {
    const item = document.createElement("div");
    item.className = "history-item";
    item.style.animationDelay = i * 0.03 + "s";
    const truncated = entry.text.length > 90 ? entry.text.slice(0, 90) + "…" : entry.text;
    const vClass = (entry.data.verdict || "uncertain").toLowerCase();
    item.innerHTML = `
      <span class="hi-num">#${inferenceHistory.length - i}</span>
      <span class="hi-text">${escapeHTML(truncated)}</span>
      <div class="hi-right">
        <span class="hi-verdict ${vClass}">${entry.data.verdict}</span>
        <span class="hi-conf">${entry.data.confidence}%</span>
      </div>`;
    list.appendChild(item);
  });
}

function clearHistory() {
  inferenceHistory = [];
  totalCount = 0; fakeCount = 0;
  document.getElementById("totalCount").textContent = "0";
  document.getElementById("fakeCount").textContent = "0";
  renderHistory();
  document.getElementById("resultPanel").classList.add("hidden");
}

/* ── Loading / Pipeline ────────────────────── */
function showLoading() {
  document.getElementById("loadingSection").classList.remove("hidden");
  document.getElementById("resultPanel").classList.add("hidden");
}
function hideLoading() {
  document.getElementById("loadingSection").classList.add("hidden");
}

let pipelineTimer;
function animatePipeline() {
  // Reset all
  for (let i = 1; i <= 5; i++) {
    document.getElementById("pip" + i).classList.remove("active", "done");
    if (i < 5) document.getElementById("pline" + i).classList.remove("active");
  }
  const msg = document.getElementById("loadingMsg");
  let step = 0;
  function advance() {
    if (step > 0) {
      document.getElementById("pip" + step).classList.remove("active");
      document.getElementById("pip" + step).classList.add("done");
      if (step < 5) document.getElementById("pline" + step).classList.add("active");
    }
    step++;
    if (step > 5) return;
    document.getElementById("pip" + step).classList.add("active");
    msg.style.opacity = "0";
    setTimeout(() => {
      msg.textContent = PIPELINE_MSGS[step - 1];
      msg.style.opacity = "1";
    }, 150);
    pipelineTimer = setTimeout(advance, 800);
  }
  advance();
}

/* ── Neural Canvas BG ──────────────────────── */
function drawNeuralBg() {
  const canvas = document.getElementById("neuralCanvas");
  const ctx    = canvas.getContext("2d");
  let W, H, nodes;

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
    nodes = Array.from({ length: 40 }, () => ({
      x: Math.random() * W,
      y: Math.random() * H,
      vx: (Math.random() - 0.5) * 0.3,
      vy: (Math.random() - 0.5) * 0.3,
      r: Math.random() * 2 + 1
    }));
  }
  resize();
  window.addEventListener("resize", resize);

  function draw() {
    ctx.clearRect(0, 0, W, H);
    // Update
    nodes.forEach(n => {
      n.x += n.vx; n.y += n.vy;
      if (n.x < 0 || n.x > W) n.vx *= -1;
      if (n.y < 0 || n.y > H) n.vy *= -1;
    });
    // Edges
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dx = nodes[i].x - nodes[j].x;
        const dy = nodes[i].y - nodes[j].y;
        const d  = Math.sqrt(dx*dx + dy*dy);
        if (d < 130) {
          ctx.beginPath();
          ctx.strokeStyle = `rgba(0,245,212,${0.04 * (1 - d/130)})`;
          ctx.lineWidth = 0.6;
          ctx.moveTo(nodes[i].x, nodes[i].y);
          ctx.lineTo(nodes[j].x, nodes[j].y);
          ctx.stroke();
        }
      }
    }
    // Nodes
    nodes.forEach(n => {
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(0,245,212,0.25)";
      ctx.fill();
    });
    requestAnimationFrame(draw);
  }
  draw();
}

/* ── Helpers ───────────────────────────────── */
function animateNum(el, from, to, dur) {
  const start = performance.now();
  function step(now) {
    const p = Math.min((now - start) / dur, 1);
    const e = 1 - Math.pow(1 - p, 3);
    el.textContent = Math.round(from + (to - from) * e);
    if (p < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

function shakeTextarea() {
  const ta = document.getElementById("newsInput");
  ta.style.borderColor = "var(--fake)";
  ta.style.animation = "shake 0.4s ease";
  setTimeout(() => { ta.style.animation = ""; ta.style.borderColor = ""; }, 500);
}

function showError(msg) {
  hideLoading();
  const panel = document.getElementById("resultPanel");
  panel.classList.remove("hidden");
  panel.innerHTML = `
    <div style="display:flex;align-items:center;gap:12px;color:var(--fake);">
      <span style="font-size:1.5rem">⚠️</span>
      <div>
        <div style="font-weight:600;margin-bottom:4px">Detection Error</div>
        <div style="font-family:var(--mono);font-size:0.72rem;color:var(--muted)">${escapeHTML(msg)}</div>
      </div>
    </div>`;
}

function escapeHTML(s) {
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

/* ── Keyboard shortcut ─────────────────────── */
document.addEventListener("keydown", e => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") analyzeNews();
});
