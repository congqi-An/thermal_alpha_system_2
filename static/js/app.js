/* ═══════════════════════════════════════════════════════════
   热膨胀自动测量系统 — 前端交互
   ═══════════════════════════════════════════════════════════ */

// ── REST helpers ──
const API = (p, o) => fetch('/api/' + p, { headers: { 'Content-Type': 'application/json' }, ...o }).then(r => r.json());
const POST = (p, b) => API(p, { method: 'POST', body: JSON.stringify(b) });

// ── Canvas 绘图工具 (科研 HUD 风: 辉光曲线 + 端点光标 + 细网格) ──
function drawLine(canvasId, data, color) {
  const c = document.getElementById(canvasId);
  if (!c) return;
  const ctx = c.getContext('2d');
  const w = c.width, h = c.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#060b13'; ctx.fillRect(0, 0, w, h);
  // 细网格 (横线实 + 竖线虚, 示波器风)
  ctx.strokeStyle = 'rgba(45,212,245,0.05)'; ctx.lineWidth = 1;
  for (let i = 1; i < 4; i++) {
    const y = (h / 4) * i;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }
  ctx.setLineDash([2, 4]);
  for (let i = 1; i < 6; i++) {
    const x = (w / 6) * i;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
  }
  ctx.setLineDash([]);
  if (!data || data.length < 2) return;
  const min = Math.min(...data), max = Math.max(...data);
  const range = (max - min) || 1;
  const px = (i) => i / (data.length - 1) * w;
  const py = (v) => h - (v - min) / range * (h - 8) - 4;
  // 渐变填充 (先铺底)
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, color.replace(')', ',0.18)').replace('rgb', 'rgba'));
  grad.addColorStop(1, 'transparent');
  ctx.beginPath();
  data.forEach((v, i) => { i === 0 ? ctx.moveTo(px(i), py(v)) : ctx.lineTo(px(i), py(v)); });
  ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath();
  ctx.fillStyle = grad; ctx.fill();
  // 辉光数据线
  ctx.save();
  ctx.shadowColor = color; ctx.shadowBlur = 7;
  ctx.strokeStyle = color; ctx.lineWidth = 1.8; ctx.lineJoin = 'round';
  ctx.beginPath();
  data.forEach((v, i) => { i === 0 ? ctx.moveTo(px(i), py(v)) : ctx.lineTo(px(i), py(v)); });
  ctx.stroke();
  ctx.restore();
  // 端点光标 (当前值)
  const lx = px(data.length - 1), ly = py(data[data.length - 1]);
  ctx.save();
  ctx.shadowColor = color; ctx.shadowBlur = 10;
  ctx.fillStyle = '#fff';
  ctx.beginPath(); ctx.arc(lx, ly, 2.4, 0, Math.PI * 2); ctx.fill();
  ctx.restore();
}

function drawBars(canvasId, items, color) {
  const c = document.getElementById(canvasId);
  if (!c) return;
  const ctx = c.getContext('2d');
  const w = c.width, h = c.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#060b13'; ctx.fillRect(0, 0, w, h);
  if (!items || !items.length) return;
  const max = Math.max(...items) || 1;
  const bw = Math.min(w / items.length * 0.65, 28);
  const gap = (w - bw * items.length) / (items.length + 1);
  ctx.save();
  ctx.shadowColor = color; ctx.shadowBlur = 6;
  items.forEach((v, i) => {
    const bh = (v / max) * (h - 8);
    const x = gap + i * (bw + gap);
    const grad = ctx.createLinearGradient(0, h - bh, 0, h);
    grad.addColorStop(0, color);
    grad.addColorStop(1, color.replace(')', ',0.25)').replace('rgb', 'rgba'));
    ctx.fillStyle = grad;
    ctx.beginPath();
    if (ctx.roundRect) { ctx.roundRect(x, h - bh, bw, bh, [3, 3, 0, 0]); }
    else { ctx.rect(x, h - bh, bw, bh); }
    ctx.fill();
    // 顶部高亮盖帽
    ctx.fillStyle = 'rgba(255,255,255,0.55)';
    ctx.fillRect(x, Math.max(h - bh - 1, 0), bw, 1.5);
  });
  ctx.restore();
}

// ── 信号波形 (横向: 时间左→右, 幅度上下) ──
function drawSignal(canvasId, data, peaks, color) {
  const c = document.getElementById(canvasId);
  if (!c) return;
  const ctx = c.getContext('2d');
  const w = c.width, h = c.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#060b13'; ctx.fillRect(0, 0, w, h);
  if (!data || data.length < 2) {
    ctx.fillStyle = '#3a4a5a'; ctx.font = '10px "JetBrains Mono", monospace'; ctx.textAlign = 'center';
    ctx.fillText('等待测量数据', w / 2, h / 2); ctx.textAlign = 'left';
    return;
  }
  const min = Math.min(...data), max = Math.max(...data);
  const range = (max - min) || 1;
  // 零线 (点划线)
  const zeroY = h - (0 - min) / range * (h - 10) - 5;
  ctx.strokeStyle = 'rgba(45,212,245,0.15)'; ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(0, zeroY); ctx.lineTo(w, zeroY); ctx.stroke();
  ctx.setLineDash([]);
  // 辉光波形
  ctx.save();
  ctx.shadowColor = color; ctx.shadowBlur = 6;
  ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.lineJoin = 'round';
  ctx.beginPath();
  data.forEach((v, i) => {
    const x = i / (data.length - 1) * w;
    const y = h - (v - min) / range * (h - 10) - 5;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.restore();
  // 峰标记 (辉光十字准星)
  if (peaks && peaks.length) {
    ctx.save();
    ctx.shadowColor = '#f87171'; ctx.shadowBlur = 8;
    ctx.fillStyle = '#f87171';
    peaks.forEach(i => {
      if (i >= data.length) return;
      const x = i / (data.length - 1) * w;
      const y = h - (data[i] - min) / range * (h - 10) - 5;
      ctx.beginPath(); ctx.arc(x, y, 2.6, 0, Math.PI * 2); ctx.fill();
    });
    ctx.restore();
  }
}

// ── 诊断图: φ(s) 外推 + 通道分散度 (带轴刻度, 中文在 HTML 头部避免乱码) ──
const _CN_FONT = '"Microsoft YaHei","PingFang SC","Noto Sans CJK SC","WenQuanYi Micro Hei",sans-serif';

function _median(a) {
  const x = [...a].sort((p, q) => p - q); const m = Math.floor(x.length / 2);
  return x.length % 2 ? x[m] : (x[m - 1] + x[m]) / 2;
}
function _mad(a) { const m = _median(a); return _median(a.map(v => Math.abs(v - m))); }

// 通用: 深色背景 + 网格 + 轴框 + 数字刻度, 返回坐标映射
function _diagAxes(ctx, w, h, xMin, xMax, yMin, yMax, xticks, yticks) {
  const padL = 34, padR = 6, padT = 6, padB = 16;
  const xr = (xMax - xMin) || 1, yr = (yMax - yMin) || 1;
  const px = v => padL + (v - xMin) / xr * (w - padL - padR);
  const py = v => padT + (1 - (v - yMin) / yr) * (h - padT - padB);
  // 网格
  ctx.strokeStyle = 'rgba(45,212,245,0.10)'; ctx.lineWidth = 1;
  xticks.forEach(t => { const X = px(t); ctx.beginPath(); ctx.moveTo(X, padT); ctx.lineTo(X, h - padB); ctx.stroke(); });
  yticks.forEach(t => { const Y = py(t); ctx.beginPath(); ctx.moveTo(padL, Y); ctx.lineTo(w - padR, Y); ctx.stroke(); });
  // 轴框
  ctx.strokeStyle = 'rgba(148,163,184,0.35)';
  ctx.strokeRect(padL, padT, w - padL - padR, h - padT - padB);
  // 刻度标签 (纯数字, 中文放 HTML 头部)
  ctx.fillStyle = '#7dd3fc'; ctx.font = '9px ' + _CN_FONT;
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  xticks.forEach(t => ctx.fillText(String(t), px(t), h - padB + 2));
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  yticks.forEach(t => ctx.fillText(String(t), padL - 4, py(t)));
  ctx.textAlign = 'left';
  return { px, py, padL, padR, padT, padB };
}

// φ(s) 外推图: K 通道相位 vs 归一化 s, LS 拟合外推线, s=0 截距 = N
function drawPhiSChart(canvasId, d) {
  const c = document.getElementById(canvasId); if (!c) return;
  const ctx = c.getContext('2d'); const w = c.width, h = c.height;
  ctx.clearRect(0, 0, w, h); ctx.fillStyle = '#060b13'; ctx.fillRect(0, 0, w, h);
  if (!d || !d.s_norm || d.s_norm.length < 2 || !d.N_k || d.N_k.length < 2) {
    ctx.fillStyle = '#3a4a5a'; ctx.font = '10px ' + _CN_FONT; ctx.textAlign = 'center';
    ctx.fillText('等待定标…', w / 2, h / 2); ctx.textAlign = 'left'; return;
  }
  const xs = d.s_norm, ys = d.N_k, np = d.N_point;
  const n = xs.length, mx = xs.reduce((a, b) => a + b, 0) / n, my = ys.reduce((a, b) => a + b, 0) / n;
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) { num += (xs[i] - mx) * (ys[i] - my); den += (xs[i] - mx) ** 2; }
  const slope = den ? num / den : 0, icpt = my - slope * mx;
  const xMax = Math.max(1, ...xs);
  const yLo = Math.min(...ys, np) - 0.1, yHi = Math.max(...ys, np) + 0.1;
  const xticks = [0, 0.25, 0.5, 0.75, 1].filter(t => t <= xMax + 1e-6);
  const yticks = []; for (let k = 0; k <= 4; k++) yticks.push(+(yLo + (yHi - yLo) * k / 4).toFixed(2));
  const { px, py } = _diagAxes(ctx, w, h, 0, xMax, yLo, yHi, xticks, yticks);
  // 外推拟合线
  ctx.strokeStyle = 'rgba(52,211,153,0.55)'; ctx.setLineDash([4, 3]); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(px(0), py(icpt)); ctx.lineTo(px(xMax), py(slope * xMax + icpt)); ctx.stroke();
  ctx.setLineDash([]);
  // 散点 (K 通道)
  ctx.save(); ctx.shadowColor = '#22d3ee'; ctx.shadowBlur = 6; ctx.fillStyle = '#22d3ee';
  for (let i = 0; i < n; i++) { ctx.beginPath(); ctx.arc(px(xs[i]), py(ys[i]), 2.6, 0, 6.2832); ctx.fill(); }
  ctx.restore();
  // s=0 截距标记 (N_point)
  ctx.save(); ctx.shadowColor = '#fbbf24'; ctx.shadowBlur = 8; ctx.fillStyle = '#fbbf24';
  ctx.beginPath(); ctx.arc(px(0), py(np), 3.6, 0, 6.2832); ctx.fill(); ctx.restore();
  ctx.fillStyle = '#fbbf24'; ctx.font = 'bold 9px ' + _CN_FONT;
  ctx.fillText('N=' + Number(np).toFixed(2), px(0) + 5, Math.max(8, py(np) - 4));
}

// 通道分散度图: 各通道 N_k 柱 + 中位线 + ±MAD 带
function drawDispChart(canvasId, d) {
  const c = document.getElementById(canvasId); if (!c) return;
  const ctx = c.getContext('2d'); const w = c.width, h = c.height;
  ctx.clearRect(0, 0, w, h); ctx.fillStyle = '#060b13'; ctx.fillRect(0, 0, w, h);
  if (!d || !d.N_k || d.N_k.length < 2) {
    ctx.fillStyle = '#3a4a5a'; ctx.font = '10px ' + _CN_FONT; ctx.textAlign = 'center';
    ctx.fillText('等待定标…', w / 2, h / 2); ctx.textAlign = 'left'; return;
  }
  const ys = d.N_k, K = ys.length;
  const med = _median(ys), sd = _mad(ys) * 1.4826;
  const yLo = Math.min(...ys, med - sd) - 0.05, yHi = Math.max(...ys, med + sd) + 0.05;
  const xticks = ys.map((_, i) => i + 1);
  const yticks = []; for (let k = 0; k <= 4; k++) yticks.push(+(yLo + (yHi - yLo) * k / 4).toFixed(2));
  const { px, py, padL, padR, padT } = _diagAxes(ctx, w, h, 0.5, K + 0.5, yLo, yHi, xticks, yticks);
  // ±MAD 带
  ctx.fillStyle = 'rgba(248,113,113,0.12)';
  ctx.fillRect(px(0.5), py(med + sd), px(K + 0.5) - px(0.5), py(med - sd) - py(med + sd));
  // 中位线
  ctx.strokeStyle = '#f59e0b'; ctx.setLineDash([5, 3]); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(px(0.5), py(med)); ctx.lineTo(px(K + 0.5), py(med)); ctx.stroke(); ctx.setLineDash([]);
  // 柱
  const bw = (px(2) - px(1)) * 0.55;
  for (let i = 0; i < K; i++) {
    const x = px(i + 1), y = py(ys[i]), base = py(yLo);
    ctx.fillStyle = 'rgba(34,211,238,0.7)';
    ctx.fillRect(x - bw / 2, Math.min(y, base), bw, Math.max(1, Math.abs(base - y)));
  }
  // σ 标注
  ctx.fillStyle = '#f59e0b'; ctx.font = 'bold 9px ' + _CN_FONT; ctx.textAlign = 'right';
  ctx.fillText('σ=' + sd.toFixed(3), w - padR, padT + 4);
  ctx.textAlign = 'left';
}

// 温度曲线 (带刻度; 替代原 drawLine 的无刻度版本)
function drawTempChart(canvasId, data) {
  const c = document.getElementById(canvasId); if (!c) return;
  const ctx = c.getContext('2d'); const w = c.width, h = c.height;
  ctx.clearRect(0, 0, w, h); ctx.fillStyle = '#060b13'; ctx.fillRect(0, 0, w, h);
  if (!data || data.length < 2) {
    ctx.fillStyle = '#3a4a5a'; ctx.font = '10px ' + _CN_FONT; ctx.textAlign = 'center';
    ctx.fillText('等待数据', w / 2, h / 2); ctx.textAlign = 'left'; return;
  }
  const n = data.length;
  const tMin = Math.min(...data), tMax = Math.max(...data);
  const span = (tMax - tMin) || 1;
  const yLo = tMin - span * 0.15, yHi = tMax + span * 0.15;
  const xT = [...new Set([0, Math.floor(n / 4), Math.floor(n / 2), Math.floor(3 * n / 4), n - 1])];
  const yT = []; for (let k = 0; k <= 4; k++) yT.push(+(yLo + (yHi - yLo) * k / 4).toFixed(1));
  const { px, py, padL, padR, padT, padB } = _diagAxes(ctx, w, h, 0, n - 1, yLo, yHi, xT, yT);
  // 渐变填充 (铺到绘图区底部)
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, 'rgba(56,189,248,0.18)'); grad.addColorStop(1, 'transparent');
  ctx.beginPath();
  data.forEach((v, i) => { i === 0 ? ctx.moveTo(px(i), py(v)) : ctx.lineTo(px(i), py(v)); });
  ctx.lineTo(px(n - 1), h - padB); ctx.lineTo(px(0), h - padB); ctx.closePath();
  ctx.fillStyle = grad; ctx.fill();
  // 辉光数据线
  ctx.save(); ctx.shadowColor = '#38bdf8'; ctx.shadowBlur = 7;
  ctx.strokeStyle = '#38bdf8'; ctx.lineWidth = 1.8; ctx.lineJoin = 'round';
  ctx.beginPath();
  data.forEach((v, i) => { i === 0 ? ctx.moveTo(px(i), py(v)) : ctx.lineTo(px(i), py(v)); });
  ctx.stroke(); ctx.restore();
  // 当前值 (端点)
  const lx = px(n - 1), ly = py(data[n - 1]);
  ctx.fillStyle = '#fff';
  ctx.beginPath(); ctx.arc(lx, ly, 2.5, 0, 6.2832); ctx.fill();
  ctx.fillStyle = '#38bdf8'; ctx.font = 'bold 9px ' + _CN_FONT; ctx.textAlign = 'left';
  ctx.fillText(Number(data[n - 1]).toFixed(1) + '°C', lx + 6, Math.max(10, ly - 6));
  // 单位标注
  ctx.fillStyle = 'rgba(148,163,184,0.7)'; ctx.font = '9px ' + _CN_FONT; ctx.textAlign = 'right';
  ctx.fillText('°C', w - padR, padT + 4);
  ctx.textAlign = 'left';
}

// ── 面板 Tab 切换 ──
document.querySelectorAll('.ptab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.ptab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.panel-body').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('panel-' + tab.dataset.tab).classList.add('active');
  });
});

// ── 视频流 ──
const vCanvas = document.getElementById('video');
const vCtx = vCanvas.getContext('2d');
vCtx.fillStyle = '#060b13'; vCtx.fillRect(0, 0, vCanvas.width, vCanvas.height);
vCtx.fillStyle = '#3a4a5a'; vCtx.font = '13px "Rajdhani", "Segoe UI", sans-serif';
vCtx.textAlign = 'center';
vCtx.fillText('相机未取流 — 在右侧"相机"面板点击"开始取流"', vCanvas.width / 2, vCanvas.height / 2);
vCtx.textAlign = 'left';

let videoWS = null;
// 扫描特效开关: 收到真实帧 → 隐藏扫描线; 2s 无帧 (停流/断开) → 恢复
let lastFrameAt = 0;
const videoBox = document.querySelector('.video-container');
setInterval(() => {
  if (videoBox) videoBox.classList.toggle('streaming', Date.now() - lastFrameAt < 2000);
}, 500);
function openVideo() {
  try {
    videoWS = new WebSocket((location.protocol === 'https:' ? 'wss' : 'ws') + '://' + location.host + '/ws/video');
    videoWS.binaryType = 'arraybuffer';
    videoWS.onmessage = async (ev) => {
      const buf = new Uint8Array(ev.data);
      const mlen = (buf[0] << 24) | (buf[1] << 16) | (buf[2] << 8) | buf[3];
      const meta = JSON.parse(new TextDecoder().decode(buf.slice(4, 4 + mlen)));
      const blob = new Blob([buf.slice(4 + mlen)], { type: 'image/jpeg' });
      const img = await createImageBitmap(blob);
      vCanvas.width = img.width; vCanvas.height = img.height;
      vCtx.drawImage(img, 0, 0);
      lastFrameAt = Date.now();
      // overlay
      document.getElementById('ov-t').textContent =
        (meta.online === false) ? '--' : (meta.T != null ? meta.T.toFixed(1) : '--');
      document.getElementById('ov-n').textContent = meta.N != null ? meta.N.toFixed(1) : '--';
      document.getElementById('ov-dl').textContent = meta.delta_L_um != null ? meta.delta_L_um.toFixed(1) : '--';
      const _ovq = document.getElementById('ov-q');
      // Q = CNN 图像质量分 (逐帧实时, q_retrained), 非时序稳定度
      const _qval = (meta.quality != null ? meta.quality : meta.stability);
      _ovq.textContent = (_qval != null ? _qval.toFixed(2) : '--') + (meta.burst ? ' ⚠BURST' : '');
      _ovq.style.color = meta.burst ? '#f87171' : '';
      document.getElementById('ov-r0').textContent = meta.r0s && meta.r0s.length ? meta.r0s.join(',') : '--';
      const chip = document.getElementById('ov-state-chip');
      chip.textContent = meta.measuring ? 'MEASURING' : 'IDLE';
      chip.className = 'ov-chip' + (meta.measuring ? ' on' : '');
    };
    // 只在 onclose 里重连: onerror 后必触发 onclose,
    // 两处都调度会指数级膨胀连接 (重连风暴)
    videoWS.onclose = () => setTimeout(openVideo, 1500);
    videoWS.onerror = () => {};
  } catch (e) { setTimeout(openVideo, 1500); }
}
openVideo();

// ── 测量数据 WS ──
let tempData = [], nData = [];
let measureWS = null;
function openMeasure() {
  try {
    measureWS = new WebSocket((location.protocol === 'https:' ? 'wss' : 'ws') + '://' + location.host + '/ws/measure');
    measureWS.onmessage = (ev) => {
      const d = JSON.parse(ev.data);
      // 无温控(视频模拟/演示): T/PV/SV/MV 显示 '--'
      document.getElementById('pv').textContent = (d.online && d.T != null) ? d.T.toFixed(1) : '--';
      document.getElementById('sv').textContent = (d.online && d.sv != null) ? d.sv.toFixed(1) : '--';
      document.getElementById('mv').textContent = (d.online && d.mv != null) ? d.mv.toFixed(1) : '--';
      // 状态徽章
      const badgeTemp = document.getElementById('badge-temp');
      badgeTemp.className = 'badge' + (d.online ? ' online' : '');
      const badgeScan = document.getElementById('badge-scan');
      badgeScan.className = 'badge' + (d.scan_active ? ' active' : '');
      const badgeCnn = document.getElementById('badge-cnn');
      // 光心/ROI 实际来源: 两路全 CNN=绿, 任一路回退 CV=橙, 模型未加载=灰
      // 兼容旧后端: 无 center_source 字段时仅按 d.cnn 显示
      const cs = d.center_source, rs = d.r0_source;
      if (!d.cnn) {
        badgeCnn.className = 'badge';
        badgeCnn.innerHTML = '<i></i>CNN未加载';
      } else if (cs === undefined || cs === 'none') {
        badgeCnn.className = 'badge online';
        badgeCnn.innerHTML = '<i></i>CNN';
      } else if (cs === 'cnn' && rs === 'cnn') {
        badgeCnn.className = 'badge online';
        badgeCnn.innerHTML = '<i></i>CNN';
      } else {
        badgeCnn.className = 'badge active';
        badgeCnn.innerHTML = `<i></i>心:${cs === 'cnn' ? 'CNN' : 'CV'} R:${rs === 'cnn' ? 'CNN' : 'CV'}`;
      }
      // 判定模式按钮同步 (防多端不一致)
      if (d.center_mode) syncCenterModeBtn(d.center_mode);
      // 温度曲线 (带刻度)
      if (d.online) { tempData.push(d.T || 0); if (tempData.length > 80) tempData.shift(); }
      drawTempChart('temp-chart', tempData);
      // ── 诊断图: φ(s) 外推 + 通道分散度 (替换旧信号图/N 图) ──
      if (d.diag) {
        drawPhiSChart('phi-chart', d.diag);
        drawDispChart('disp-chart', d.diag);
        const npt = document.getElementById('diag-npoint'); if (npt) npt.textContent = Number(d.diag.N_point).toFixed(2);
        const sigEl = document.getElementById('diag-sigma');
        if (sigEl && d.diag.N_k && d.diag.N_k.length) sigEl.textContent = (_mad(d.diag.N_k) * 1.4826).toFixed(3);
      }
      // ── 反应层告警: Q 跳变时 seq 递增, 高于上次就插入对话气泡 ──
      if (d.alert && d.alert.seq) {
        if (!window._lastAlertSeq) window._lastAlertSeq = 0;
        if (d.alert.seq > window._lastAlertSeq) {
          window._lastAlertSeq = d.alert.seq;
          const div = document.createElement('div');
          div.className = 'chat-msg assistant alert-msg';
          div.innerHTML = '<div class="chat-bubble alert-bubble">'
            + '<span class="alert-badge">⚠️ 实验系统提醒</span><br>'
            + escapeHtml(d.alert.text) + '</div>';
          document.getElementById('chat-messages').appendChild(div);
          document.getElementById('chat-messages').scrollTop =
            document.getElementById('chat-messages').scrollHeight;
        }
      }
    };
    measureWS.onclose = () => setTimeout(openMeasure, 1500);
    measureWS.onerror = () => {};
  } catch (e) { setTimeout(openMeasure, 1500); }
}
openMeasure();

// ── 扫描 WS ──
let scanWS = null;
function openScan() {
  try {
    scanWS = new WebSocket((location.protocol === 'https:' ? 'wss' : 'ws') + '://' + location.host + '/ws/scan');
    scanWS.onmessage = (ev) => {
      const d = JSON.parse(ev.data);
      document.getElementById('scan-msg').textContent = (d.phase || '') + ' — ' + (d.message || '');
      const prog = d.total_seg ? (d.current_seg / d.total_seg * 100) : 0;
      document.getElementById('progress-bar').style.width = (d.phase === 'done' ? 100 : prog) + '%';
      // 段表
      const tbody = document.querySelector('#seg-table tbody');
      tbody.innerHTML = '';
      (d.segments || []).forEach(s => {
        const ci = s.alpha_ci ? `[${s.alpha_ci[0]}~${s.alpha_ci[1]}]` : '';
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${s.seg}</td><td>${s.T1}</td><td>${s.T2}</td><td>${s.N.toFixed(1)}</td><td>${s.alpha.toFixed(1)}<br><small>${ci}</small></td>`;
        tbody.appendChild(tr);
      });
      // 结果
      if (d.alpha_avg) {
        document.getElementById('alpha-avg').textContent = d.alpha_avg;
        document.getElementById('alpha-std').textContent = d.alpha_std;
        document.getElementById('alpha-err').textContent = d.error_pct;
      }
    };
    scanWS.onclose = () => setTimeout(openScan, 1500);
    scanWS.onerror = () => {};
  } catch (e) { setTimeout(openScan, 1500); }
}
openScan();

// ── 相机控制 ──
function camStart() {
  POST('camera/start').then(r => {
    if (r && r.grabbing) {   // 真实取流成功才变绿
      document.getElementById('cam-info').textContent = '取流中 ' + (r.width ? `${r.width}×${r.height}` : '');
      document.getElementById('badge-camera').className = 'badge online';
    } else {
      const err = (r && (r.detail || r.msg)) || '取流失败';
      document.getElementById('cam-info').textContent = '❌ ' + err;
      document.getElementById('badge-camera').className = 'badge';
    }
  }).catch(() => {
    document.getElementById('cam-info').textContent = '❌ 取流失败（服务不可达）';
    document.getElementById('badge-camera').className = 'badge';
  });
}
function camStop() {
  POST('camera/stop');
  document.getElementById('badge-camera').className = 'badge';
  document.getElementById('cam-info').textContent = '已停止';
}
function camCapture() {
  POST('camera/capture').then(r => {
    const el = document.getElementById('cap-count');
    if (r.ok) el.textContent = `(已存 ${r.total} 帧)`;
    else el.textContent = r.msg || '失败';
  });
}

// ── 视频模拟源 (加载本地视频当相机输入) ──
let videoStatusTimer = null;
function videoLoad() {
  const path = document.getElementById('video-path').value.trim();
  if (!path) { document.getElementById('video-status').textContent = '请先输入视频路径'; return; }
  POST('camera/video/load', { path: path, loop: false }).then(r => {
    if (!r.ok) { document.getElementById('video-status').textContent = r.msg || '加载失败'; return; }
    updateVideoStatus(r);
    startVideoPoll();
  });
}
function videoToggle() {
  // 按当前状态切换 播放/暂停
  fetch('/api/camera/video/status').then(r => r.json()).then(s => {
    if (!s.loaded) return;
    videoCtrl(s.playing ? 'pause' : 'play');
  });
}
function videoCtrl(action) {
  POST('camera/video/ctrl', { action: action }).then(r => {
    if (action === 'unload') {
      document.getElementById('video-status').textContent = '已卸载, 画面回到相机';
      if (videoStatusTimer) { clearInterval(videoStatusTimer); videoStatusTimer = null; }
      return;
    }
    if (r.ok) updateVideoStatus(r);
  });
}
function updateVideoStatus(s) {
  if (!s.loaded) return;
  const pct = s.n_frames > 0 ? Math.round(s.frame_idx / s.n_frames * 100) : 0;
  const st = s.finished ? '播完' : (s.playing ? '播放中' : '已暂停');
  document.getElementById('video-status').textContent =
    `${s.name} | ${s.size[0]}x${s.size[1]} @${s.fps}fps | ${st} ${s.frame_idx}/${s.n_frames} (${pct}%)`;
  document.getElementById('video-play-btn').textContent = s.playing ? '暂停' : '播放';
}
function startVideoPoll() {
  if (videoStatusTimer) clearInterval(videoStatusTimer);
  videoStatusTimer = setInterval(() => {
    fetch('/api/camera/video/status').then(r => r.json()).then(s => {
      if (!s.loaded) { clearInterval(videoStatusTimer); videoStatusTimer = null; return; }
      updateVideoStatus(s);
    }).catch(() => {});
  }, 1000);
}

// ── 光心/ROI 判定模式切换 (cnn ↔ cv 循环, 默认 cnn; 已去掉 auto 自动仲裁) ──
let centerMode = 'cnn';
const CENTER_MODE_LABEL = { cnn: '判定:CNN', cv: '判定:CV' };
function syncCenterModeBtn(mode) {
  if (mode === centerMode) return;
  centerMode = mode;
  const btn = document.getElementById('btn-center-mode');
  if (btn) {
    btn.textContent = CENTER_MODE_LABEL[mode] || `判定:${mode}`;
    btn.className = 'mode-btn manual';
  }
}
function cycleCenterMode() {
  const next = { cnn: 'cv', cv: 'cnn' }[centerMode] || 'cnn';
  POST('camera/center_mode', { mode: next }).then(r => {
    if (r.ok) syncCenterModeBtn(r.mode);
    else alert(r.msg || '切换失败');
  }).catch(() => alert('切换失败: 服务不可达'));
}

function reconnectCamera() {
  document.getElementById('badge-camera').className = 'badge';
  POST('camera/reconnect').then(r => {
    if (r.ok) {
      document.getElementById('badge-camera').className = 'badge online';
      document.getElementById('cam-info').textContent = `已重连 ${r.width}×${r.height}`;
    } else {
      document.getElementById('cam-info').textContent = '重连失败: ' + (r.msg || '');
    }
  }).catch(() => {});
}
function reconnectTemp() {
  document.getElementById('badge-temp').className = 'badge';
  POST('temp/reconnect').then(r => {
    if (r.ok) document.getElementById('badge-temp').className = 'badge online';
  }).catch(() => {});
}
// 参数钳制: 非法输入不再以 NaN→null 静默下发 (后端会忽略)
function setExp() {
  const v = +document.getElementById('exp').value;
  if (!isFinite(v) || isNaN(v)) return;
  POST('camera/exposure', { exposure_us: Math.round(Math.min(500000, Math.max(50, v))) });
}
function setGain() {
  const v = +document.getElementById('gain').value;
  if (!isFinite(v) || isNaN(v)) return;
  POST('camera/gain', { gain_db: Math.min(30, Math.max(0, v)) });
}
function setFps() {
  const v = +document.getElementById('fps').value;
  if (!isFinite(v) || isNaN(v)) return;
  POST('camera/framerate', { fps: Math.min(120, Math.max(1, v)) });
}
fetch('/api/camera/info').then(r => r.json()).then(d => {
  document.getElementById('cam-info').textContent = d.connected ? `已连接 ${d.width}×${d.height}` : '相机未连接';
  if (d.connected) document.getElementById('badge-camera').className = 'badge online';
}).catch(() => {});

// ── 温控 ──
function setSV() {
  // 修复: 安全钳制 20~60°C + 设后自动切 PID (原仅写 SV, 温控器处于手动模式时设了不加热,
  // 行为依赖历史状态; 与 Agent set_temperature 的 regulate_to 语义一致)
  const v = +document.getElementById('sv-input').value;
  if (!isFinite(v) || isNaN(v)) {
    document.getElementById('cam-info').textContent = '❌ SV 无效';
    return;
  }
  const clamped = Math.min(60, Math.max(20, v));
  if (clamped !== v) document.getElementById('cam-info').textContent =
    'SV 已钳制到 ' + clamped + '°C（安全范围 20~60）';
  POST('temp/set_sv', { sv: clamped }).then(r => {
    if (r.ok) {
      POST('temp/heat', { action: 'on' }).then(h => {
        document.getElementById('cam-info').textContent =
          h.ok ? `SV=${clamped}°C，PID 加热已启动` : 'SV 已设，但 PID 启动失败';
      });
    } else {
      document.getElementById('cam-info').textContent = r.msg || 'SV 设定失败（温控可能离线）';
    }
  });
}
function heat(a) {
  if (a === 'on') {
    // 修复: 输入框有 SV 时先写 SV 再启 PID (用户预期"填了 40 点加热 → 加热到 40")
    const v = +document.getElementById('sv-input').value;
    if (!isNaN(v) && isFinite(v)) {
      POST('temp/set_sv', { sv: Math.min(60, Math.max(20, v)) });
    }
  }
  // 危险操作确认: 满功率忽略 SV, 持续加热至 60°C 硬停
  if (a === 'full' && !confirm('满功率加热忽略设定温度，会持续加热到 60°C 自动硬停，确定？')) return;
  POST('temp/heat', { action: a });
}

// ── 扫描 ──
let scanMode = 'temp';   // temp=温度步进 / fringe=N步进 / free=即时计数
const SCAN_MODE_HINT = {
  temp: '按温度步长分段, 每段升温稳定后结算 α',
  fringe: '连续升温 T1→T2, 每累计 N 条条纹结算一段 (dT 实测)',
  free: '不控温, 启动即计数, 点停止时结算 N/dT/α',
};
function setScanMode(mode) {
  scanMode = mode;
  document.querySelectorAll('.scan-mode-tab').forEach(b =>
    b.classList.toggle('active', b.dataset.mode === mode));
  document.getElementById('scan-mode-hint').textContent = SCAN_MODE_HINT[mode];
  // 参数行显隐: temp 显 scan-p-temp, fringe 显 scan-p-fringe, free 全隐
  document.querySelectorAll('#panel-scan .param-item').forEach(el => {
    el.style.display = el.classList.contains('scan-p-' + mode) ? '' : 'none';
  });
}
function scanConfigBody() {
  const body = { mode: scanMode };
  if (scanMode !== 'free') {
    body.T_start = +document.getElementById('t_start').value;
    body.T_end = +document.getElementById('t_end').value;
  }
  if (scanMode === 'temp') {
    body.step = +document.getElementById('step').value;
    body.stabilize_s = +document.getElementById('stab').value;
  }
  if (scanMode === 'fringe') {
    body.n_step = +document.getElementById('n_step').value;
  }
  return body;
}
function scanConfig() {
  POST('scan/config', scanConfigBody()).then(d => {
    if (d.ok === false) { document.getElementById('scan-msg').textContent = d.msg; return; }
    const label = { temp: '段', fringe: '段(估计)', free: '段' }[d.mode] || '段';
    document.getElementById('scan-msg').textContent =
      `已配置 [${d.mode}]: ${d.total_segments} ${label}`;
  });
}
function scanStart() {
  // 启动前先同步当前面板模式/参数 (避免忘点"应用配置"用了旧模式)
  POST('scan/config', scanConfigBody()).then(() => POST('scan/start').then(r => {
    if (r.ok === false) document.getElementById('scan-msg').textContent = r.msg || '启动失败';
  }));
}
function scanStop() { POST('scan/stop'); }

// ── Agent 聊天界面 ──
let pendingImg = null;   // {b64, dataUrl} 附带画面快照 (点击拍摄, 所见即所析)

// Markdown + KaTeX 渲染 (占位符保护公式不被 marked 破坏)
function renderMarkdown(text) {
  if (!text) return '';
  const placeholders = [];
  let html = text;
  function stash(tex, display) {
    const idx = placeholders.length;
    if (typeof katex !== 'undefined') {
      try { placeholders.push(katex.renderToString(tex.trim(), { displayMode: display, throwOnError: false })); }
      catch(e) { placeholders.push(`<code>${tex}</code>`); }
    } else {
      placeholders.push(`<code>${tex}</code>`);
    }
    return `%%MATH_${idx}%%`;
  }
  // 1. 块级: $$...$$ 或 \[...\] 或独立行 [\n...\n]
  html = html.replace(/\$\$([\s\S]*?)\$\$/g, (_, tex) => stash(tex, true));
  html = html.replace(/\\\[([\s\S]*?)\\\]/g, (_, tex) => stash(tex, true));
  html = html.replace(new RegExp('\\n\\[\\s*\\n([\\s\\S]*?)\\n\\s*\\]\\n', 'g'), (_, tex) => stash(tex, true));
  // 2. 行内: $...$ 或 \(...\) 或 ( ... )
  html = html.replace(/\$([^\$\n]+?)\$/g, (_, tex) => stash(tex, false));
  html = html.replace(/\\\(([^)]*?)\\\)/g, (_, tex) => stash(tex, false));
  html = html.replace(/\(\s*([\\][a-zA-Z][^)]*?)\s*\)/g, (_, tex) => stash(tex, false));
  // 3. Markdown 渲染
  if (typeof marked !== 'undefined') {
    html = marked.parse(html, { breaks: true });
  } else {
    html = html.replace(/\n/g, '<br>');
  }
  // 4. 还原公式
  html = html.replace(/%%MATH_(\d+)%%/g, (_, idx) => placeholders[+idx] || '');
  return html;
}

function addChatMsg(role, content, imageUrl) {
  const container = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.className = 'chat-msg ' + role;
  let imgHtml = imageUrl ? `<img class="chat-img" src="${imageUrl}" alt="frame">` : '';
  let bubbleContent = role === 'assistant' ? renderMarkdown(content) : escapeHtml(content);
  div.innerHTML = `<div class="chat-bubble">${imgHtml}${bubbleContent}</div>`;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  return div;
}

// 深度思考原子图标 (与聊天区开关同款, 45° 交叉双轨道)
const ATOM_SVG = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" style="vertical-align:-2px;margin-right:3px"><circle cx="12" cy="12" r="1.8" fill="currentColor" stroke="none"/><ellipse cx="12" cy="12" rx="8.5" ry="3.4" transform="rotate(45 12 12)"/><ellipse cx="12" cy="12" rx="8.5" ry="3.4" transform="rotate(135 12 12)"/><circle cx="18" cy="18" r="1.4" fill="currentColor" stroke="none"/></svg>';

// Agent 完整回复渲染: 可折叠的深度思考 + 可折叠的工具调用轨迹 + 正式回答
function addAgentMsg(r) {
  const container = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.className = 'chat-msg assistant';
  let html = '';
  if (r.thinking) {
    html += `<details class="chat-think"><summary>${ATOM_SVG} 深度思考</summary>`
         +  `<div class="chat-think-body">${renderMarkdown(r.thinking)}</div></details>`;
  }
  if (r.tool_trace && r.tool_trace.length) {
    const items = r.tool_trace.map(t =>
      `<div class="tool-trace-item">`
      + `<div class="tool-trace-name">🔧 ${escapeHtml(t.name)} <code>${escapeHtml(JSON.stringify(t.args))}</code></div>`
      + `<div class="tool-trace-result">${escapeHtml(t.result)}</div>`
      + `</div>`).join('');
    html += `<details class="chat-tools"><summary>🔧 调用了 ${r.tool_trace.length} 个工具</summary>`
         +  `<div class="chat-tools-body">${items}</div></details>`;
  }
  html += `<div class="chat-answer">${renderMarkdown(r.text || '')}</div>`;
  div.innerHTML = `<div class="chat-bubble">${html}</div>`;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  return div;
}

function addTypingIndicator() {
  const container = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.className = 'chat-msg assistant typing';
  div.id = 'typing-indicator';
  div.innerHTML = '<div class="chat-bubble"><span class="dots"><i></i><i></i><i></i></span></div>';
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}
function removeTypingIndicator() {
  const el = document.getElementById('typing-indicator');
  if (el) el.remove();
}

function escapeHtml(t) {
  return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\n/g,'<br>');
}

// 附带当前画面: 点击即拍快照 → 预览 → 随下一条消息发送 (所见即所析)
async function attachFrame() {
  try {
    const resp = await fetch('/api/camera/snapshot');
    if (!resp.ok) {
      addChatMsg('assistant', '❌ 拍快照失败：无可用画面，请先在相机页开始取流。');
      return;
    }
    const blob = await resp.blob();
    const dataUrl = await new Promise(res => {
      const fr = new FileReader();
      fr.onload = () => res(fr.result);
      fr.readAsDataURL(blob);
    });
    pendingImg = { b64: dataUrl.split(',')[1], dataUrl };
    document.getElementById('attach-preview-img').src = dataUrl;
    document.getElementById('attach-preview').style.display = 'flex';
  } catch (e) {
    addChatMsg('assistant', '❌ 拍快照失败: ' + e);
  }
}
function clearAttach() {
  pendingImg = null;
  document.getElementById('attach-preview').style.display = 'none';
}

function clearChat() {
  document.getElementById('chat-messages').innerHTML = '';
  addChatMsg('assistant', '对话已清空。有什么可以帮你的？');
  POST('agent/clear_history');  // 同步清除后端记忆
}

// ── 深度思考开关 + 流式中断 (类似 DeepSeek) ──
let thinkEnabled = true;   // 深度思考: 开=true / 关=false
let streaming = false;     // 是否正在生成回答
let streamAbort = null;    // AbortController (流式中断)

function updateThinkToggleUI() {
  const btn = document.getElementById('think-toggle');
  if (!btn) return;
  btn.classList.toggle('off', !thinkEnabled);   // 亮=开, 暗=关 (无文字)
}

function initThinkToggle() {
  // 从 /api/agent/status 读取当前开关状态 (后端持久, 刷新不丢)
  fetch('/api/agent/status').then(r => r.json()).then(d => {
    if (d && typeof d.thinking === 'boolean') { thinkEnabled = d.thinking; updateThinkToggleUI(); }
  }).catch(() => {});
}

async function toggleThinking() {
  thinkEnabled = !thinkEnabled;
  updateThinkToggleUI();
  try { await POST('agent/config', { thinking: thinkEnabled }); }
  catch (e) { console.error('切换深度思考失败', e); }
}

function setStreamingUI(on) {
  streaming = on;
  const btn = document.getElementById('chat-send-btn');
  if (btn) {
    btn.textContent = on ? '⏹' : '➤';   // 运行中方块, 空闲箭头
    btn.classList.toggle('stop', on);
  }
}

function stopStream() {
  if (streamAbort) streamAbort.abort();
}

// ── 语音输入 (Web Speech API, 需麦克风权限; 不支持时隐藏按钮) ──
let voiceRec = null, voiceOn = false;
function initVoice() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  const btn = document.getElementById('voice-btn');
  if (!SR) { if (btn) btn.style.display = 'none'; return; }
  voiceRec = new SR();
  voiceRec.lang = 'zh-CN';
  voiceRec.interimResults = false;
  voiceRec.continuous = false;
  voiceRec.onresult = (e) => {
    let t = '';
    for (let i = e.resultIndex; i < e.results.length; i++) t += e.results[i][0].transcript;
    const input = document.getElementById('chat-input');
    if (input) { input.value = (input.value ? input.value + ' ' : '') + t; input.focus(); }
  };
  voiceRec.onend = () => setVoiceUI(false);
  voiceRec.onerror = () => setVoiceUI(false);
}
function toggleVoice() {
  if (!voiceRec) return;
  if (voiceOn) { try { voiceRec.stop(); } catch (e) {} setVoiceUI(false); }
  else { try { voiceRec.start(); setVoiceUI(true); } catch (e) { setVoiceUI(false); } }
}
function setVoiceUI(on) {
  voiceOn = on;
  const btn = document.getElementById('voice-btn');
  if (btn) btn.classList.toggle('active', on);
}

// ── 联网搜索开关 (🌐 亮=开, 暗=关; 开启后下一条消息联网) ──
let webSearchOn = false;
let pendingFile = null;   // 上传文件 {name, content}
function updateWebSearchUI() {
  const btn = document.getElementById('web-search-btn');
  if (btn) btn.classList.toggle('off', !webSearchOn);
}
function toggleWebSearch() {
  webSearchOn = !webSearchOn;
  updateWebSearchUI();
}

// ── 文件上传 ("+" 按钮): 图片→快照预览; 文本/PDF/DOCX→后端提取文本 ──
function handleFileSelect(ev) {
  const f = ev.target.files && ev.target.files[0];
  ev.target.value = '';
  if (!f) return;
  if (f.type.startsWith('image/')) {
    const reader = new FileReader();
    reader.onload = (e) => {
      const dataUrl = e.target.result;
      pendingImg = { b64: dataUrl.split(',')[1], dataUrl };
      document.getElementById('attach-preview-img').src = dataUrl;
      document.getElementById('attach-preview').style.display = 'flex';
    };
    reader.readAsDataURL(f);
    return;
  }
  const fd = new FormData();
  fd.append('file', f);
  fetch('/api/agent/upload', { method: 'POST', body: fd })
    .then(r => r.json())
    .then(res => {
      if (!res.ok) { addChatMsg('assistant', '❌ 文件上传失败: ' + (res.msg || '未知错误')); return; }
      if (res.type === 'image') {
        pendingImg = { b64: res.base64, dataUrl: 'data:image/jpeg;base64,' + res.base64 };
        document.getElementById('attach-preview-img').src = pendingImg.dataUrl;
        document.getElementById('attach-preview').style.display = 'flex';
        return;
      }
      pendingFile = { name: res.name, content: res.content || '' };
      document.getElementById('file-preview-name').textContent = res.name;
      document.getElementById('file-preview').style.display = 'flex';
    })
    .catch(e => addChatMsg('assistant', '❌ 文件上传失败: ' + e));
}
function clearFile() {
  pendingFile = null;
  document.getElementById('file-preview').style.display = 'none';
}

// 初始化开关 + 语音 + 联网 (页面加载后)
function initChatExtras() { initThinkToggle(); initVoice(); updateWebSearchUI(); }
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initChatExtras);
} else {
  initChatExtras();
}

function agentChat(preset) {
  if (streaming) return;   // 生成中不发送新消息 (停止走 stopStream)
  const input = document.getElementById('chat-input');
  const prompt = preset || input.value.trim();
  if (!prompt) return;
  input.value = '';

  // 显示用户消息 (附带快照/文件时展示)
  const img = pendingImg;
  const file = pendingFile;
  // 修复: 默认附当前相机帧 (与 README"附带当前画面"一致; 原仅含"条纹"的预设 chip 带图,
  // 手动输入"评估当前条纹质量"等视觉问题无图)
  const useFrame = !img && !file;
  const userText = file ? `[📄 ${file.name}] ${prompt}` : prompt;
  addChatMsg('user', userText, img ? img.dataUrl : null);
  if (img) clearAttach();
  if (file) clearFile();

  streamAgentChat(prompt, useFrame, img, file, webSearchOn);
}

// ── 智能滚动: 生成开始时伴随流式向下滚, 之后用户可自由上下滚动, 不强制跟随 ──
const chatScroller = (() => {
  let container = null, atBottom = true;
  const ensure = () => {
    if (!container) {
      container = document.getElementById('chat-messages');
      if (container) {
        container.addEventListener('scroll', () => {
          // 距底部 <80px 视为"在底部" → 保持跟随; 否则用户在上方阅读 → 停止跟随
          atBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 80;
        }, { passive: true });
      }
    }
    return container;
  };
  return {
    begin() {          // 开始生成: 强制到底 + 允许跟随
      if (ensure()) { atBottom = true; container.scrollTop = container.scrollHeight; }
    },
    follow() {         // 流式更新: 仅在"在底部"时跟随, 用户上翻即暂停
      if (ensure() && atBottom) container.scrollTop = container.scrollHeight;
    },
  };
})();

// 流式对话: SSE 增量渲染 (思考 → 工具 → 回答逐字出现)
function streamAgentChat(prompt, useFrame, img, file, webSearchOn) {
  const container = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.className = 'chat-msg assistant';
  div.innerHTML = `<div class="chat-bubble">`
    + `<details class="chat-think" style="display:none" open><summary>${ATOM_SVG} 深度思考中…</summary><div class="chat-think-body"></div></details>`
    + `<details class="chat-tools" style="display:none" open><summary>🔧 工具调用</summary><div class="chat-tools-body"></div></details>`
    + `<div class="chat-answer"><span class="dots"><i></i><i></i><i></i></span></div>`
    + `</div>`;
  container.appendChild(div);
  chatScroller.begin();

  const thinkBox = div.querySelector('.chat-think');
  const thinkBody = div.querySelector('.chat-think-body');
  const toolsBox = div.querySelector('.chat-tools');
  const toolsBody = div.querySelector('.chat-tools-body');
  const answerEl = div.querySelector('.chat-answer');
  let thinkText = '', ansText = '', toolCount = 0, mdTimer = null;

  const renderAnswer = () => {
    answerEl.innerHTML = ansText
      ? renderMarkdown(ansText)
      : '<span class="dots"><i></i><i></i><i></i></span>';
    chatScroller.follow();
  };
  const scheduleRender = () => {
    if (mdTimer) return;
    mdTimer = setTimeout(() => { mdTimer = null; renderAnswer(); }, 120);
  };

  const handle = (ev) => {
    if (ev.t === 'think') {
      thinkBox.style.display = '';
      thinkText += ev.d;
      thinkBody.textContent = thinkText;
      thinkBody.scrollTop = thinkBody.scrollHeight;
      chatScroller.follow();
    } else if (ev.t === 'tok') {
      ansText += ev.d;
      scheduleRender();
    } else if (ev.t === 'tool') {
      toolsBox.style.display = '';
      toolCount++;
      toolsBox.querySelector('summary').textContent = `🔧 调用工具中… (${toolCount})`;
      const item = document.createElement('div');
      item.className = 'tool-trace-item';
      item.dataset.name = ev.name;
      item.innerHTML = `<div class="tool-trace-name">🔧 ${escapeHtml(ev.name)} <code>${escapeHtml(JSON.stringify(ev.args || {}))}</code></div>`
        + `<div class="tool-trace-result">执行中…</div>`;
      toolsBody.appendChild(item);
      chatScroller.follow();
    } else if (ev.t === 'tresult') {
      const items = toolsBody.querySelectorAll('.tool-trace-item');
      for (let i = items.length - 1; i >= 0; i--) {
        if (items[i].dataset.name === ev.name) {
          items[i].querySelector('.tool-trace-result').textContent = ev.result;
          break;
        }
      }
    } else if (ev.t === 'error') {
      ansText += '\n❌ ' + ev.d;
      renderAnswer();
    } else if (ev.t === 'done') {
      if (mdTimer) { clearTimeout(mdTimer); mdTimer = null; }
      renderAnswer();
      thinkBox.removeAttribute('open');   // 完成后默认收起, 可点击展开
      toolsBox.removeAttribute('open');
      if (toolCount) toolsBox.querySelector('summary').textContent = `🔧 调用了 ${toolCount} 个工具`;
      if (thinkText) thinkBox.querySelector('summary').innerHTML = ATOM_SVG + ' 深度思考';
    }
  };

  // 生成中: 发送按钮 → 停止按钮, AbortController 支持中断
  const controller = new AbortController();
  streamAbort = controller;
  setStreamingUI(true);

  fetch('/api/agent/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, use_frame: useFrame,
                           image_base64: img ? img.b64 : null,
                           web_search: !!webSearchOn,
                           file_content: file ? file.content : null,
                           file_name: file ? file.name : null }),
    signal: controller.signal
  }).then(async resp => {
    if (!resp.ok) {
      ansText += '\n❌ 请求失败 (' + resp.status + ')';
      renderAnswer();
      return;
    }
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf('\n\n')) >= 0) {
        const frame = buf.slice(0, idx); buf = buf.slice(idx + 2);
        const line = frame.split('\n').find(l => l.startsWith('data: '));
        if (!line) continue;
        try { handle(JSON.parse(line.slice(6))); } catch (e) {}
      }
    }
  }).catch(e => {
    if (e.name === 'AbortError') {
      // 用户点"停止": 保留已渲染内容, 标记已停止
      if (mdTimer) clearTimeout(mdTimer);
      ansText += '\n\n⏹ 已停止生成';
      renderAnswer();
      thinkBox.removeAttribute('open'); toolsBox.removeAttribute('open');
      if (thinkText) thinkBox.querySelector('summary').innerHTML = ATOM_SVG + ' 深度思考';
      if (toolCount) toolsBox.querySelector('summary').textContent = `🔧 调用了 ${toolCount} 个工具`;
    } else {
      if (mdTimer) clearTimeout(mdTimer);
      ansText += '\n❌ 网络错误: ' + e;
      renderAnswer();
    }
  }).finally(() => {
    setStreamingUI(false);
    streamAbort = null;
  });
}

// ── 分阶段实验指导 (agent 驱动: 阶段卡片 + 讲解流 + 验证 + 软门控) ──
let guidePollTimer = null;
let curGuideMsg = null;   // 对话中当前阶段消息的 DOM 节点

// 顶部 agent 工具栏的上一步/下一步按钮 (在"开始实验"右侧)
function updateGuideToolbar(active, stepNo, total) {
  const prev = document.getElementById('btn-guide-prev');
  const next = document.getElementById('btn-guide-next');
  const rpt = document.getElementById('btn-guide-report');
  const upc = document.getElementById('btn-upload-coarse');
  if (!prev || !next) return;
  prev.style.display = active ? '' : 'none';
  next.style.display = active ? '' : 'none';
  // 粗调拍照/生成报告 已迁移到 guide card 内 (guideStepHtml), toolbar 中不再显示
  if (rpt) rpt.style.display = 'none';
  if (upc) upc.style.display = 'none';
  if (active) {
    prev.disabled = (stepNo || 1) <= 1;
    next.textContent = (stepNo && stepNo >= total) ? '完成实验 ▶' : '下一步 ▶';
  }
}

// ── 生成实验报告 PDF (第九阶段) ──
function generateReport() {
  if (streaming) return;
  addChatMsg('assistant', '📄 正在生成实验报告（实测数据 + LLM 结论 + VLM 评价 + 图表）…');
  fetch('/api/agent/report/generate', { method: 'POST' })
    .then(r => r.json())
    .then(res => {
      if (res.ok && res.pdf_url) {
        const link = ` <a href="${res.pdf_url}" target="_blank" class="report-link">📄 ${res.pdf_name}（${res.size_kb} KB${res.latex_used ? ' · LaTeX公式' : ' · mathtext降级'}）</a>`;
        const last = document.querySelector('#chat-messages .chat-msg:last-child .chat-bubble');
        if (last) last.innerHTML += link;
      } else {
        addChatMsg('assistant', '❌ ' + (res.msg || '报告生成失败'));
      }
    })
    .catch(e => addChatMsg('assistant', '❌ 报告生成失败: ' + e.message));
}

// ── 拍摄光路粗调光点照片 (进报告图1) — 自动取帧存档 + VLM 分析光点重合 ──
function captureCoarsePhoto() {
  if (streaming) return;
  addChatMsg('assistant', '📷 正在拍摄粗调光点画面并保存…');
  fetch('/api/agent/report/capture-coarse', { method: 'POST' })
    .then(r => r.json())
    .then(res => {
      if (res.ok) {
        const note = res.analysis ? '\n\n🤖 VLM 分析：' + res.analysis : '';
        addChatMsg('assistant', '✅ 粗调光点照片已保存，将出现在报告的图 1。' + note);
      } else {
        addChatMsg('assistant', '❌ 拍摄失败: ' + (res.msg || '未知错误'));
      }
    })
    .catch(e => addChatMsg('assistant', '❌ 拍摄失败: ' + e.message));
}

// 兼容旧手动上传 (report/asset), 保留
function uploadCoarsePhoto(ev) {
  const file = ev.target.files && ev.target.files[0];
  if (!file) return;
  const fd = new FormData();
  fd.append('file', file);
  fetch('/api/agent/report/asset', { method: 'POST', body: fd })
    .then(r => r.json())
    .then(res => {
      if (res.ok) addChatMsg('assistant', '📷 粗调光点照片已保存，将出现在报告的图 1。');
      else addChatMsg('assistant', '❌ 上传失败: ' + (res.msg || '未知错误'));
    })
    .catch(e => addChatMsg('assistant', '❌ 上传失败: ' + e.message));
  ev.target.value = '';
}

function guideStepHtml(step) {
  const v = step.verify || {};
  let vHtml = '';
  if (v.type && v.type !== 'none') {
    const label = v.passed === true ? '✅ 系统检测：已达标'
      : (v.passed === false ? '⏳ 系统检测：未达标' : '⏳ 尚未验证（点 🔍 验证）');
    vHtml = `<div class="guide-auto ${v.passed ? 'pass' : 'wait'}">${label}</div>`;
  }
  const actionsHtml = (step.actions || []).map(a => `<li>${escapeHtml(a)}</li>`).join('');
  const last = step.step_no >= step.total;
  const briefBtn = step.step_no ? '<button class="btn sm" onclick="streamGuideBriefing()">🔁 重新讲解</button>' : '';
  const verifyBtn = (v.type === 'vision') ? '<button class="btn sm" onclick="guideVerify()">🔍 验证</button>' : '';
  // 粗调拍照按钮 → 阶段3 (光路粗调); 生成报告按钮 → 阶段9 (报告与关机)
  const coarseBtn = (step.step_no === 3) ? '<button class="btn sm" onclick="captureCoarsePhoto()">📷 拍摄粗调图</button>' : '';
  const reportBtn = last ? '<button class="btn primary sm" onclick="generateReport()">📄 生成报告</button>' : '';
  // 隐藏的 file input (粗调上传用, 只放一个在页面里, id 在 index.html 已定义)
  return `
    <div class="guide-head">
      <span class="guide-progress">第 ${step.step_no} / ${step.total} 阶段</span>
      <span class="guide-title">${escapeHtml(step.title)}</span>
    </div>
    <div class="guide-objective">目标：${escapeHtml(step.objective)}</div>
    <ol class="guide-actions">${actionsHtml}</ol>
    <div class="guide-check">✔ 通过标准：${escapeHtml(step.checkpoint)}</div>
    ${vHtml}
    ${step.tips ? `<div class="guide-tips">💡 ${escapeHtml(step.tips)}</div>` : ''}
    <div class="guide-btns">
      <button class="btn sm" onclick="guidePrev()" ${step.step_no <= 1 ? 'disabled' : ''}>上一步</button>
      ${verifyBtn}
      ${coarseBtn}
      ${briefBtn}
      ${reportBtn}
      <button class="btn primary sm" onclick="guideNext()">${last ? '完成实验' : '下一步'}</button>
    </div>`;
}
function addGuideMsg(step) {
  const container = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.className = 'chat-msg assistant guide-pinned';   // guide-pinned: 置顶不随对话滚走
  div.innerHTML = `<div class="chat-bubble guide-bubble">${guideStepHtml(step)}</div>`;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  curGuideMsg = div;
}
function updateGuideMsg(step) {
  if (!curGuideMsg) { addGuideMsg(step); return; }
  const bubble = curGuideMsg.querySelector('.guide-bubble');
  if (bubble) bubble.innerHTML = guideStepHtml(step);
  const c = document.getElementById('chat-messages');
  c.scrollTop = c.scrollHeight;
}
function guideStart() {
  POST('agent/start', {}).then(r => {
    if (r.ok) {
      addChatMsg('assistant', '🎓 实验指导已开始！跟着下面的阶段卡片操作，完成一步后点"下一步"，遇到问题随时问我。');
      addGuideMsg(r.step);
      updateGuideToolbar(true, r.step.step_no, r.step.total);
      pollGuide();
      streamGuideBriefing();
    } else {
      addChatMsg('assistant', '❌ ' + (r.msg || '启动失败'));
    }
  });
}
function agentStop() {
  POST('agent/stop').then(() => {
    document.getElementById('agent-phase').textContent = 'IDLE';
    if (guidePollTimer) { clearInterval(guidePollTimer); guidePollTimer = null; }
    if (curGuideMsg) curGuideMsg.classList.remove('guide-pinned');   // 解除置顶
    curGuideMsg = null;
    updateGuideToolbar(false);
    addChatMsg('assistant', '⏹ 实验指导已结束。');
  });
}
function pollGuide() {
  if (guidePollTimer) clearInterval(guidePollTimer);
  guidePollTimer = setInterval(async () => {
    try {
      const g = await API('agent/guide');
      if (!g.ok) return;
      if (!g.active) { clearInterval(guidePollTimer); guidePollTimer = null; updateGuideToolbar(false); return; }
      updateGuideToolbar(true, g.step_no, g.total);
      if (!curGuideMsg) return;
      const cur = g.current || {};
      const v = cur.verify || {};
      // 刷新验证徽章 (不重建按钮, 不打断点击)
      const el = curGuideMsg.querySelector('.guide-auto');
      if (el && v.type && v.type !== 'none') {
        el.className = 'guide-auto ' + (v.passed ? 'pass' : 'wait');
        el.textContent = v.passed === true ? '✅ 系统检测：已达标'
          : (v.passed === false ? '⏳ 系统检测：未达标' : '⏳ 尚未验证（点 🔍 验证）');
      }
    } catch (e) {}
  }, 3000);
}
function guideNext() {
  POST('agent/guide/next').then(r => {
    if (!r.ok) {
      if (r.warn) {
        addChatMsg('assistant', '⚠️ ' + (r.msg || '本阶段尚未达标'));
        if (confirm('系统检测本阶段尚未达标（' + ((r.checks && r.checks.detail) || '') + '）。仍要继续下一步吗？')) {
          return POST('agent/guide/next', { force: true }).then(handleGuideNextResp);
        }
      } else {
        addChatMsg('assistant', '❌ ' + (r.msg || ''));
      }
      return;
    }
    handleGuideNextResp(r);
  });
}
function handleGuideNextResp(r) {
  if (!r || !r.ok) return;
  if (r.finished) {
    if (curGuideMsg) {
      curGuideMsg.querySelector('.guide-bubble').innerHTML =
        '🎉 <b>实验全部流程完成！</b>记得按最后一步整理收尾，也可以让我生成实验总结。';
      curGuideMsg.classList.remove('guide-pinned');   // 解除置顶, 恢复正常滚动
    }
    curGuideMsg = null;
    updateGuideToolbar(false);
  } else {
    updateGuideMsg(r.step_no ? r : r.step);
    updateGuideToolbar(true, r.step_no, r.total);
    streamGuideBriefing();
  }
}
function guidePrev() {
  POST('agent/guide/prev').then(r => {
    if (r.ok) API('agent/guide').then(g => {
      if (g.ok && g.active) {
        updateGuideToolbar(true, g.step_no, g.total);
        updateGuideMsg(g.current);
        streamGuideBriefing();
      }
    });
  });
}
function guideVerify() {
  // 即时反馈: VLM 分析需数秒, 先显示"正在验证"防止误以为无反应
  const pending = addChatMsg('assistant', '🔍 正在验证（AI 正在分析画面…）');
  // 禁用验证按钮, 防重复点击 (多次点击会反复调 VLM)
  const vbtn = curGuideMsg ? curGuideMsg.querySelector('.guide-btns button[onclick="guideVerify()"]') : null;
  if (vbtn) vbtn.disabled = true;
  POST('agent/guide/verify').then(r => {
    if (vbtn) vbtn.disabled = false;
    const bubble = pending && pending.querySelector ? pending.querySelector('.chat-bubble') : null;
    if (!r.ok) {
      if (bubble) bubble.innerHTML = '❌ ' + escapeHtml(r.msg || '');
      else addChatMsg('assistant', '❌ ' + (r.msg || ''));
      return;
    }
    const icon = r.passed === null ? '⏳' : (r.passed ? '✅' : '❌');
    const text = `${icon} 验证结果：${r.detail || ''}`;
    if (bubble) bubble.innerHTML = renderMarkdown(text);
    else addChatMsg('assistant', text);
    if (curGuideMsg) {
      const el = curGuideMsg.querySelector('.guide-auto');
      if (el) {
        el.className = 'guide-auto ' + (r.passed ? 'pass' : 'wait');
        el.textContent = icon + ' ' + (r.detail || '');
      }
    }
  }).catch(e => {
    if (vbtn) vbtn.disabled = false;
    const bubble = pending && pending.querySelector ? pending.querySelector('.chat-bubble') : null;
    if (bubble) bubble.innerHTML = '❌ 验证失败: ' + escapeHtml(e.message || '');
    else addChatMsg('assistant', '❌ 验证失败: ' + (e.message || ''));
  });
}
// agent 讲解流式进入聊天 (SSE /api/agent/guide/briefing)
function streamGuideBriefing() {
  const container = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.className = 'chat-msg assistant';
  div.innerHTML = '<div class="chat-bubble guide-brief"><span class="guide-brief-title">🎓 助手讲解中…</span><div class="guide-brief-body"></div></div>';
  container.appendChild(div);
  chatScroller.begin();
  const titleEl = div.querySelector('.guide-brief-title');
  const bodyEl = div.querySelector('.guide-brief-body');
  let buf = '';
  fetch('/api/agent/guide/briefing', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}'
  }).then(async resp => {
    if (!resp.ok || !resp.body) { bodyEl.textContent = '❌ 讲解生成失败'; return; }
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let b = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      b += dec.decode(value, { stream: true });
      let idx;
      while ((idx = b.indexOf('\n\n')) >= 0) {
        const frame = b.slice(0, idx); b = b.slice(idx + 2);
        const line = frame.split('\n').find(l => l.startsWith('data: '));
        if (!line) continue;
        let ev; try { ev = JSON.parse(line.slice(6)); } catch (e) { continue; }
        if (ev.t === 'tok') { buf += ev.d; bodyEl.textContent = buf; chatScroller.follow(); }
        else if (ev.t === 'error') { buf += '\n❌ ' + ev.d; bodyEl.textContent = buf; }
        else if (ev.t === 'done') {
          titleEl.textContent = '🎓 助手讲解';
          bodyEl.innerHTML = renderMarkdown(buf);
        }
      }
    }
  }).catch(e => { bodyEl.textContent = '❌ 网络错误: ' + e; });
}
// 对话图片点击放大 (灯箱)
document.getElementById('chat-messages').addEventListener('click', (e) => {
  if (e.target.tagName === 'IMG' && e.target.classList.contains('chat-img')) {
    document.getElementById('lightbox-img').src = e.target.src;
    document.getElementById('img-lightbox').style.display = 'flex';
  }
});

// ── 实验叙事 WebSocket ──
let narrWs = null;
function connectNarration() {
  if (narrWs && narrWs.readyState <= 1) return;
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  narrWs = new WebSocket(`${proto}://${location.host}/api/agent/ws/narration`);
  narrWs.onmessage = (ev) => {
    const d = JSON.parse(ev.data);
    if (d.type === 'narration') addNarration(d.text, d.type, d.time);
  };
  narrWs.onclose = () => { setTimeout(connectNarration, 5000); };
}
function addNarration(text, ntype, time) {
  const feed = document.getElementById('narration-feed');
  if (!feed) return;
  feed.style.display = 'block';
  const div = document.createElement('div');
  div.className = 'narration-item n-' + (ntype || 'system');
  div.innerHTML = `<span class="n-time">${time || ''}</span>${escapeHtml(text)}`;
  feed.appendChild(div);
  feed.scrollTop = feed.scrollHeight;
  // 最多保留 30 条
  while (feed.children.length > 30) feed.removeChild(feed.firstChild);
}
// 页面加载时连接叙事 WebSocket
connectNarration();

// ── 阶段感知快捷按钮 ──
const PHASE_PRESETS = {
  idle: [['开始实验', '开始实验'], ['光路指导', '光路怎么搭建和调节？'], ['α原理', '解释α=N·λ/2/(L₀·ΔT)原理']],
  setup: [['优化条纹', '评估当前条纹质量并优化'], ['开始扫描', '配置并启动温度扫描'], ['光路指导', '光路搭建指导']],
  heating: [['当前进度', '现在实验进展如何？'], ['温度多少', '当前温度多少？还在升温吗？'], ['条纹状态', '条纹现在什么状态？']],
  measuring: [['计数多少', '目前计了多少条纹？'], ['正常吗', '测量过程正常吗？有没有异常？'], ['暂停', '暂停当前操作']],
  stabilizing: [['快好了吗', '温度稳定了吗？还要等多久？'], ['当前N', '目前条纹计数是多少？']],
  complete: [['生成报告', '请生成完整实验报告'], ['结果如何', '实验结果怎么样？α是多少？'], ['再来一次', '重新开始一轮实验']],
  error: [['什么问题', '发生了什么异常？'], ['怎么处理', '建议怎么处理当前异常？'], ['停止', '停止实验']],
};
function updatePhaseButtons(phase) {
  const el = document.getElementById('chat-presets');
  if (!el) return;
  const presets = PHASE_PRESETS[phase] || PHASE_PRESETS.idle;
  el.innerHTML = presets.map(([label, msg]) => {
    // 修复: "开始实验" chip 应启动分阶段指导, 而不是把文本发给 LLM (修复双入口不一致)
    if (label === '开始实验' && msg === '开始实验') {
      return '<button class="chip" onclick="guideStart()">开始实验</button>';
    }
    return `<button class="chip" onclick="agentChat('${msg}')">${label}</button>`;
  }).join('');
}

// ── 实验阶段轮询（独立于 Agent 状态轮询） ──
setInterval(async () => {
  try {
    const p = await API('agent/phase');
    if (p.ok && p.phase) {
      const bar = document.getElementById('exp-phase-bar');
      const txt = document.getElementById('exp-phase-text');
      const names = {idle:'待命',setup:'准备中',heating:'升温中',measuring:'测量中',
                     stabilizing:'等稳定',analyzing:'分析中',complete:'已完成',error:'异常'};
      if (bar) bar.style.display = p.phase !== 'idle' ? 'flex' : 'none';
      if (txt) txt.textContent = names[p.phase] || p.phase;
      updatePhaseButtons(p.phase);
    }
  } catch(e) {}
}, 4000);

// Agent 状态轮询
let agentPollTimer = null;
function pollAgentStatus() {
  if (agentPollTimer) clearInterval(agentPollTimer);
  agentPollTimer = setInterval(async () => {
    try {
      const s = await API('agent/status');
      const phase = document.getElementById('agent-phase');
      const step = document.getElementById('agent-step');
      phase.textContent = (s.phase || 'IDLE').toUpperCase();
      phase.className = 'agent-phase ' + s.phase;
      step.textContent = '';
      // 日志
      if (s.log_tail && s.log_tail.length) {
        const logDiv = document.getElementById('agent-log');
        logDiv.style.display = 'block';
        logDiv.innerHTML = s.log_tail.map(e =>
          `<div class="log-entry ${e.role}"><b>${e.time}</b> ${e.tool ? '['+e.tool+']' : ''} ${(e.content||e.result||'').slice(0,80)}</div>`
        ).join('');
      }
      if (['done', 'error', 'stopped', 'idle'].includes(s.phase)) {
        clearInterval(agentPollTimer); agentPollTimer = null;
        if (s.has_report) {
          const rpt = await API('agent/report');
          if (rpt.report) addReportMsg(rpt.report);
        }
        if (s.phase === 'done') addChatMsg('assistant', '✅ 实验指导已完成。');
      }
    } catch (e) { /* ignore */ }
  }, 3000);
}

// ── 设置面板 ──
async function openSettings() {
  const m = document.getElementById('settings-modal');
  m.style.display = 'flex';
  try {
    const cfg = await fetch('/api/config').then(r => r.json());
    document.getElementById('set-exp').value = cfg.camera.exposure_us;
    document.getElementById('set-gain').value = cfg.camera.gain_db;
    document.getElementById('set-fps').value = cfg.camera.fps;
    document.getElementById('set-gamma').value = cfg.camera.gamma;
    document.getElementById('set-blk').value = cfg.camera.blacklevel;
    const qg = cfg.quality_gate ? cfg.quality_gate.enabled : true;
    document.getElementById('btn-quality-gate').textContent = qg ? '开' : '关';
    document.getElementById('set-mb-port').textContent = cfg.modbus.port;
    document.getElementById('set-mb-baud').textContent = cfg.modbus.baud;
    document.getElementById('set-mb-slave').textContent = cfg.modbus.slave;
    document.getElementById('set-t1').value = cfg.scan.T_start;
    document.getElementById('set-t2').value = cfg.scan.T_end;
    document.getElementById('set-step').value = cfg.scan.step;
    document.getElementById('set-stab').value = cfg.scan.stabilize_s;
    document.getElementById('set-band').value = cfg.scan.stable_band;
    document.getElementById('set-l0').textContent = cfg.algo.L0_mm;
    document.getElementById('set-lam').textContent = cfg.algo.lam_nm;
    document.getElementById('set-ref').textContent = cfg.algo.ref_alpha;
    document.getElementById('set-r0s').textContent = cfg.algo.count_r0s.join(',');
    document.getElementById('set-qt').textContent = cfg.algo.quality_threshold;
    // 摄像头高级参数
    const camSel = document.getElementById('cam-param');
    camSel.innerHTML = '';
    (cfg.camera_params || []).forEach(p => {
      camSel.innerHTML += `<option value="${p.name}" data-type="${p.type}">${p.name} — ${p.desc}</option>`;
    });
    // 温控寄存器
    const tempSel = document.getElementById('temp-reg');
    tempSel.innerHTML = '';
    (cfg.modbus_regs || []).forEach(r => {
      tempSel.innerHTML += `<option value="${r.name}">${r.name} — ${r.desc}</option>`;
    });
  } catch (e) { console.warn('config load', e); }
  fetch('/api/agent/health').then(r => r.json()).then(d => {
    document.getElementById('set-vlm-status').textContent = d.llm_online ? 'localhost:11434 (在线)' : '离线';
  }).catch(() => {});
}
function closeSettings() { document.getElementById('settings-modal').style.display = 'none'; }
function switchTab(tab) {
  document.querySelectorAll('.mtab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  const tabs = document.querySelectorAll('.mtab');
  const panes = ['camera', 'modbus', 'scan', 'algo', 'vlm'];
  const idx = panes.indexOf(tab);
  if (idx >= 0) tabs[idx].classList.add('active');
  document.getElementById('pane-' + tab).classList.add('active');
}
async function toggleQualityGate() {
  const btn = document.getElementById('btn-quality-gate');
  const next = btn.textContent !== '开';   // 当前"开"→ 切到关, 反之亦然
  const r = await fetch('/api/camera/quality_gate', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ enabled: next })
  }).then(x => x.json());
  if (r.quality_gate_enabled != null) btn.textContent = r.quality_gate_enabled ? '开' : '关';
}
function applyCameraExp() { POST('camera/exposure', { exposure_us: +document.getElementById('set-exp').value }); }
function applyCameraGain() { POST('camera/gain', { gain_db: +document.getElementById('set-gain').value }); }
function applyCameraFps() { POST('camera/framerate', { fps: +document.getElementById('set-fps').value }); }
function applyGamma() { POST('camera/gamma', { gamma: +document.getElementById('set-gamma').value }); }
function applyBlack() { POST('camera/blacklevel', { blacklevel: +document.getElementById('set-blk').value }); }
function applyAutoExp(on) { POST('camera/auto_exposure', { on }); }
function applySetSV() { POST('temp/set_sv', { sv: +document.getElementById('set-sv').value }); }
function applyScanCfg() {
  POST('scan/config', {
    T_start: +document.getElementById('set-t1').value,
    T_end: +document.getElementById('set-t2').value,
    step: +document.getElementById('set-step').value,
    stabilize_s: +document.getElementById('set-stab').value,
  }).then(d => alert('扫描配置: ' + d.total_segments + ' 段'));
}
function applyAiCfg() {
  POST('agent/config', {
    temperature: +document.getElementById('set-ai-temp').value,
    max_tokens: +document.getElementById('set-ai-maxtok').value,
  }).then(r => {
    if (r.ok) alert('AI 参数已应用');
    else alert('失败: ' + (r.msg || ''));
  });
}
function applyCamParam() {
  const sel = document.getElementById('cam-param');
  const name = sel.value;
  const type = sel.options[sel.selectedIndex].dataset.type;
  const val = +document.getElementById('cam-param-val').value;
  POST('camera/set', { param: name, value: val, type }).then(r => console.log('cam set', name, val, r));
}
function applyTempReg() {
  const name = document.getElementById('temp-reg').value;
  const val = +document.getElementById('temp-reg-val').value;
  POST('temp/write', { reg: name, value: val });
}
function readTempReg() {
  const name = document.getElementById('temp-reg').value;
  fetch('/api/temp/get?reg=' + name).then(r => r.json()).then(d => {
    document.getElementById('temp-reg-val').value = d.value ?? '';
  });
}
// 点击遮罩关闭
document.getElementById('settings-modal').addEventListener('click', e => {
  if (e.target.id === 'settings-modal') closeSettings();
});

// ── CSV 下载 ──
function downloadCSV() {
  fetch('/api/scan/result').then(r => r.json()).then(d => {
    if (!d.csv_path) { alert('无扫描结果'); return; }
    const name = d.csv_path.split(/[\\/]/).pop();
    fetch('/measurements/' + name).then(r => r.blob()).then(b => {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(b); a.download = name; a.click();
    }).catch(() => alert('CSV 路径: ' + d.csv_path));
  }).catch(() => alert('无结果'));
}

// 底部结果栏: 常显不折叠 (折叠功能已移除)

// ── 报告 Markdown 渲染 ──
function addReportMsg(md) {
  const container = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.className = 'chat-msg assistant';
  const html = typeof marked !== 'undefined' ? marked.parse(md) : escapeHtml(md);
  div.innerHTML = `<div class="chat-bubble report-bubble">${html}
    <div style="margin-top:8px;text-align:right">
      <button class="chip" onclick="downloadReport()">下载报告.md</button>
    </div></div>`;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  window._lastReport = md;
}
function downloadReport() {
  if (!window._lastReport) return;
  const blob = new Blob([window._lastReport], {type: 'text/markdown'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'experiment_report.md';
  a.click();
}

// 修复: pollAgentStatus 此前从未被调用, Agent 状态徽标/日志恒不更新 (页面加载后启动轮询)
pollAgentStatus();
