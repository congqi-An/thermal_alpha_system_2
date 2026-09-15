/* ============================================================
   实验 Agent · 手机前端逻辑
   原生 JS + Canvas, 无框架无构建
   数据流: /ws/measure (常开) /ws/scan (常开) /ws/video (监控区展开时)
   ============================================================ */
(function () {
  'use strict';

  // ───────────────────────── 工具函数 ─────────────────────────

  async function api(url, method, body) {
    const opts = { method: method || 'GET', headers: {} };
    if (body !== undefined) {
      opts.headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(body);
    }
    const r = await fetch(url, opts);
    try { return await r.json(); } catch (e) { return { ok: false, msg: '响应解析失败' }; }
  }

  function $(id) { return document.getElementById(id); }

  // 深度思考原子图标 (与聊天区开关同款, 45° 交叉双轨道)
  const ATOM_SVG = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" style="vertical-align:-2px;margin-right:3px"><circle cx="12" cy="12" r="1.8" fill="currentColor" stroke="none"/><ellipse cx="12" cy="12" rx="8.5" ry="3.4" transform="rotate(45 12 12)"/><ellipse cx="12" cy="12" rx="8.5" ry="3.4" transform="rotate(135 12 12)"/><circle cx="18" cy="18" r="1.4" fill="currentColor" stroke="none"/></svg>';

  function escapeHtml(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  /** Markdown + KaTeX 渲染 (与桌面端同思路: 摘除 math → marked → 还原) */
  function renderMarkdown(text) {
    const math = [];
    if (typeof marked === 'undefined') return escapeHtml(text);
    let t = text
      .replace(/\$\$([\s\S]+?)\$\$/g, (m, e) => { math.push('$$' + e + '$$'); return '@@M' + (math.length - 1) + '@@'; })
      .replace(/\$([^$\n]+?)\$/g, (m, e) => { math.push('$' + e + '$'); return '@@M' + (math.length - 1) + '@@'; });
    let html = marked.parse(t);
    html = html.replace(/@@M(\d+)@@/g, (m, i) => {
      const f = math[+i];
      if (typeof katex === 'undefined') return '<code>' + escapeHtml(f) + '</code>';
      try {
        if (f.startsWith('$$')) return katex.renderToString(f.slice(2, -2), { displayMode: true, throwOnError: false });
        return katex.renderToString(f.slice(1, -1), { throwOnError: false });
      } catch (e) { return '<code>' + escapeHtml(f) + '</code>'; }
    });
    return html;
  }

  /** 图片前端压缩: canvas 缩放 + JPEG, 返回 data URI */
  function compressImage(file, maxDim, quality) {
    maxDim = maxDim || 1024; quality = quality || 0.8;
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = () => reject(new Error('读取文件失败'));
      reader.onload = (e) => {
        const img = new Image();
        img.onerror = () => reject(new Error('图片解码失败'));
        img.onload = () => {
          const scale = Math.min(1, maxDim / Math.max(img.width, img.height));
          const canvas = document.createElement('canvas');
          canvas.width = Math.max(1, Math.round(img.width * scale));
          canvas.height = Math.max(1, Math.round(img.height * scale));
          canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height);
          resolve(canvas.toDataURL('image/jpeg', quality));
        };
        img.src = e.target.result;
      };
      reader.readAsDataURL(file);
    });
  }

  function fmt(v, d) { return (v === null || v === undefined || isNaN(v)) ? '--' : Number(v).toFixed(d === undefined ? 1 : d); }

  // ───────────────────────── 状态 ─────────────────────────

  const state = {
    pendingImage: null,      // {dataUrl, name}
    sending: false,
    scanning: false,
    centerMode: 'cnn',
    scanMode: 'temp',
    nBuffer: [],
    tBuffer: [],
    signalBuffer: [],
    agentPhase: 'idle',
    videoWS: null,
    videoOpen: false,
  };

  // ───────────────────────── WebSocket 公共 ─────────────────────────

  function wsConnect(path, onMessage, onOpen, onClose) {
    let ws;
    function connect() {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws';
      ws = new WebSocket(`${proto}://${location.host}${path}`);
      // 修复: /ws/video 发二进制帧 (meta+JPEG), 默认 binaryType='blob' 会使
      // new Uint8Array(e.data) 抛 TypeError, 监控画面静默失效 (与桌面端 app.js 对齐)
      ws.binaryType = 'arraybuffer';
      ws.onopen = () => onOpen && onOpen();
      ws.onclose = () => { onClose && onClose(); setTimeout(connect, 1500); };
      ws.onerror = () => { try { ws.close(); } catch (e) {} };
      ws.onmessage = onMessage;
    }
    connect();
    return {
      close: () => { try { ws.close(); } catch (e) {} },
    };
  }

  // ───────────────────────── /ws/measure (常开) ─────────────────────────

  function updateStatusBar(d) {
    const T = $('stat-T'), N = $('stat-N'), Q = $('stat-Q');
    T.textContent = fmt(d.T);
    N.textContent = fmt(d.N, 1);
    T.classList.toggle('warn', d.online === false);
    Q.textContent = fmt(d.quality, 2);
    Q.className = 'stat-value';
    if (d.quality !== undefined) {
      if (d.quality < 0.4) Q.classList.add('danger');
      else if (d.quality < 0.6) Q.classList.add('warn');
    }
    // 温控徽标
    $('badge-temp').className = 'badge ' + (d.online ? 'online' : 'offline');
    // 相机徽标: 修复—原错误跟随温控 online; 用 center_source (画面在分析中 = 帧源在线)
    const frameOn = !!d.center_source && d.center_source !== 'none';
    $('badge-cam').className = 'badge ' + (frameOn ? 'online' : 'offline');
    // 温控子区读数 (修复: temp-readout 此前从未填充, 一直空白)
    const tr = $('temp-readout');
    if (tr) {
      tr.innerHTML =
        '<div class="ro"><span class="k">PV</span><span class="v">' + fmt(d.T) + ' °C</span></div>' +
        '<div class="ro"><span class="k">SV</span><span class="v">' + fmt(d.sv) + ' °C</span></div>' +
        '<div class="ro"><span class="k">MV</span><span class="v">' + fmt(d.mv) + ' %</span></div>' +
        '<div class="ro"><span class="k">状态</span><span class="v">' + (d.online ? '在线' : '离线') + '</span></div>';
    }
    // 监控读数
    updateMonitorReadout(d);
    // 诊断图: φ(s) 外推 + 通道分散度 (替换旧信号图/N 图)
    if (d.diag) drawDiagMobile(d.diag);
    // 温度曲线 (带刻度)
    pushTemp(d.T);
    // 测量读数
    updateMeasureReadout(d);
    // 扫描进行中标记
    if (d.scan_active !== undefined && d.scan_active !== state.scanning) {
      state.scanning = d.scan_active;
      $('scan-message').textContent = d.scan_active ? '扫描进行中…' : '';
    }
    // ── 反应层告警: Q 跳变时 seq 递增, 高于上次就插入对话气泡 ──
    if (d.alert && d.alert.seq) {
      if (!window._lastAlertSeq) window._lastAlertSeq = 0;
      if (d.alert.seq > window._lastAlertSeq) {
        window._lastAlertSeq = d.alert.seq;
        const bubble = addMessage('assistant',
          '<span class="alert-badge">⚠️ 实验系统提醒</span><br>' + renderMarkdown(d.alert.text));
        bubble.classList.add('alert-bubble');
      }
    }
  }

  function updateMonitorReadout(d) {
    const el = $('monitor-readout');
    if (!el) return;
    el.innerHTML =
      '<div class="ro"><span class="k">温度 T</span><span class="v">' + fmt(d.T) + ' °C</span></div>' +
      '<div class="ro"><span class="k">条纹 N</span><span class="v">' + fmt(d.N, 1) + '</span></div>' +
      '<div class="ro"><span class="k">质量 Q</span><span class="v">' + fmt(d.quality, 2) + '</span></div>' +
      '<div class="ro"><span class="k">光心</span><span class="v">' + (d.center_source || '-') + '</span></div>' +
      '<div class="ro"><span class="k">r0 源</span><span class="v">' + (d.r0_source || '-') + '</span></div>' +
      '<div class="ro"><span class="k">稳定度</span><span class="v">' + fmt(d.stability, 2) + (d.burst ? ' ⚠' : '') + '</span></div>';
  }

  function updateMeasureReadout(d) {
    const el = $('measure-readout');
    if (!el) return;
    el.innerHTML =
      '<div class="ro"><span class="k">测量中</span><span class="v">' + (d.measuring ? '是' : '否') + '</span></div>' +
      '<div class="ro"><span class="k">N</span><span class="v">' + fmt(d.N, 1) + '</span></div>' +
      '<div class="ro"><span class="k">ΔL μm</span><span class="v">' + fmt(d.delta_L_um, 2) + '</span></div>';
  }

  wsConnect('/ws/measure', (e) => { try { updateStatusBar(JSON.parse(e.data)); } catch (err) {} });

  // ───────────────────────── /ws/scan (常开) ─────────────────────────

  function updateScanFromStatus(s) {
    state.scanning = !!s.active;
    $('scan-message').textContent = s.message || '';
    const pct = s.total_seg ? (s.current_seg / s.total_seg) * 100 : 0;
    $('scan-progress-bar').style.width = (s.active ? pct : 0) + '%';
    if (s.alpha_avg) $('stat-alpha').textContent = fmt(s.alpha_avg);
    else if (s.alpha_avg === 0) $('stat-alpha').textContent = '--';
    // 扫描完成或停止 → 自动刷新结果区（含误差预算）
    if (!s.active && (s.phase === 'done' || s.phase === 'stopped') && (s.segments || []).length > 0) {
      refreshResult();
    }
  }

  wsConnect('/ws/scan', (e) => {
    try { updateScanFromStatus(JSON.parse(e.data)); } catch (err) {}
  });

  // ───────────────────────── 图表 (Canvas) ─────────────────────────

  function setupChart(canvas) {
    // 修复: 仅尺寸变化时才重设 canvas (原每次赋值清空画布+强制同步布局,
    // ws/measure 0.2s 一次 × 3 图表 → 持续卡顿)
    const w = canvas.clientWidth * 2, h = canvas.clientHeight * 2;
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w; canvas.height = h;
    }
  }
  function drawLineChart(canvas, data, color, label) {
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    if (!data.length) return;
    ctx.strokeStyle = color || '#6366f1';
    ctx.lineWidth = 2;
    ctx.beginPath();
    const n = data.length;
    for (let i = 0; i < n; i++) {
      const x = (i / (n - 1 || 1)) * W;
      const min = Math.min(...data), max = Math.max(...data);
      const span = (max - min) || 1;
      const y = H - ((data[i] - min) / span) * (H - 6) - 3;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  function pushSeries(key, v) {
    if (v === undefined || v === null || isNaN(v)) return;
    const buf = state[key + 'Buffer'];
    buf.push(v);
    if (buf.length > 200) buf.shift();
    drawLineChart($('chart-' + key), buf, '#f59e0b');
  }
  // 温度曲线 (带刻度); 节流 0.5s — ws/measure 0.2s 推送, 全速重绘没必要
  let lastTempDraw = 0;
  function pushTemp(v) {
    if (v === undefined || v === null || isNaN(v)) return;
    const buf = state.tBuffer;
    buf.push(v);
    if (buf.length > 200) buf.shift();
    const c = $('chart-temp'); if (!c) return;
    const now = performance.now();
    if (now - lastTempDraw < 500) return;
    lastTempDraw = now;
    setupChart(c);
    drawTempMobile(c, buf);
  }

  // ── 诊断图: φ(s) 外推 + 通道分散度 (带刻度; 中文放 HTML 标题避免乱码) ──
  const _CN_FONT = '"Microsoft YaHei","PingFang SC","Noto Sans CJK SC","WenQuanYi Micro Hei",sans-serif';
  function _median(a) {
    const x = [...a].sort((p, q) => p - q); const m = Math.floor(x.length / 2);
    return x.length % 2 ? x[m] : (x[m - 1] + x[m]) / 2;
  }
  function _mad(a) { const m = _median(a); return _median(a.map(v => Math.abs(v - m))); }

  function _diagAxes(ctx, W, H, xMin, xMax, yMin, yMax, xticks, yticks) {
    const s = Math.max(1, H / 120);            // 高分屏字体缩放
    const padL = 34 * s, padR = 6 * s, padT = 6 * s, padB = 16 * s;
    const xr = (xMax - xMin) || 1, yr = (yMax - yMin) || 1;
    const px = v => padL + (v - xMin) / xr * (W - padL - padR);
    const py = v => padT + (1 - (v - yMin) / yr) * (H - padT - padB);
    ctx.strokeStyle = 'rgba(45,212,245,0.10)'; ctx.lineWidth = 1;
    xticks.forEach(t => { const X = px(t); ctx.beginPath(); ctx.moveTo(X, padT); ctx.lineTo(X, H - padB); ctx.stroke(); });
    yticks.forEach(t => { const Y = py(t); ctx.beginPath(); ctx.moveTo(padL, Y); ctx.lineTo(W - padR, Y); ctx.stroke(); });
    ctx.strokeStyle = 'rgba(148,163,184,0.35)';
    ctx.strokeRect(padL, padT, W - padL - padR, H - padT - padB);
    ctx.fillStyle = '#7dd3fc'; ctx.font = Math.round(9 * s) + 'px ' + _CN_FONT;
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    xticks.forEach(t => ctx.fillText(String(t), px(t), H - padB + 2));
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    yticks.forEach(t => ctx.fillText(String(t), padL - 4, py(t)));
    ctx.textAlign = 'left';
    return { px, py, padL, padR, padT, padB };
  }

  function drawPhiSMobile(canvas, d) {
    const ctx = canvas.getContext('2d'); const W = canvas.width, H = canvas.height;
    const s = Math.max(1, H / 120);
    ctx.clearRect(0, 0, W, H); ctx.fillStyle = '#0b1120'; ctx.fillRect(0, 0, W, H);
    if (!d || !d.s_norm || d.s_norm.length < 2 || !d.N_k || d.N_k.length < 2) {
      ctx.fillStyle = '#3a4a5a'; ctx.font = Math.round(10 * s) + 'px ' + _CN_FONT; ctx.textAlign = 'center';
      ctx.fillText('等待定标…', W / 2, H / 2); ctx.textAlign = 'left'; return;
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
    const { px, py } = _diagAxes(ctx, W, H, 0, xMax, yLo, yHi, xticks, yticks);
    ctx.strokeStyle = 'rgba(52,211,153,0.55)'; ctx.setLineDash([4, 3]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(px(0), py(icpt)); ctx.lineTo(px(xMax), py(slope * xMax + icpt)); ctx.stroke(); ctx.setLineDash([]);
    ctx.save(); ctx.shadowColor = '#22d3ee'; ctx.shadowBlur = 6; ctx.fillStyle = '#22d3ee';
    for (let i = 0; i < n; i++) { ctx.beginPath(); ctx.arc(px(xs[i]), py(ys[i]), 2.6 * s, 0, 6.2832); ctx.fill(); }
    ctx.restore();
    ctx.save(); ctx.shadowColor = '#fbbf24'; ctx.shadowBlur = 8; ctx.fillStyle = '#fbbf24';
    ctx.beginPath(); ctx.arc(px(0), py(np), 3.6 * s, 0, 6.2832); ctx.fill(); ctx.restore();
    ctx.fillStyle = '#fbbf24'; ctx.font = 'bold ' + Math.round(9 * s) + 'px ' + _CN_FONT;
    ctx.fillText('N=' + Number(np).toFixed(2), px(0) + 5 * s, Math.max(8 * s, py(np) - 4 * s));
  }

  function drawDispMobile(canvas, d) {
    const ctx = canvas.getContext('2d'); const W = canvas.width, H = canvas.height;
    const s = Math.max(1, H / 120);
    ctx.clearRect(0, 0, W, H); ctx.fillStyle = '#0b1120'; ctx.fillRect(0, 0, W, H);
    if (!d || !d.N_k || d.N_k.length < 2) {
      ctx.fillStyle = '#3a4a5a'; ctx.font = Math.round(10 * s) + 'px ' + _CN_FONT; ctx.textAlign = 'center';
      ctx.fillText('等待定标…', W / 2, H / 2); ctx.textAlign = 'left'; return;
    }
    const ys = d.N_k, K = ys.length;
    const med = _median(ys), sd = _mad(ys) * 1.4826;
    const yLo = Math.min(...ys, med - sd) - 0.05, yHi = Math.max(...ys, med + sd) + 0.05;
    const xticks = ys.map((_, i) => i + 1);
    const yticks = []; for (let k = 0; k <= 4; k++) yticks.push(+(yLo + (yHi - yLo) * k / 4).toFixed(2));
    const { px, py, padL, padR, padT } = _diagAxes(ctx, W, H, 0.5, K + 0.5, yLo, yHi, xticks, yticks);
    ctx.fillStyle = 'rgba(248,113,113,0.12)';
    ctx.fillRect(px(0.5), py(med + sd), px(K + 0.5) - px(0.5), py(med - sd) - py(med + sd));
    ctx.strokeStyle = '#f59e0b'; ctx.setLineDash([5, 3]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(px(0.5), py(med)); ctx.lineTo(px(K + 0.5), py(med)); ctx.stroke(); ctx.setLineDash([]);
    const bw = (px(2) - px(1)) * 0.55;
    for (let i = 0; i < K; i++) {
      const x = px(i + 1), y = py(ys[i]), base = py(yLo);
      ctx.fillStyle = 'rgba(34,211,238,0.7)';
      ctx.fillRect(x - bw / 2, Math.min(y, base), bw, Math.max(1, Math.abs(base - y)));
    }
    ctx.fillStyle = '#f59e0b'; ctx.font = 'bold ' + Math.round(9 * s) + 'px ' + _CN_FONT; ctx.textAlign = 'right';
    ctx.fillText('σ=' + sd.toFixed(3), W - padR, padT + 4 * s);
    ctx.textAlign = 'left';
  }

  function drawTempMobile(canvas, data) {
    const ctx = canvas.getContext('2d'); const W = canvas.width, H = canvas.height;
    const s = Math.max(1, H / 120);
    ctx.clearRect(0, 0, W, H); ctx.fillStyle = '#0b1120'; ctx.fillRect(0, 0, W, H);
    if (!data || data.length < 2) {
      ctx.fillStyle = '#3a4a5a'; ctx.font = Math.round(10 * s) + 'px ' + _CN_FONT; ctx.textAlign = 'center';
      ctx.fillText('等待数据', W / 2, H / 2); ctx.textAlign = 'left'; return;
    }
    const n = data.length;
    const tMin = Math.min(...data), tMax = Math.max(...data);
    const span = (tMax - tMin) || 1;
    const yLo = tMin - span * 0.15, yHi = tMax + span * 0.15;
    const xT = [...new Set([0, Math.floor(n / 4), Math.floor(n / 2), Math.floor(3 * n / 4), n - 1])];
    const yT = []; for (let k = 0; k <= 4; k++) yT.push(+(yLo + (yHi - yLo) * k / 4).toFixed(1));
    const { px, py, padL, padR, padT, padB } = _diagAxes(ctx, W, H, 0, n - 1, yLo, yHi, xT, yT);
    // 渐变填充
    const grad = ctx.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0, 'rgba(56,189,248,0.18)'); grad.addColorStop(1, 'transparent');
    ctx.beginPath();
    data.forEach((v, i) => { i === 0 ? ctx.moveTo(px(i), py(v)) : ctx.lineTo(px(i), py(v)); });
    ctx.lineTo(px(n - 1), H - padB); ctx.lineTo(px(0), H - padB); ctx.closePath();
    ctx.fillStyle = grad; ctx.fill();
    // 辉光数据线
    ctx.save(); ctx.shadowColor = '#38bdf8'; ctx.shadowBlur = 7;
    ctx.strokeStyle = '#38bdf8'; ctx.lineWidth = 1.8 * s; ctx.lineJoin = 'round';
    ctx.beginPath();
    data.forEach((v, i) => { i === 0 ? ctx.moveTo(px(i), py(v)) : ctx.lineTo(px(i), py(v)); });
    ctx.stroke(); ctx.restore();
    // 当前值 (端点)
    const lx = px(n - 1), ly = py(data[n - 1]);
    ctx.fillStyle = '#fff';
    ctx.beginPath(); ctx.arc(lx, ly, 2.5 * s, 0, 6.2832); ctx.fill();
    ctx.fillStyle = '#38bdf8'; ctx.font = 'bold ' + Math.round(9 * s) + 'px ' + _CN_FONT; ctx.textAlign = 'left';
    ctx.fillText(Number(data[n - 1]).toFixed(1) + '°C', lx + 6 * s, Math.max(10 * s, ly - 6 * s));
    // 单位标注
    ctx.fillStyle = 'rgba(148,163,184,0.7)'; ctx.font = Math.round(9 * s) + 'px ' + _CN_FONT; ctx.textAlign = 'right';
    ctx.fillText('°C', W - padR, padT + 4 * s);
    ctx.textAlign = 'left';
  }

  // 诊断图节流 0.4s (数字读数仍随 ws/measure 实时更新, 只节流 canvas 重绘)
  let lastDiagDraw = 0;
  function drawDiagMobile(d) {
    const pc = $('chart-phi'), dc = $('chart-disp');
    if (!pc || !dc) return;
    const now = performance.now();
    if (now - lastDiagDraw < 400) return;
    lastDiagDraw = now;
    setupChart(pc); setupChart(dc);
    drawPhiSMobile(pc, d);
    drawDispMobile(dc, d);
    const npt = $('m-diag-npoint'); if (npt) npt.textContent = Number(d.N_point).toFixed(2);
    const sig = $('m-diag-sigma');
    if (sig && d.N_k && d.N_k.length) sig.textContent = (_mad(d.N_k) * 1.4826).toFixed(3);
    state.lastDiag = d;
  }

  window.addEventListener('resize', () => {
    if (state.tBuffer.length) { const c = $('chart-temp'); if (c) { setupChart(c); drawTempMobile(c, state.tBuffer); } }
    if (state.lastDiag) {
      const pc = $('chart-phi'), dc = $('chart-disp');
      if (pc && dc) { setupChart(pc); setupChart(dc); drawPhiSMobile(pc, state.lastDiag); drawDispMobile(dc, state.lastDiag); }
    }
  });

  // ───────────────────────── 聊天 ─────────────────────────

  const messagesEl = $('chat-messages');
  // 智能滚动: 生成开始时伴随流式向下滚, 之后用户可自由滚动, 不强制跟随
  const mScroller = (() => {
    let atBottom = true;
    const nearBottom = () => messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight < 80;
    messagesEl.addEventListener('scroll', () => { atBottom = nearBottom(); }, { passive: true });
    return {
      begin() { atBottom = true; messagesEl.scrollTop = messagesEl.scrollHeight; },
      follow() { if (atBottom) messagesEl.scrollTop = messagesEl.scrollHeight; },
    };
  })();
  const inputEl = $('chat-input');
  const sendBtn = $('btn-send');

  function addMessage(role, html, imageUrl) {
    const div = document.createElement('div');
    div.className = 'msg ' + role;
    const avatar = document.createElement('div');
    avatar.className = 'msg-avatar';
    avatar.textContent = role === 'user' ? '👤' : '🤖';
    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';
    if (imageUrl) {
      const img = document.createElement('img');
      img.className = 'user-image';
      img.src = imageUrl;
      bubble.appendChild(img);
    }
    if (html) {
      const wrapper = document.createElement('div');
      wrapper.innerHTML = html;
      bubble.appendChild(wrapper);
    }
    div.appendChild(avatar);
    div.appendChild(bubble);
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return bubble;
  }

  function addTyping() {
    const bubble = addMessage('assistant', '');
    bubble.innerHTML = '<div class="typing-dots"><span></span><span></span><span></span></div>';
    return bubble;
  }

  // ── 深度思考开关 (类似 DeepSeek, 后端 /api/agent/config 持久) ──
  let thinkEnabled = true;
  let sendAbort = null;   // AbortController: 生成中可中断

  function updateMobileThinkUI() {
    const btn = $('think-toggle');
    if (!btn) return;
    btn.classList.toggle('off', !thinkEnabled);   // 亮=开, 暗=关 (无文字)
  }
  async function toggleThinking() {
    thinkEnabled = !thinkEnabled;
    updateMobileThinkUI();
    try { await api('/api/agent/config', 'POST', { thinking: thinkEnabled }); } catch (e) {}
  }
  // 修复: HTML 内联 onclick 调用的是全局函数, IIFE 内函数必须显式挂 window
  // (此前 3 个开关按钮点击报 ReferenceError 全部失效)
  window.toggleThinking = toggleThinking;
  (function initMobileThink() {
    fetch('/api/agent/status').then(r => r.json()).then(d => {
      if (d && typeof d.thinking === 'boolean') { thinkEnabled = d.thinking; updateMobileThinkUI(); }
    }).catch(() => {});
  })();

  // ── 语音输入 (Web Speech API; 不支持时隐藏按钮) ──
  let voiceRec = null, voiceOn = false;
  function initMobileVoice() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    const btn = $('voice-btn');
    if (!SR) { if (btn) btn.style.display = 'none'; return; }
    voiceRec = new SR();
    voiceRec.lang = 'zh-CN';
    voiceRec.interimResults = false;
    voiceRec.continuous = false;
    voiceRec.onresult = (e) => {
      let t = '';
      for (let i = e.resultIndex; i < e.results.length; i++) t += e.results[i][0].transcript;
      inputEl.value = (inputEl.value ? inputEl.value + ' ' : '') + t;
      inputEl.focus(); autoGrow();
    };
    voiceRec.onend = () => setMobileVoiceUI(false);
    voiceRec.onerror = () => setMobileVoiceUI(false);
  }
  function toggleVoice() {
    if (!voiceRec) return;
    if (voiceOn) { try { voiceRec.stop(); } catch (e) {} setMobileVoiceUI(false); }
    else { try { voiceRec.start(); setMobileVoiceUI(true); } catch (e) { setMobileVoiceUI(false); } }
  }
  window.toggleVoice = toggleVoice;
  function setMobileVoiceUI(on) {
    voiceOn = on;
    const btn = $('voice-btn');
    if (btn) btn.classList.toggle('active', on);
  }
  initMobileVoice();

  // ── 联网搜索开关 (🌐 亮=开/暗=关) ──
  let webSearchOn = false;
  let pendingMobileFile = null;   // {name, content}
  function updateMobileWebUI() {
    const btn = $('web-search-btn');
    if (btn) btn.classList.toggle('off', !webSearchOn);
  }
  function toggleWebSearch() {
    webSearchOn = !webSearchOn;
    updateMobileWebUI();
  }
  window.toggleWebSearch = toggleWebSearch;

  // ── 文件上传 ("+" 按钮): 图片→附加; 文本/PDF/DOCX→后端提取 ──
  function handleMobileFileSelect(ev) {
    const f = ev.target.files && ev.target.files[0];
    ev.target.value = '';
    if (!f) return;
    if (f.type.startsWith('image/')) {
      const r = new FileReader();
      r.onload = () => {
        state.pendingImage = { dataUrl: r.result, name: f.name };
        $('attached-preview').src = r.result;
        $('attached-image').classList.remove('hidden');
      };
      r.readAsDataURL(f);
      return;
    }
    const fd = new FormData();
    fd.append('file', f);
    fetch('/api/agent/upload', { method: 'POST', body: fd })
      .then(r => r.json())
      .then(res => {
        if (!res.ok) {
          addMessage('assistant', '<div class="err-banner">⚠️ ' + escapeHtml(res.msg || '上传失败') + '</div>');
          return;
        }
        pendingMobileFile = { name: res.name, content: res.content || '' };
        $('mobile-file-name').textContent = res.name;
        $('mobile-file-chip').classList.remove('hidden');
      })
      .catch(e => addMessage('assistant', '<div class="err-banner">⚠️ 上传失败: ' + escapeHtml(e.message) + '</div>'));
  }
  function clearMobileFile() {
    pendingMobileFile = null;
    $('mobile-file-chip').classList.add('hidden');
  }

  function setSendBtn(stopping) {
    sendBtn.textContent = stopping ? '⏹' : '➤';
    sendBtn.setAttribute('aria-label', stopping ? '停止' : '发送');
    sendBtn.classList.toggle('stopping', stopping);
  }
  function stopMobileStream() { if (sendAbort) sendAbort.abort(); }

  async function sendMessage(text, imageDataUrl) {
    if (state.sending || !text.trim()) return;
    state.sending = true;
    setSendBtn(true);

    // 展示用户消息
    const mobileFile = pendingMobileFile;   // 修复: 先快照, clearMobileFile() 会置 null
    const userText = mobileFile ? '[📄 ' + mobileFile.name + '] ' + text.trim() : text.trim();
    addMessage('user', escapeHtml(userText), imageDataUrl);
    inputEl.value = '';
    clearPendingImage();
    clearMobileFile();
    autoGrow();

    // 回答气泡: 流式增量渲染 (思考折叠 + 正式回答)
    const bubble = addMessage('assistant',
      '<div class="answer-body"><span class="typing-dots"><span></span><span></span><span></span></span></div>');
    mScroller.begin();   // 开始生成: 伴随流式向下滚
    const answerEl = bubble.querySelector('.answer-body');
    let ansText = '', thinkText = '', thinkEl = null, toolsEl = null, toolCount = 0;

    const controller = new AbortController();
    sendAbort = controller;
    const body = { prompt: text.trim(), web_search: webSearchOn };
    if (imageDataUrl) body.image_base64 = imageDataUrl;
    else body.use_frame = true;   // 修复: 默认附当前相机帧 (与桌面端一致); 原恒 false 致"评估条纹质量"等视觉问题无图
    if (mobileFile) { body.file_content = mobileFile.content; body.file_name = mobileFile.name; }

    // 超时看门狗: 180s 未收到任何数据视为响应超时 (移动网络网关常掐断长静默连接)
    let lastDataTs = Date.now();
    let timedOut = false;
    const watchTimer = setInterval(() => {
      if (Date.now() - lastDataTs > 180000 && !timedOut) {
        timedOut = true;
        try { controller.abort(); } catch (e) {}
      }
    }, 10000);

    try {
      const resp = await fetch('/api/mobile/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      if (!resp.ok || !resp.body) {
        let detail = '';
        try { detail = await resp.text(); } catch (e) {}
        throw new Error('HTTP ' + resp.status + (detail ? '（' + detail.slice(0, 120) + '）' : ''));
      }
      const reader = resp.body.getReader();
      const dec = new TextDecoder();
      let buf = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        lastDataTs = Date.now();
        buf += dec.decode(value, { stream: true });
        let idx;
        while ((idx = buf.indexOf('\n\n')) >= 0) {
          const frame = buf.slice(0, idx); buf = buf.slice(idx + 2);
          const line = frame.split('\n').find(l => l.startsWith('data: '));
          if (!line) continue;
          let ev; try { ev = JSON.parse(line.slice(6)); } catch (e) { continue; }
          if (ev.t === 'think') {
            if (!thinkEl) {
              thinkEl = document.createElement('details');
              thinkEl.className = 'mobile-think';
              thinkEl.open = true;
              thinkEl.innerHTML = '<summary>' + ATOM_SVG + ' 深度思考</summary><div class="think-text"></div>';
              bubble.insertBefore(thinkEl, answerEl);
            }
            thinkText += ev.d;
            thinkEl.querySelector('.think-text').textContent = thinkText;
          } else if (ev.t === 'tool') {
            // 每个被调用的工具都要显示: 名称 + 参数 + 执行结果
            if (!toolsEl) {
              toolsEl = document.createElement('details');
              toolsEl.className = 'mobile-tools';
              toolsEl.open = true;
              toolsEl.innerHTML = '<summary>🔧 调用工具中…</summary><div class="tools-body"></div>';
              bubble.insertBefore(toolsEl, answerEl);
            }
            toolCount++;
            toolsEl.querySelector('summary').textContent = '🔧 调用工具中… (' + toolCount + ')';
            const item = document.createElement('div');
            item.className = 'tool-trace-item';
            item.dataset.name = ev.name;
            item.innerHTML = '<div class="tool-trace-name">🔧 ' + escapeHtml(ev.name)
              + ' <code>' + escapeHtml(JSON.stringify(ev.args || {})) + '</code></div>'
              + '<div class="tool-trace-result">执行中…</div>';
            toolsEl.querySelector('.tools-body').appendChild(item);
          } else if (ev.t === 'tresult') {
            if (toolsEl) {
              const items = toolsEl.querySelectorAll('.tool-trace-item');
              for (let i = items.length - 1; i >= 0; i--) {
                if (items[i].dataset.name === ev.name) {
                  items[i].querySelector('.tool-trace-result').textContent = ev.result;
                  break;
                }
              }
            }
          } else if (ev.t === 'tok') {
            ansText += ev.d;
            answerEl.innerHTML = renderMarkdown(ansText);
          } else if (ev.t === 'error') {
            ansText += '\n⚠️ ' + ev.d;
          }
          mScroller.follow();
        }
      }
    } catch (e) {
      if (e.name === 'AbortError') {
        ansText = (ansText ? ansText + '\n\n' : '') +
          (timedOut ? '⏱ 响应超时（180 秒未收到数据）' : '⏹ 已停止生成') +
          '，可重新发送或换个问题试试';
      } else {
        ansText = (ansText ? ansText + '\n\n' : '') +
          '⚠️ 网络错误: ' + escapeHtml(e.message || '未知错误') +
          '<br>请检查手机与电脑是否同一网络，或稍后重试';
      }
    } finally {
      clearInterval(watchTimer);
      if (thinkEl) thinkEl.removeAttribute('open');   // 完成后默认收起
      if (toolsEl) {   // 工具调用盒: 收起 + 显示总数
        toolsEl.removeAttribute('open');
        toolsEl.querySelector('summary').textContent = '🔧 调用了 ' + toolCount + ' 个工具';
      }
      answerEl.innerHTML = ansText ? renderMarkdown(ansText) : '<i>无回复</i>';
      state.sending = false;
      setSendBtn(false);
      sendAbort = null;
      inputEl.focus();
    }
  }

  // 生成中: 发送按钮 → 停止按钮; Enter 在生成中忽略
  sendBtn.addEventListener('click', () => {
    if (state.sending) { stopMobileStream(); return; }
    sendMessage(inputEl.value, state.pendingImage && state.pendingImage.dataUrl);
  });
  inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (state.sending) return;
      sendMessage(inputEl.value, state.pendingImage && state.pendingImage.dataUrl);
    }
  });

  // 输入框自动增高
  function autoGrow() {
    inputEl.style.height = 'auto';
    inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + 'px';
  }
  inputEl.addEventListener('input', autoGrow);

  // 预设 chips (无 data-prompt 的按钮跳过, 如"生成报告")
  document.querySelectorAll('.chip').forEach(chip => {
    if (!chip.dataset.prompt) return;
    chip.addEventListener('click', () => {
      const p = chip.dataset.prompt;
      inputEl.value = p;
      autoGrow();
      sendMessage(p, state.pendingImage && state.pendingImage.dataUrl);
    });
  });

  // ── 9 阶段指导推进 (修复: 移动端此前无 next/prev 入口, 指导卡在第 1 阶段无法推进) ──
  function guideStep(action) {
    api('/api/agent/guide/' + action, 'POST', {}).then(r => {
      if (r && r.ok) {
        if (r.finished) addMessage('assistant', '🎉 ' + (r.msg || '全部流程完成！'));
        else addMessage('assistant',
          '📋 第' + r.step_no + '/' + r.total + '阶段《' + r.title + '》\n' + (r.objective || ''));
      } else if (r && r.warn) {
        addMessage('assistant', '⚠️ ' + (r.msg || '本阶段未达标，可继续或先完成该阶段'));
      } else {
        addMessage('assistant', (action === 'next' ? '➡' : '⬅') + ' ' +
          ((r && r.msg) || '实验指导未在进行，可先点"开始实验"'));
      }
    }).catch(() => addMessage('assistant', '⚠️ 指导操作失败，请稍后重试'));
  }
  const gp = $('btn-guide-prev'), gn = $('btn-guide-next');
  if (gp) gp.addEventListener('click', () => guideStep('prev'));
  if (gn) gn.addEventListener('click', () => guideStep('next'));

  // 生成实验报告 (第九阶段): 调后端生成 PDF, 聊天显示下载链接
  const mReportBtn = $('btn-mobile-report');
  if (mReportBtn) mReportBtn.addEventListener('click', async () => {
    addMessage('assistant', '📄 正在生成实验报告（实测数据 + LLM 结论 + VLM 评价 + 图表）…');
    try {
      const res = await api('/api/agent/report/generate', 'POST', {});
      if (res.ok && res.pdf_url) {
        addMessage('assistant',
          '📄 实验报告已生成：<a href="' + res.pdf_url + '" target="_blank">' +
          res.pdf_name + '（' + res.size_kb + ' KB' +
          (res.latex_used ? ' · LaTeX公式' : ' · mathtext降级') + '）</a>');
      } else {
        addMessage('assistant', '❌ ' + (res.msg || '报告生成失败'));
      }
    } catch (e) {
      addMessage('assistant', '❌ 报告生成失败: ' + e.message);
    }
  });

  // ───────────────────────── 图片上传 ─────────────────────────

  const fileInput = $('file-input');
  $('btn-attach').addEventListener('click', () => fileInput.click());

  fileInput.addEventListener('change', async () => {
    const file = fileInput.files && fileInput.files[0];
    if (!file) return;
    fileInput.value = '';
    try {
      const dataUrl = await compressImage(file, 1024, 0.8);
      state.pendingImage = { dataUrl, name: file.name };
      const img = $('attached-preview');
      img.src = dataUrl;
      $('attached-image').classList.remove('hidden');
      $('attached-caption').textContent = file.name + ' · 已压缩';
    } catch (e) {
      addMessage('assistant', '<div class="err-banner">⚠️ 图片处理失败: ' + escapeHtml(e.message) + '</div>');
    }
  });

  function clearPendingImage() {
    state.pendingImage = null;
    $('attached-image').classList.add('hidden');
    $('attached-preview').src = '';
  }
  $('btn-remove-image').addEventListener('click', clearPendingImage);

  // ── "+" 上传文件/图片 (图片走上面, 文本/PDF/DOCX 走后端提取) ──
  const mobileFileInput = $('mobile-file-input');
  $('btn-upload').addEventListener('click', () => mobileFileInput.click());
  mobileFileInput.addEventListener('change', handleMobileFileSelect);
  $('btn-clear-file').addEventListener('click', clearMobileFile);
  updateMobileWebUI();

  // 清空历史
  $('btn-clear-history') && $('btn-clear-history').addEventListener('click', async () => {
    await api('/api/mobile/clear_history', 'POST', {});
    messagesEl.innerHTML = '';
  });

  // ───────────────────────── Agent 状态徽标 / 实验指导 ─────────────────────────

  async function refreshAgentStatus() {
    const s = await api('/api/agent/status');
    if (s.phase) {
      state.agentPhase = s.phase;
      $('badge-agent').className = 'badge ' + (s.phase === 'running' ? 'online' : (s.phase === 'idle' ? '' : 'offline'));
      $('badge-agent').textContent = s.phase === 'running' ? 'Agent 运行中' : 'Agent';
    }
  }
  refreshAgentStatus();
  setInterval(refreshAgentStatus, 5000);

  // ───────────────────────── 抽屉 ─────────────────────────

  const drawer = $('drawer');
  const overlay = $('drawer-overlay');

  function openDrawer() { drawer.classList.add('open'); overlay.classList.remove('hidden'); }
  function closeDrawer() { drawer.classList.remove('open'); overlay.classList.add('hidden'); }
  $('btn-drawer').addEventListener('click', openDrawer);
  $('btn-close-drawer').addEventListener('click', closeDrawer);
  overlay.addEventListener('click', closeDrawer);

  // 监控区展开/收起 → 控制 /ws/video 开关
  const monitorSection = $('section-monitor');
  monitorSection.addEventListener('toggle', () => {
    if (monitorSection.open) startVideo();
    else stopVideo();
  });

  // ───────────────────────── /ws/video (监控区展开时) ─────────────────────────

  function startVideo() {
    if (state.videoOpen) return;
    state.videoOpen = true;
    $('video-idle').textContent = '连接中…';
    // ?side=960: 后端长边降采样编码 (全分辨率 5MP/帧 ~300-500KB 在移动网络下是卡顿主因)
    state.videoWS = wsConnect('/ws/video?side=960', (e) => {
      const buf = new Uint8Array(e.data);
      if (buf.length < 4) return;
      const metaLen = (buf[0] << 24) | (buf[1] << 16) | (buf[2] << 8) | buf[3];
      let meta = {};
      try { meta = JSON.parse(new TextDecoder().decode(buf.subarray(4, 4 + metaLen))); } catch (err) {}
      const jpeg = buf.subarray(4 + metaLen);
      const canvas = $('video-canvas');
      createImageBitmap(new Blob([jpeg], { type: 'image/jpeg' })).then(bmp => {
        canvas.width = bmp.width;
        canvas.height = bmp.height;
        canvas.getContext('2d').drawImage(bmp, 0, 0);
        bmp.close();
      }).catch(() => {});
      $('video-idle').classList.add('hidden');
      // 用 meta 更新底栏 N (视频自带 meta)
      if (meta.N !== undefined) $('stat-N').textContent = fmt(meta.N, 1);
      if (meta.T !== undefined) $('stat-T').textContent = fmt(meta.T);
      $('monitor-ws-hint').textContent = '● 视频流';
    }, () => {
      $('monitor-ws-hint').textContent = '';
    });
  }

  function stopVideo() {
    state.videoOpen = false;
    if (state.videoWS) { state.videoWS.close(); state.videoWS = null; }
    $('video-idle').textContent = '视频未连接';
    $('video-idle').classList.remove('hidden');
    $('monitor-ws-hint').textContent = '';
  }

  // ───────────────────────── 温控控制 ─────────────────────────

  $('btn-set-sv').addEventListener('click', async () => {
    const v = parseFloat($('sv-input').value);
    if (isNaN(v)) { flash('SV 无效', 'err'); return; }
    // 安全钳制 20~60°C (与 Agent 工具一致; 原无钳制可写超限 SV, 触发反应层反复硬停)
    const clamped = Math.min(60, Math.max(20, v));
    if (clamped !== v) flash('SV 已钳制到 ' + clamped + '°C（安全范围 20~60）', 'err');
    const r = await api('/api/temp/set_sv', 'POST', { sv: clamped });
    if (r.ok) {
      // 修复: 设 SV 后自动切 PID (原仅写 SV 寄存器, 温控器处于手动模式时设了不加热,
      // 行为依赖历史状态易误判"控制不对"; 与 Agent set_temperature 的 regulate_to 语义一致)
      const h = await api('/api/temp/heat', 'POST', { action: 'on' });
      flash(h.ok ? 'SV=' + clamped + '°C，PID 加热已启动' : 'SV 已设，但 PID 启动失败', h.ok ? 'ok' : 'err');
    } else {
      flash(r.msg || 'SV 设定失败（温控可能离线）', 'err');
    }
  });
  $('btn-heat-on').addEventListener('click', async () => {
    // 修复: 输入框有 SV 时先写 SV 再启 PID (用户预期"填了 40 点加热 → 加热到 40")
    const v = parseFloat($('sv-input').value);
    if (!isNaN(v)) {
      const clamped = Math.min(60, Math.max(20, v));
      if (clamped !== v) flash('SV 已钳制到 ' + clamped + '°C（安全范围 20~60）', 'err');
      const svr = await api('/api/temp/set_sv', 'POST', { sv: clamped });
      if (!svr.ok) { flash(svr.msg || 'SV 设定失败（温控可能离线）', 'err'); return; }
    }
    const r = await api('/api/temp/heat', 'POST', { action: 'on' });
    flash(r.ok ? 'PID 加热已启动' : ('加热启动失败: ' + (r.msg || '')), r.ok ? 'ok' : 'err');
  });
  $('btn-heat-full').addEventListener('click', async () => {
    // 危险操作确认: 满功率忽略 SV, 持续加热至 60°C 反应层硬停 (原无确认易误触过冲)
    if (!window.confirm('满功率加热忽略设定温度，会持续加热到 60°C 自动硬停，确定？')) return;
    const r = await api('/api/temp/heat', 'POST', { action: 'full' });
    showResult(r, '满功率加热已开', '加热启动失败');
  });
  $('btn-heat-off').addEventListener('click', async () => {
    const r = await api('/api/temp/heat', 'POST', { action: 'off' });
    showResult(r, '加热已关', '关闭失败');
  });
  $('btn-temp-reconnect').addEventListener('click', async () => {
    const r = await api('/api/temp/reconnect', 'POST', {});
    showResult(r, '温控已重连', '温控重连失败');
  });

  // ───────────────────────── 相机控制 ─────────────────────────

  $('btn-cam-start').addEventListener('click', async () => {
    const r = await api('/api/camera/start', 'POST', {});
    showResult(r, '取流已启动', '取流启动失败');
  });
  $('btn-cam-stop').addEventListener('click', async () => {
    const r = await api('/api/camera/stop', 'POST', {});
    showResult(r, '取流已停止', '停止失败');
  });
  $('btn-set-exposure').addEventListener('click', async () => {
    const v = parseInt($('exposure-input').value);
    if (isNaN(v)) return;
    const r = await api('/api/camera/exposure', 'POST', { exposure_us: v });
    showResult(r, '曝光已设', '曝光设定失败');
  });
  $('btn-set-gain').addEventListener('click', async () => {
    const v = parseFloat($('gain-input').value);
    if (isNaN(v)) return;
    const r = await api('/api/camera/gain', 'POST', { gain_db: v });
    showResult(r, '增益已设', '增益设定失败');
  });
  $('btn-capture').addEventListener('click', async () => {
    const r = await api('/api/camera/capture', 'POST', {});
    showResult(r, r.file ? '已保存: ' + r.file : '已采集', '采集失败');
  });
  $('btn-cycle-center').addEventListener('click', async () => {
    const order = ['cnn', 'cv'];
    const idx = order.indexOf(state.centerMode);
    const next = order[(idx + 1) % order.length];
    const r = await api('/api/camera/center_mode', 'POST', { mode: next });
    if (r.ok) {
      state.centerMode = next;
      $('btn-cycle-center').textContent = '光心模式: ' + next;
    } else {
      showResult(r, '', '切换失败');
    }
  });

  // ───────────────────────── 扫描控制 ─────────────────────────

  document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.scanMode = btn.dataset.mode;
    });
  });

  $('btn-scan-config').addEventListener('click', async () => {
    // 修复: NaN 经 JSON.stringify 变 null → 后端静默用默认值, 用户输入无效参数无感知
    const t1 = parseFloat($('scan-t1').value), t2 = parseFloat($('scan-t2').value),
          st = parseFloat($('scan-step').value), ns = parseInt($('scan-nstep').value, 10);
    if (!isFinite(t1) || !isFinite(t2) || !isFinite(st) || !isFinite(ns)) {
      flash('扫描参数无效，请检查输入', 'err');
      return;
    }
    const body = {
      mode: state.scanMode,
      T_start: t1, T_end: t2, step: st, n_step: ns,
    };
    const r = await api('/api/scan/config', 'POST', body);
    if (r.mode) { $('scan-message').textContent = '配置生效: ' + r.mode + ' 共 ' + r.total_segments + ' 段'; }
    else showResult(r, '', '配置失败');
  });

  $('btn-scan-start').addEventListener('click', async () => {
    const r = await api('/api/scan/start', 'POST', {});
    if (r.ok) { $('scan-message').textContent = '扫描已启动…'; }
    else showResult(r, '', '扫描启动失败');
  });
  $('btn-scan-stop').addEventListener('click', async () => {
    const r = await api('/api/scan/stop', 'POST', {});
    if (r.ok) { $('scan-message').textContent = '扫描已停止'; }
  });

  // ───────────────────────── 测量控制 ─────────────────────────

  $('btn-measure-start').addEventListener('click', async () => {
    const r = await api('/api/measure/start', 'POST', {});
    // 修复: already_measuring 非错误, 原落入失败分支误报"计数启动失败"
    if (r.status === 'already_measuring') { flash('测量已在进行中', 'ok'); return; }
    showResult(r, '计数已开始', '计数启动失败');
  });
  $('btn-measure-stop').addEventListener('click', async () => {
    const r = await api('/api/measure/stop', 'POST', {});
    if (r.N !== undefined) {
      $('measure-readout').innerHTML =
        '<div class="ro"><span class="k">N</span><span class="v">' + fmt(r.N, 1) + '</span></div>' +
        '<div class="ro"><span class="k">帧数</span><span class="v">' + (r.n_frames || 0) + '</span></div>';
    } else showResult(r, '已停止', '停止失败');
  });

  // ───────────────────────── 结果区 ─────────────────────────

  async function refreshResult() {
    const r = await api('/api/scan/result');
    const summaryEl = $('result-summary');
    const budgetEl = $('result-budget');
    const tableEl = $('result-table');
    if (r.error) { summaryEl.innerHTML = '<div class="err-banner">' + escapeHtml(r.error) + '</div>'; return; }

    const segs = r.segments || [];
    // 修复: 有效段以 s.valid 为准 (合理性自检段 valid=False 但 alpha>0, 原 alpha>0 过滤口径与后端不一致)
    const valid = segs.filter(s => s && s.valid !== false);
    const n = valid.length;
    const avg = r.alpha_avg || 0;

    // ── 汇总 ──
    let sumHTML =
      '<div class="row"><span class="k">累计 α</span><span class="big">' + fmt(avg) + '</span></div>' +
      '<div class="row"><span class="k">参考值</span><span>' + (r.reference_alpha || 20.8) + ' ×10⁻⁶/K</span></div>' +
      '<div class="row"><span class="k">相对误差</span><span>' + fmt(r.error_pct) + '%</span></div>' +
      '<div class="row"><span class="k">有效段数</span><span>' + n + ' / ' + segs.length + '</span></div>';

    summaryEl.innerHTML = sumHTML;
    if (budgetEl) budgetEl.innerHTML = '';

    // ── 逐段表 ──
    if (segs.length) {
      const header = '<tr><th>段</th><th>ΔT°C</th><th>N</th><th>α</th></tr>';
      const rows = segs.map(s =>
        '<tr class="' + (s.flag === 'implausible' || s.alpha === 0 ? 'flagged' : '') + '">' +
          '<td>' + (s.seg || '-') + '</td>' +
          '<td>' + fmt(s.dT) + '</td>' +
          '<td>' + fmt(s.N, 1) + '</td>' +
          '<td>' + fmt(s.alpha) + (s.flag === 'implausible' ? ' ⚠' : '') + '</td>' +
        '</tr>'
      ).join('');
      tableEl.innerHTML =
        '<div class="result-table"><table><thead>' + header + '</thead><tbody>' + rows + '</tbody></table></div>';
    } else {
      tableEl.innerHTML = '';
    }
  }
  $('btn-refresh-result').addEventListener('click', refreshResult);
  refreshResult();

  // ───────────────────────── 通用反馈 ─────────────────────────

  let toastTimer = null;
  function showResult(r, okMsg, errMsg) {
    if (r && (r.ok || r.status === 'ok' || r.mode)) {
      if (okMsg) flash(okMsg, 'ok');
    } else {
      const m = (r && (r.msg || r.error)) || errMsg || '操作失败';
      flash(m, 'err');
    }
  }
  function flash(text, type) {
    let el = $('toast');
    if (!el) {
      el = document.createElement('div');
      el.id = 'toast';
      document.body.appendChild(el);
    }
    el.textContent = text;
    el.className = 'toast ' + type;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.className = 'toast'; }, 2200);
  }

  // 启动
  autoGrow();
})();
