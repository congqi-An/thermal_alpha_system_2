"""基于计算机视觉和轻量VLM的干涉法测量热膨胀智能实验系统 — FastAPI 启动

运行: conda activate ocv && cd thermal_alpha_system_2 && uvicorn app.main:app --host 0.0.0.0 --port 8080
"""
import sys, threading, time, asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2
import torch
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app import state
from app.config import (WEB_HOST, WEB_PORT, CAMERA_EXPOSURE_US, CAMERA_GAIN_DB, CAMERA_FPS,
                        CAMERA_GAMMA, CAMERA_BLACKLEVEL,
                        MODBUS_PORT, MODBUS_BAUD, MODBUS_SLAVE, MEASURE_SAVE_DIR,
                        QUALITY_MODEL_PATH, QUALITY_THRESHOLD, VLM_ENABLED,
                        CENTER_METHOD, CENTER_CNN_PATH,
                        CENTER_R0_NET_PATH, CENTER_R0_METHOD, COUNT_R0S,
                        FRAME_ANALYSIS_NET_PATH, LLM_SERVER_URL)
from app.camera.hik_camera import HikCamera
from app.controller.modbus_temp import TempController
from app.measurement.center_tracker import (find_radial_center, auto_r0_detect,
                                             refine_center_symmetry)
from app.measurement.center_cnn import CenterNet, find_center_cnn
# 带符号空间相位计数（替代时序过零法；方向敏感，吞吐往复自动抵消），drop-in
from app.measurement.phase_fringe_counter import FringeCounter
from app.scan.scan_runner import ScanRunner

app = FastAPI(title="基于计算机视觉和轻量VLM的干涉法测量热膨胀智能实验系统", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_running = False


def _find_center_and_r0(red, H, W):
    """光心+ROI 定位分发:
    优先 CenterR0Net (一次推理出 center + r0), 回退 CenterNet/CV + 固定 r0.
    返回 (cx, cy, score, r0s)."""
    # 优先: CenterR0Net 双头 CNN (光心 + r0 一次出)
    if state.center_r0_method == "cnn" and state.center_r0_net is not None:
        from app.measurement.center_r0_net import predict_center_r0
        try:
            (cx, cy, score), r0s = predict_center_r0(
                red, H, W, state.center_r0_net, state.center_r0_device)
            if 0 <= cx < W and 0 <= cy < H:
                return cx, cy, score, r0s
        except Exception:
            pass
    # 回退: CenterNet/CV 光心 + 固定 r0
    cx, cy, score = _find_center(red, H, W)
    return cx, cy, score, COUNT_R0S


def _find_center(red, H, W):
    """光心定位分发: CNN(默认) 或 CV, CNN 失败/越界回退 CV. 返回 (cx, cy, score)."""
    if state.center_method == "cnn" and state.center_cnn is not None:
        prev = getattr(state, "current_center", None)
        pcx = prev[0] if prev is not None else None
        pcy = prev[1] if prev is not None else None
        out = find_center_cnn(red, H, W, state.center_cnn, state.center_cnn_device, pcx, pcy)
        if out is not None:
            cx, cy = out
            if 0 <= cx < W and 0 <= cy < H:
                return cx, cy, 1.0
            # 越界 -> 回退 CV
    return find_radial_center(red, H, W)


def _grab_loop():
    """取流 + (测量时) 中心追踪 + 条纹计数 update

    帧源优先级: 视频模拟源 (video_source, 回放实验视频) > 相机实拍。
    视频源走与实拍完全相同的分析链路 (光心/r0/质量/计数/信号图)。"""
    global _running
    while _running:
        # 取帧单独防护: 停止取流/重连的瞬间 grab() 可能抛 MvsError,
        # 不能让竞态异常杀死线程 (否则视频/测量永久停摆)
        try:
            vs = state.video_source
            if vs is not None:
                # 视频模拟模式: 加载即接管帧源 (卸载才回相机);
                # 暂停/播完/未到出帧时间返 None → 画面定格在 latest_frame
                frame = vs.grab()
                if frame is None:
                    # 视频播完(非循环): 退出测量(无论 auto 还是手动, 防永久挂起),
                    # 画面定格在 latest_frame
                    if getattr(vs, "finished", False) and state.measure_active:
                        with state.measure_lock:
                            if state.measure_active:
                                state.measure_active = False
                            state.video_auto_measure = False
                    time.sleep(0.005); continue
                if getattr(vs, "looped", False):   # 视频循环/重播回卷: 每圈重新计数
                    vs.looped = False
                    if state.fringe_counter is not None:
                        state.fringe_counter.reset()
                    state.n_gate = None
                    state.gated_N = 0.0
                    state.gated_N_abs = 0.0
            elif state.camera is None or not state.camera.is_grabbing:
                time.sleep(0.05); continue
            else:
                frame = state.camera.grab(timeout_ms=200)
                if frame is None:
                    time.sleep(0.01); continue
            with state._frame_lock:
                state.latest_frame = frame
        except Exception as e:
            print(f'[grab] grab err: {e}')
            time.sleep(0.1); continue
        # 始终算 center + r0 + 质量(实时显示); 计数仅 measure_active 且质量过关
        try:
            red = frame[:, :, 0].astype(np.float32) / 255.0
            H, W = red.shape
            state._r0_cnt += 1

            # 优先: FrameAnalysisNet 三头一次推理 (center + r0 + quality)
            # 光心/r0 判定模式 (state.center_mode, 前端按钮切换):
            #   cnn  = 纯 CNN (center/r0 头直接采用, 退化仅警告不切换)
            #   cv   = 纯 CV (峰谷对称法光心 + 径向 std 法 r0, CNN 仅留质量头)
            if state.frame_analysis_net is not None:
                from app.measurement.frame_analysis_net import predict_frame, r0_degenerate
                (cx_raw, cy_raw), r0s, q, r_raw = predict_frame(
                    red, H, W, state.frame_analysis_net, state.frame_analysis_device,
                    return_raw=True)
                state.current_quality = q
                mode = state.center_mode

                # 每 30 帧: 按模式更新光心源 + 显示 r0
                if state._r0_cnt % 30 == 1:
                    if mode == "cnn":
                        # 纯 CNN: 直接采用三头输出, 不跟 CV 交叉
                        state.center_source = "cnn"
                        if r0_degenerate(r_raw):
                            print("[grab] 警告: CNN r0 头输出退化 "
                                  f"(raw={np.round(r_raw, 3)}), 纯CNN模式不切换")
                        state.r0_source = "cnn"
                        state.current_r0s = r0s
                        state._cv_center_ref = None
                        print(f"[grab] center=({cx_raw:.0f},{cy_raw:.0f})[cnn] "
                              f"r0={state.current_r0s}[cnn] q={q:.2f} (mode=cnn)")
                    elif mode == "cv":
                        # 纯 CV: 峰谷对称法光心 + 径向 std 法 r0, CNN 仅取质量头 q
                        cv_cx, cv_cy, cv_score = find_radial_center(red, H, W)
                        cv_cx, cv_cy, sym = refine_center_symmetry(red, cv_cx, cv_cy, H, W)
                        r0s = auto_r0_detect(red, cv_cx, cv_cy, r_max=560)
                        # 过滤中心亮斑区 (r<120 通道污染严重)
                        r0s = tuple(int(r) for r in r0s if r >= 120) or (200, 300, 400)
                        state.center_source = "cv"
                        state.r0_source = "cv-std"
                        state.current_r0s = r0s
                        state._cv_center_ref = (cv_cx, cv_cy)
                        print(f"[grab] center=({cv_cx:.0f},{cv_cy:.0f})[cv] sym={sym:.2f} "
                              f"r0={state.current_r0s}[cv-std] q={q:.2f} (mode=cv)")
                # 周期内: 光心按模式/校验结果选源 (CV 用缓存参照, 避免每帧跑 CV)
                if state.center_source == "cv" and getattr(state, "_cv_center_ref", None):
                    cx_raw, cy_raw = state._cv_center_ref

            else:
                # 回退: 分离的 CenterR0Net/CenterNet + QualityNet
                if state._r0_cnt % 30 == 1:
                    cx_raw, cy_raw, score, r0s = _find_center_and_r0(red, H, W)
                    state.current_r0s = r0s
                    state.center_source = "cnn" if (state.center_r0_net is not None
                                                    or state.center_cnn is not None) else "cv"
                    state.r0_source = "cnn" if state.center_r0_net is not None else "cv-std"
                    print(f"[grab] center=({cx_raw:.0f},{cy_raw:.0f}) r0={r0s}")
                else:
                    cx_raw, cy_raw, score = _find_center(red, H, W)
                if state.quality_net is not None:
                    inp = cv2.resize(red, (256, 256))
                    qt = torch.from_numpy(inp).unsqueeze(0).unsqueeze(0)
                    qt = qt.to(next(state.quality_net.parameters()).device)
                    with torch.no_grad():
                        q = float(torch.sigmoid(state.quality_net(qt)).item())
                    state.current_quality = q

            # 中心滑动平均(压扰动, 减少中心抖动对 r0/N 的影响)
            # 低质量帧(暗帧/模糊)的 center 不可信, 不污染滑动窗口
            has_q = (state.frame_analysis_net is not None
                     or state.quality_net is not None)
            # 质量门控: 关闭时 CNN 质量头不参与计数决策 (所有帧光心都进 EMA, 原行为)
            if not has_q or not state.quality_gate_enabled \
               or state.current_quality >= QUALITY_THRESHOLD \
               or not state.center_history:
                state.center_history.append((cx_raw, cy_raw))
                if len(state.center_history) > 5:
                    state.center_history.pop(0)
            cx_s = sum(c[0] for c in state.center_history) / len(state.center_history)
            cy_s = sum(c[1] for c in state.center_history) / len(state.center_history)
            state.current_center = (cx_s, cy_s)
            cx, cy = cx_s, cy_s
            if state.current_r0s is None:
                state.current_r0s = COUNT_R0S
            # 质量门控: Q<0.5 时计数器惯性滑行(用100好帧sum/100推进坏帧)
            # update() 与 scan_runner/measure_api 的 reset() 用 measure_lock 串行化,
            # 防段边界 reset 与 update() 并发导致 N 撕裂/丢失; 锁内重查 measure_active
            # (避免等待锁期间测量已被关闭, 拿到锁后仍处理已关闭段)。
            with state.measure_lock:
                if state.measure_active and state.fringe_counter is not None:
                    if hasattr(state.fringe_counter, 'set_ring_mode'):
                        state.fringe_counter.set_ring_mode(state.center_mode)
                    state.fringe_counter.update(red, cx, cy, T=state.current_T,
                                                t=time.time(), r0_hint=state.current_r0s,
                                                q_hint=state.current_quality)
                    # N 振动控制门 (惯性覆盖, ≤2 条/s): 门后 N 供显示/结算, 防大震动跳变
                    # 热身期(_cfg 未定标) N_signed 恒 0.0——若喂门会把 clean 窗灌满 0,
                    # 定标后首次真实 N 跳变被当振动限速爬坡 → 结算低估 (S3)。定标前不喂门,
                    # 门保持 None, 定标后第一次进才懒建, 自然从真实 N 起步。
                    if state.fringe_counter._cfg is not None:
                        if state.n_gate is None:
                            from app.measurement.n_rate_gate import NRateGate
                            state.n_gate = NRateGate()
                        state.gated_N = state.n_gate.update(
                            state.fringe_counter.N_signed, time.time())
                        state.gated_N_abs = abs(state.gated_N)
                    # 时序稳定度驱动"质量": 替代恒1的CNN q → 激活 center 门(q<0.4 排除
                    # 振动帧光心) + 暴露 burst。仅测量期覆盖; 空闲时 current_quality 仍=
                    # CNN q(=1), center 门不触发(与今日行为一致)。
                    # 稳定度/burst 仅作遥测(显示+Agent 告警), 不参与计数/门控 —— 干净段
                    # 逐帧噪声过高, 自动 coast/门控 会回归(验证: 干净+coast=28.8 vs 16.46)
                    if hasattr(state.fringe_counter, 'q_stab'):
                        state.stability = round(state.fringe_counter.q_stab, 3)
                        state.burst = bool(state.fringe_counter.burst)
                        # 报告图表: 稳定度/振动时序 (诊断振动), 降采样 0.5s 限长
                        now = time.time()
                        if now - state._last_disp_t >= 0.5:
                            state.disp_history.append(
                                (now, state.stability, state.burst))
                            state._last_disp_t = now
                            if len(state.disp_history) > 20000:
                                del state.disp_history[:1000]
        except Exception as e:
            print(f'[grab] err: {e}')
        time.sleep(0.005)


def _temp_sync_loop():
    """把 TempController 的 PV/SV/MV 同步到 state（供 API/前端读）"""
    global _running
    while _running:
        if state.temp_ctrl is not None:
            state.current_T = state.temp_ctrl.pv
            state.current_sv = state.temp_ctrl.sv
            state.current_mv = state.temp_ctrl.mv
            state.temp_online = state.temp_ctrl.online
            state.temp_history = list(state.temp_ctrl.history)
        time.sleep(0.5)


async def _reaction_loop():
    """反应层后台: 每 2s 运行 SafetyGuard 全量检查, 检测到异常时调 LLM 生成提醒。

    纯规则检测 (<1ms), LLM 调用 ~1s 且受冷却保护 (冷却时长见 SafetyGuard:
    quality_drop 60s, 其余同类事件 10s 不重复)。
    同时驱动 Agent 实验阶段推断与阶段解说。
    """
    from app.agent.safety import SafetyGuard
    from app.agent.event_bus import event_bus, EventType

    safety = SafetyGuard(event_bus)
    print("  [Reaction] SafetyGuard online (Q<0.4/3s → 提醒, T>60°C → 硬停)")

    while True:
        try:
            safety.check_all()

            # 实验阶段推断 (纯 state 读取, <1ms, 无副作用):
            # 阶段转换时自动生成叙事 (heating/measuring/complete 等)
            try:
                if state.agent is not None:
                    state.agent._detect_phase_transition()
            except Exception:
                pass

            events = event_bus.pop_all()
            for ev in events:
                if ev.type == EventType.QUALITY_DROP:
                    # Q < 0.4 持续 3s → LLM 生成提醒注 入对话
                    await _inject_quality_drop_alert(ev.data)
                elif ev.type in (EventType.TEMP_SAFETY, EventType.FRINGE_LOST):
                    await _inject_critical_alert(ev.type, ev.data)

            await asyncio.sleep(2.0)
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"  [Reaction] 异常: {e}")
            await asyncio.sleep(5.0)


async def _inject_quality_drop_alert(data: dict):
    """Q 持续低 (<0.4 达 3s) → LLM 生成提醒文案 → 注入对话"""
    if state.agent is None or state.agent.llm is None:
        return
    q = data.get("quality", 0)
    dur = data.get("duration_s", 0)

    prompt = (
        f"实验系统自动监测：条纹质量持续偏低（当前 {q:.2f}，已 {dur:.0f} 秒低于 0.4 阈值）。"
        f"请用 1 句通俗温和的话提醒用户（不用标题/格式），语气像实验搭档，"
        f"可以建议检查光路、调节相机参数或等待恢复。")
    try:
        resp = await state.agent.llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.7, max_tokens=60, think=False)
        text = (resp.content or "").strip()
    except Exception:
        text = ""
    if not text:
        text = f"条纹质量连续 {dur:.0f} 秒偏低（当前 {q:.2f}），建议检查光路或调节曝光。"

    _deliver_alert(text, "warn")


async def _inject_critical_alert(ev_type, data: dict):
    """安全联锁硬停 → LLM 生成警告"""
    if state.agent is None or state.agent.llm is None:
        return
    if ev_type.value == "temp_safety":
        prompt = (f"紧急：实验温度达到 {data.get('temperature','?')}°C，"
                  f"已自动降温。请用 1 句温和的话告知用户。")
    else:
        prompt = (f"条纹疑似消失（质量 {data.get('quality','?')}），"
                  f"请用 1 句温和的话提醒用户检查光路。")
    try:
        resp = await state.agent.llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.7, max_tokens=60, think=False)
        text = (resp.content or "").strip()
    except Exception:
        text = ""
    if not text:
        text = "系统检测到异常，已自动采取保护措施。"

    _deliver_alert(text, "danger")


def _deliver_alert(text: str, level: str):
    """告警文案注入共享聊天历史 (双端可见) + /ws/measure 推送"""
    import time as _time
    msg = {"role": "assistant", "content": f"⚠️ {text}"}
    if state.agent:
        state.agent._chat_history.append(msg)
    state.alert_seq += 1
    state.last_alert = {
        "seq": state.alert_seq, "text": text, "level": level,
        "ts": round(_time.time(), 1),
    }
    print(f"  [Reaction] 告警已投递 (seq={state.alert_seq}): {text[:80]}...")


@app.on_event("startup")
async def startup():
    global _running
    print("=" * 60)
    print("  基于计算机视觉和轻量VLM的干涉法测量热膨胀智能实验系统  (Michelson + He-Ne 632.8nm)")
    print("=" * 60)
    MEASURE_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # 相机
    try:
        state.camera = HikCamera()
        state.camera.open()
        state.camera.configure(exposure_us=CAMERA_EXPOSURE_US,
                                gain_db=CAMERA_GAIN_DB, fps=CAMERA_FPS)
        # 应用 Gamma/黑电平 (config.py 定义但 configure() 不含, 此处补上)
        state.camera.set_gamma(CAMERA_GAMMA)
        state.camera.set_blacklevel(CAMERA_BLACKLEVEL)
        print(f"  [OK] Camera {state.camera.width}x{state.camera.height} (待开始取流)")
    except Exception as e:
        state.camera = None
        print(f"  [WARN] Camera: {e}")

    # 温控
    try:
        state.temp_ctrl = TempController(MODBUS_PORT, MODBUS_BAUD, MODBUS_SLAVE)
        if state.temp_ctrl.open():
            state.temp_ctrl.start_monitoring()
            print(f"  [OK] TempController {MODBUS_PORT}@{MODBUS_BAUD}")
            # PID 参数自检: 仪表被复位回出厂参数时提醒重新整定
            try:
                pp = state.temp_ctrl.read_pid_params()
                if (pp.get('P1'), pp.get('P2'), pp.get('rt')) == (116, 71, 353):
                    print("  [WARN] 温控器 PID 参数为出厂默认(未整定), "
                          "建议运行 autotune_pid.py")
                elif pp.get('P1') is not None:
                    print(f"  [temp] PID 已整定: P1={pp['P1']} P2={pp['P2']} rt={pp['rt']}")
            except Exception:
                pass
        else:
            print(f"  [WARN] TempController {MODBUS_PORT} 未连接 — 演示模式")
    except Exception as e:
        print(f"  [WARN] TempController: {e}")

    # 测量算法
    state.fringe_counter = FringeCounter()
    state.scan_runner = ScanRunner()
    state.current_center = None

    # 帧分析三头 CNN FrameAnalysisNet (优先: 光心+ROI+质量 一次推理)
    try:
        from app.measurement.frame_analysis_net import load_frame_analysis_net
        fan = load_frame_analysis_net(FRAME_ANALYSIS_NET_PATH,
                                      "cuda" if torch.cuda.is_available() else "cpu")
        if fan is not None:
            state.frame_analysis_net = fan
            state.frame_analysis_device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"  [OK] FrameAnalysisNet loaded on {state.frame_analysis_device} (光心+ROI+质量 三合一)")
        else:
            raise FileNotFoundError(str(FRAME_ANALYSIS_NET_PATH))
    except Exception as e:
        print(f"  [WARN] FrameAnalysisNet not loaded, 回退分离模型: {e}")
        state.frame_analysis_net = None

    # 条纹质量评估 QualityNet (FrameAnalysisNet 不可用时回退)
    if state.frame_analysis_net is None:
        try:
            from app.measurement.quality_net import QualityNet
            qn = QualityNet()
            qn.load_state_dict(torch.load(str(QUALITY_MODEL_PATH), map_location="cpu",
                                           weights_only=True), strict=False)
            qn.eval()
            if torch.cuda.is_available():
                qn = qn.cuda()
            state.quality_net = qn
            print(f"  [OK] QualityNet loaded (回退)")
        except Exception as e:
            print(f"  [WARN] QualityNet not loaded: {e}")

    # 光心+ROI 双头 CNN CenterR0Net (FrameAnalysisNet 不可用时回退)
    if state.frame_analysis_net is None:
        try:
            from app.measurement.center_r0_net import CenterR0Net
            crn = CenterR0Net()
            crn.load_state_dict(torch.load(str(CENTER_R0_NET_PATH), map_location="cpu",
                                            weights_only=True), strict=True)
            crn.eval()
            crdev = "cuda" if torch.cuda.is_available() else "cpu"
            if crdev == "cuda":
                crn = crn.cuda()
            state.center_r0_net = crn
            state.center_r0_device = crdev
            state.center_r0_method = CENTER_R0_METHOD
            print(f"  [OK] CenterR0Net loaded on {crdev} (回退光心+ROI)")
        except Exception as e:
            print(f"  [WARN] CenterR0Net not loaded: {e}")
            state.center_r0_net = None
            state.center_r0_method = "fixed"

    # 光心定位 CenterNet (最终回退)
    try:
        cn = CenterNet()
        cn.load_state_dict(torch.load(str(CENTER_CNN_PATH), map_location="cpu",
                                       weights_only=True), strict=True)
        cn.eval()
        cdev = "cuda" if torch.cuda.is_available() else "cpu"
        if cdev == "cuda":
            cn = cn.cuda()
        state.center_cnn = cn
        state.center_cnn_device = cdev
        state.center_method = CENTER_METHOD
        print(f"  [OK] CenterCNN loaded on {cdev} (回退光心)")
    except Exception as e:
        print(f"  [WARN] CenterCNN not loaded, 用 CV find_radial_center: {e}")
        state.center_cnn = None
        state.center_method = "cv"

    # 多模态 VLM 助手 (可选, 显存大加载慢, VLM_ENABLED=False 时跳过)
    if VLM_ENABLED:
        try:
            from app.ai.vlm import VLMAssistant
            state.vlm = VLMAssistant()
            state.vlm.load()
            print(f"  [OK] VLM loaded")
        except Exception as e:
            print(f"  [WARN] VLM not loaded: {e}")
    else:
        print(f"  [SKIP] VLM disabled (config VLM_ENABLED=False)")

    # RAG 知识库 (启动时加载向量索引; 无索引时提示构建, 不阻断服务)
    try:
        from app.rag import rag_store
        if rag_store.load():
            print(f"  [OK] RAG index loaded ({len(rag_store.chunks)} chunks)")
        else:
            print("  [WARN] RAG index missing, 运行 docs/build_index.py 构建后重启生效")
    except Exception as e:
        print(f"  [WARN] RAG load failed: {e}")

    # Qwen3-VL 4B Agent (Ollama): 9 阶段指导 + 对话工具 + 反应层
    try:
        from app.agent.agent_core import ExperimentAgent
        state.agent = ExperimentAgent(llm_url=LLM_SERVER_URL)
        print(f"  [OK] Agent initialized (Ollama: {LLM_SERVER_URL})")
    except Exception as e:
        print(f"  [WARN] Agent not initialized: {e}")

    _running = True
    threading.Thread(target=_grab_loop, daemon=True).start()
    threading.Thread(target=_temp_sync_loop, daemon=True).start()

    # Ollama 模型生命周期: 后台预热双模型 + 保活循环 (退出时自动卸载)
    from app.ai.ollama_lifecycle import warmup, keepalive_loop
    asyncio.create_task(warmup())
    asyncio.create_task(keepalive_loop())

    # ── 反应层: 纯规则安全检查后台循环 (温度/Q跳变/光心漂移) ──
    asyncio.create_task(_reaction_loop())
    print("  [OK] Reaction layer started (SafetyGuard every 2s)")

    print(f"  Server: http://localhost:{WEB_PORT}")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown():
    global _running
    _running = False
    # 退出时卸载 Ollama 模型, 释放显存 (不残留)
    try:
        from app.ai.ollama_lifecycle import unload_models
        await unload_models()
    except Exception:
        pass
    if state.camera is not None:
        try:
            if state.camera.is_grabbing:
                state.camera.stop()
            state.camera.close()
        except Exception:
            pass
    if state.temp_ctrl is not None:
        state.temp_ctrl.close()


from app.routers import camera_api, temp_api, measure_api, scan_api, vlm_api, config_api, ws, agent_api, mobile_api
app.include_router(camera_api.router)
app.include_router(temp_api.router)
app.include_router(measure_api.router)
app.include_router(scan_api.router)
app.include_router(vlm_api.router)
app.include_router(config_api.router)
app.include_router(ws.router)
app.include_router(agent_api.router)
app.include_router(mobile_api.router)

measurements_dir = Path(__file__).resolve().parent.parent / "measurements"
measurements_dir.mkdir(parents=True, exist_ok=True)
app.mount("/measurements", StaticFiles(directory=str(measurements_dir)), name="measurements")

static_dir = Path(__file__).resolve().parent.parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=WEB_HOST, port=WEB_PORT, reload=True)
