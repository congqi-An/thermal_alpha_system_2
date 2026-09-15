"""WebSocket: 视频流 / 实时测量 / 扫描进度"""
import json, struct, time, asyncio
import cv2
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .. import state
from ..config import VIDEO_JPEG_QUALITY, VIDEO_MAX_FPS
from ..measurement.alpha_calc import delta_L_from_N

router = APIRouter()


def _encode_video_frame(frame, now, max_side=0):
    """绘图 + JPEG 编码 (5MP 编码 30-80ms, 放入 worker 线程防阻塞事件循环)

    max_side>0: 长边降采样后编码 (移动端传 ?side=960, 帧体积 ~5MB/s→~1.5MB/s,
    手机解码开销同步下降; 桌面端默认全分辨率)。overlay 坐标按 scale 缩放。
    """
    # 相机返回 RGB, 转为 BGR 供 OpenCV 绘图 (cv2 颜色按 BGR 约定)
    vis = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    scale = 1.0
    if max_side and max(vis.shape[:2]) > max_side:
        scale = max_side / max(vis.shape[:2])
        vis = cv2.resize(vis, (int(vis.shape[1] * scale), int(vis.shape[0] * scale)))
    # overlay 中心十字 + r0 环
    center = getattr(state, "current_center", None)
    if center is not None:
        cx, cy = int(center[0] * scale), int(center[1] * scale)
        cv2.drawMarker(vis, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS,
                       int(40 * scale), 2)
        r0s = getattr(state, "current_r0s", None) or (150, 200, 250)
        for r0 in r0s:
            cv2.circle(vis, (cx, cy), int(r0 * scale), (0, 255, 0), 1)
    # 角标
    N_avg = 0.0
    if state.fringe_counter is not None:
        N_avg = state.gated_N_abs   # N 振动门 (惯性覆盖) 后
    dL = delta_L_from_N(N_avg)
    cv2.rectangle(vis, (0, 0), (520, 90), (0, 0, 0), -1)
    cv2.putText(vis, f"T={state.current_T:.1f}C  SV={state.current_sv:.1f}C",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis, f"N={N_avg:.1f}  dL={dL*1e6:.1f}um  Q={state.current_quality:.2f}",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(vis, f"{'MEASURING' if state.measure_active else 'idle'}",
                (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0) if state.measure_active else (128, 128, 128), 2)
    if getattr(state, "burst", False):
        cv2.putText(vis, "VIBRATION BURST", (150, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    meta = {"t": round(now, 3), "N": round(N_avg, 1),
            "delta_L_um": round(dL * 1e6, 2),
            "quality": round(state.current_quality, 2),
            "T": round(state.current_T, 2), "online": state.temp_online,
            "measuring": state.measure_active,
            "r0s": [int(r) for r in (state.current_r0s or ())],
            "center_source": state.center_source,
            "r0_source": state.r0_source,
            "stability": round(getattr(state, "stability", 1.0), 2),
            "burst": bool(getattr(state, "burst", False))}
    ok, jpeg = cv2.imencode('.jpg', vis,
                             [int(cv2.IMWRITE_JPEG_QUALITY), VIDEO_JPEG_QUALITY])
    meta_b = json.dumps(meta).encode()
    return meta_b, (jpeg.tobytes() if ok else b"")


# 同帧编码缓存: 多 ws 客户端 (桌面+手机) 同一帧只编码一次。
# 键 (max_side, id(frame)); id 复用窗口极短且以时间戳兜底 (300ms 内同 id 才命中)。
_video_cache = {}

@router.websocket("/ws/video")
async def ws_video(ws: WebSocket, side: int = 0):
    await ws.accept()
    max_side = side if 0 < side <= 1280 else 0   # 移动端传 ?side=960
    last = 0
    interval = 1.0 / VIDEO_MAX_FPS if VIDEO_MAX_FPS > 0 else 0.033
    try:
        while True:
            # 帧源就绪 = 视频模拟源已加载 或 相机在取流
            if state.video_source is None and (
                    state.camera is None or not state.camera.is_grabbing):
                await asyncio.sleep(0.1); continue
            with state._frame_lock:
                frame = state.latest_frame
            if frame is None:
                await asyncio.sleep(0.03); continue
            now = time.time()
            if now - last < interval:
                await asyncio.sleep(0.005); continue
            last = now

            # 同帧多客户端复用编码结果 (桌面/手机同时观看时编码开销减半)
            key = (max_side, id(frame))
            ent = _video_cache.get(key)
            if ent is None or now - ent[0] > 0.3:
                # 绘图+编码放入 worker 线程, 不阻塞事件循环 (Agent SSE / VLM ask 等仍可响应)
                meta_b, jpeg_bytes = await asyncio.to_thread(
                    _encode_video_frame, frame, now, max_side)
                payload = struct.pack('>I', len(meta_b)) + meta_b + jpeg_bytes
                _video_cache[key] = (now, payload)
                if len(_video_cache) > 8:
                    _video_cache.clear()
            else:
                payload = ent[1]
            await ws.send_bytes(payload)
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass


@router.websocket("/ws/measure")
async def ws_measure(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            N_avg = 0.0; N_per = [0]
            diag = None   # 诊断图数据: φ(s) 外推 + 通道分散度
            if state.fringe_counter is not None:
                _, N_per = state.fringe_counter.total_N
                N_avg = state.gated_N_abs   # N 振动门 (惯性覆盖) 后
                # 计数器内部统计量 → 前端诊断图 (φ(s) 外推 / 通道分散度)
                try:
                    st = state.fringe_counter.get_error_stats()
                    if st and st.get('s_norm') and len(st.get('N_per_channel', [])) >= 2:
                        diag = {
                            "s_norm": st['s_norm'],
                            "N_k": st['N_per_channel'],
                            "N_point": st['N_point'],
                            "amp_k": st['amp_k'],
                            "r0s": st.get('r0s_full', []),
                        }
                except Exception:
                    diag = None
            dL = delta_L_from_N(N_avg)
            # CNN 状态
            cnn_ok = getattr(state, 'frame_analysis_net', None) is not None
            await ws.send_json({
                "t": round(time.time(), 3),
                "N": round(N_avg, 2), "N_per": N_per,
                "delta_L_um": round(dL * 1e6, 3),
                "T": round(state.current_T, 2), "sv": round(state.current_sv, 2),
                "mv": round(state.current_mv, 2), "online": state.temp_online,
                "measuring": state.measure_active, "scan_active": state.scan_active,
                "quality": round(state.current_quality, 3),
                "stability": round(getattr(state, "stability", 1.0), 3),
                "burst": bool(getattr(state, "burst", False)),
                "cnn": cnn_ok,
                "center_source": state.center_source,
                "r0_source": state.r0_source,
                "center_mode": state.center_mode,
                "diag": diag,   # None 或 {s_norm, N_k, N_point, amp_k, r0s} (φ(s) + 分散度)
                "alert": state.last_alert,   # None 或 {seq, text, level, ts} (反应层 Q 跳变推送)
            })
            await asyncio.sleep(0.2)
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass


@router.websocket("/ws/scan")
async def ws_scan(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            if state.scan_runner is not None:
                await ws.send_json(state.scan_runner.status())
            await asyncio.sleep(1.0)
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
