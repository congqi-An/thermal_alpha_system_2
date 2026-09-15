"""相机控制 REST API"""
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel

from .. import state

router = APIRouter(prefix="/api/camera", tags=["camera"])


def _cam():
    if state.camera is None:
        raise HTTPException(503, "Camera not initialized")
    return state.camera


@router.get("/info")
async def info():
    cam = _cam()
    return {"connected": cam.is_open and cam.width > 0, "grabbing": cam.is_grabbing,
            "width": cam.width, "height": cam.height, "model": "MV-CS050-60GC"}


@router.post("/start")
async def start():
    cam = _cam()
    # 未真正连接/未打开 → 拒绝, 不假装取流成功 (修复: 之前无脑返回 grabbing=true)
    if not cam.is_open or cam.width <= 0:
        raise HTTPException(503, "相机未连接，无法取流")
    try:
        cam.start()
    except Exception as e:
        raise HTTPException(500, f"开始取流失败: {e}")
    if not cam.is_grabbing:
        raise HTTPException(500, "开始取流失败：相机未进入取流状态")
    return {"status": "ok", "ok": True, "grabbing": True,
            "width": cam.width, "height": cam.height}


@router.post("/stop")
async def stop():
    cam = _cam()
    if cam.is_open:
        try:
            cam.stop()
        except Exception as e:
            raise HTTPException(500, f"停止取流失败: {e}")
    return {"status": "ok", "ok": True, "grabbing": bool(cam.is_grabbing)}


@router.post("/reconnect")
async def reconnect():
    """Reconnect camera: close + re-enumerate + reopen + start grab."""
    from ..camera.hik_camera import HikCamera
    from ..config import (CAMERA_EXPOSURE_US, CAMERA_GAIN_DB, CAMERA_FPS,
                          CAMERA_GAMMA, CAMERA_BLACKLEVEL)
    # 先关旧相机 (GigE 独占访问, 新句柄必须等旧句柄释放)
    old = state.camera
    if old is not None:
        try:
            if old.is_grabbing:
                old.stop()
            old.close()
        except Exception:
            pass
    # 重新打开; 失败时关闭半开的新对象并置 state.camera=None (干净"未初始化"),
    # 避免 state.camera 指向已关闭的旧对象/泄漏半开句柄
    cam = None
    try:
        cam = HikCamera()
        cam.open()
        cam.configure(exposure_us=CAMERA_EXPOSURE_US, gain_db=CAMERA_GAIN_DB, fps=CAMERA_FPS)
        cam.set_gamma(CAMERA_GAMMA)
        cam.set_blacklevel(CAMERA_BLACKLEVEL)
        cam.start()
        state.camera = cam
        return {"ok": True, "width": cam.width, "height": cam.height}
    except Exception as e:
        if cam is not None:
            try:
                cam.close()
            except Exception:
                pass
        state.camera = None
        return {"ok": False, "msg": str(e)}


class ExpReq(BaseModel):
    exposure_us: int

class GainReq(BaseModel):
    gain_db: float

class FpsReq(BaseModel):
    fps: float


@router.post("/exposure")
async def exposure(req: ExpReq):
    _cam().set_exposure(req.exposure_us)
    return {"status": "ok", "exposure_us": req.exposure_us}


@router.post("/gain")
async def gain(req: GainReq):
    _cam().set_gain(req.gain_db)
    return {"status": "ok", "gain_db": req.gain_db}


@router.post("/framerate")
async def framerate(req: FpsReq):
    _cam().set_framerate(req.fps)
    return {"status": "ok", "fps": req.fps}


class GammaReq(BaseModel):
    gamma: float

class BlackReq(BaseModel):
    blacklevel: float

class AutoExpReq(BaseModel):
    on: bool


@router.post("/gamma")
async def gamma(req: GammaReq):
    _cam().set_gamma(req.gamma)
    return {"status": "ok", "gamma": req.gamma}

@router.post("/blacklevel")
async def blacklevel(req: BlackReq):
    _cam().set_blacklevel(req.blacklevel)
    return {"status": "ok", "blacklevel": req.blacklevel}

@router.post("/auto_exposure")
async def auto_exposure(req: AutoExpReq):
    _cam().set_auto_exposure(req.on)
    return {"status": "ok", "auto_exposure": req.on}


class SetReq(BaseModel):
    param: str
    value: float
    type: str = "float"      # float / enum / int


@router.post("/set")
async def set_param(req: SetReq):
    cam = _cam()
    if req.type == "enum":
        ok = cam.set_enum(req.param, int(req.value))   # enum 须 int
    elif req.type == "int":
        ok = cam.set_int(req.param, int(req.value))
    else:
        ok = cam.set_float(req.param, req.value)
    return {"ok": ok, "param": req.param, "value": req.value}


@router.get("/get")
async def get_param(param: str, ptype: str = "float"):
    cam = _cam()
    v = cam.get_float(param) if ptype == "float" else cam.get_int(param)
    return {"param": param, "value": v}


@router.get("/snapshot")
async def snapshot():
    """返回当前帧 JPEG (供对话界面附带画面: 点击时拍快照, 与发给模型的图一致)
    长边压缩到 1280: 控制 VLM 图像 token 开销 (全分辨率会超上下文窗口)"""
    import cv2
    with state._frame_lock:
        frame = state.latest_frame
    if frame is None:
        raise HTTPException(404, "无可用画面, 请先开始取流")
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    if max(h, w) > 1280:
        scale = 1280 / max(h, w)
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise HTTPException(500, "帧编码失败")
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@router.post("/capture")
async def capture():
    """保存当前帧到 captures/ (供 CNN 实况微调采集训练数据)"""
    import cv2, time
    from pathlib import Path
    with state._frame_lock:
        frame = state.latest_frame
    if frame is None:
        return {"ok": False, "msg": "无可用帧, 请先开始取流"}
    cap_dir = Path(__file__).resolve().parent.parent.parent / "captures"
    cap_dir.mkdir(exist_ok=True)
    fname = f"live_{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time()*1000)%1000:03d}.png"
    path = cap_dir / fname
    # frame 是 RGB, imwrite 需 BGR
    cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    n_total = len(list(cap_dir.glob("live_*.png")))
    return {"ok": True, "file": fname, "total": n_total}


class CenterModeReq(BaseModel):
    mode: str   # "cnn" / "cv"


@router.get("/center_mode")
async def get_center_mode():
    """当前光心/ROI 判定模式"""
    return {"mode": state.center_mode,
            "cnn_loaded": state.frame_analysis_net is not None}


@router.post("/center_mode")
async def set_center_mode(req: CenterModeReq):
    """切换光心/ROI 判定模式：cnn=纯 CNN，cv=纯 CV。"""
    mode = req.mode.lower()
    if mode not in ("cnn", "cv"):
        return {"ok": False, "msg": f"无效模式: {req.mode} (应为 cnn/cv)",
                "mode": state.center_mode}
    if mode == "cnn" and state.frame_analysis_net is None:
        return {"ok": False, "msg": "CNN 模型未加载, 仅支持 cv 模式",
                "mode": state.center_mode}
    state.center_mode = mode
    print(f"[camera_api] 光心/ROI 判定模式切换: {mode}")
    return {"ok": True, "mode": mode}


# ── 视频模拟源: 加载本地视频当相机输入, 走完整分析链路 ──

class VideoLoadReq(BaseModel):
    path: str
    loop: bool = False   # 默认不循环: 播完定格
    speed: float = 1.0


class VideoCtrlReq(BaseModel):
    action: str   # play / pause / rewind / unload


class QualityGateReq(BaseModel):
    enabled: bool


@router.get("/quality_gate")
async def get_quality_gate():
    return {"quality_gate_enabled": state.quality_gate_enabled}


@router.post("/quality_gate")
async def set_quality_gate(req: QualityGateReq):
    state.quality_gate_enabled = req.enabled
    print(f"[camera_api] 质量门控 = {'开' if req.enabled else '关'} "
          f"(关=CNN质量头不参与计数决策)")
    return {"quality_gate_enabled": state.quality_gate_enabled}


@router.post("/video/load")
async def video_load(req: VideoLoadReq):
    """加载视频模拟源 (接管帧源, 卸载才回相机)。加载后自动播放。"""
    from ..camera.video_source import VideoFileSource
    # 先卸旧源
    if state.video_source is not None:
        state.video_source.release()
        state.video_source = None
    try:
        vs = VideoFileSource(req.path, loop=req.loop, speed=req.speed)
    except Exception as e:
        return {"ok": False, "msg": str(e)}
    vs.play()
    state.video_source = vs
    # 视频模拟 = 自动进入测量: grab 线程喂计数器, N/ΔL 实时计数 (无需手动点开始测量)
    with state.measure_lock:
        if state.fringe_counter is None:
            from app.measurement.phase_fringe_counter import FringeCounter
            state.fringe_counter = FringeCounter()
        state.fringe_counter.reset()      # 总是重置: 每次加载都从头计数
        state.n_gate = None
        state.gated_N = 0.0
        state.gated_N_abs = 0.0
        if not state.measure_active:
            state.measure_active = True
            state.video_auto_measure = True
    print(f"[camera_api] 视频模拟源加载: {vs.name} "
          f"({vs.width}x{vs.height} @{vs.fps:.1f}fps, {vs.n_frames}帧) "
          f"— 自动进入测量 (N 实时计数)")
    return {"ok": True, **vs.status()}


@router.post("/video/ctrl")
async def video_ctrl(req: VideoCtrlReq):
    """视频控制: play/pause/rewind/unload"""
    vs = state.video_source
    if vs is None:
        return {"ok": False, "msg": "未加载视频"}
    if req.action == "play":
        vs.play()
    elif req.action == "pause":
        vs.pause()
    elif req.action == "rewind":
        vs.rewind()
    elif req.action == "unload":
        vs.release()
        state.video_source = None
        # 退出视频模拟自动测量
        with state.measure_lock:
            if state.video_auto_measure:
                state.measure_active = False
                state.video_auto_measure = False
        print("[camera_api] 视频模拟源已卸载, 帧源回到相机")
        return {"ok": True, "loaded": False}
    else:
        return {"ok": False, "msg": f"无效动作: {req.action}"}
    return {"ok": True, **vs.status()}


@router.get("/video/status")
async def video_status():
    if state.video_source is None:
        return {"loaded": False}
    return state.video_source.status()
