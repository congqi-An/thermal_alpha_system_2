"""自动温度扫描 REST API"""
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from .. import state

router = APIRouter(prefix="/api/scan", tags=["scan"])


class ScanConfig(BaseModel):
    T_start: Optional[float] = None
    T_end: Optional[float] = None
    step: Optional[float] = None
    stabilize_s: Optional[float] = None
    mode: Optional[str] = None      # temp=温度步进 / fringe=N步进 / free=即时计数
    n_step: Optional[int] = None    # fringe 模式每段条纹数


@router.post("/config")
async def config(req: ScanConfig):
    if state.scan_runner is None:
        return {"ok": False, "msg": "scan_runner not init"}
    if req.mode is not None and req.mode not in ("temp", "fringe", "free"):
        return {"ok": False, "msg": f"无效模式: {req.mode} (应为 temp/fringe/free)"}
    return state.scan_runner.configure(req.T_start, req.T_end, req.step,
                                       req.stabilize_s, req.mode, req.n_step)


@router.post("/start")
async def start():
    if state.scan_runner is None:
        return {"ok": False, "msg": "not init"}
    # 硬件前置检查: 无帧源时禁止启动扫描; 视频模拟源已加载时放行
    if state.video_source is None and (
            state.camera is None or not state.camera.is_grabbing):
        return {"ok": False, "msg": "相机未取流且未加载视频, 请先开始取流或加载视频"}
    ok = state.scan_runner.start()
    return {"ok": ok}


@router.post("/stop")
async def stop():
    if state.scan_runner is None:
        return {"ok": False}
    state.scan_runner.stop()
    return {"ok": True}


@router.get("/status")
async def status():
    if state.scan_runner is None:
        return {"active": False}
    return state.scan_runner.status()


@router.get("/result")
async def result():
    if state.scan_runner is None:
        return {}
    return state.scan_runner.result()
