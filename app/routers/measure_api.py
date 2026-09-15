"""手动单段测量 REST API"""
from fastapi import APIRouter
from pydantic import BaseModel

from .. import state
from ..measurement.phase_fringe_counter import FringeCounter
from ..measurement.alpha_calc import delta_L_from_N

router = APIRouter(prefix="/api/measure", tags=["measure"])


class StartReq(BaseModel):
    T_start: float = 0.0


@router.post("/start")
async def start(req: StartReq):
    # 硬件前置检查: 无帧源时禁止启动 (否则 N 恒为 0 误导用户);
    # 视频模拟源已加载时放行 (回放视频同样可计数)
    if state.video_source is None and (
            state.camera is None or not state.camera.is_grabbing):
        return {"status": "error", "msg": "相机未取流且未加载视频, 请先开始取流或加载视频"}
    with state.measure_lock:
        if state.measure_active:
            return {"status": "already_measuring"}
        if state.fringe_counter is None:
            state.fringe_counter = FringeCounter()
        state.fringe_counter.reset()
        state.n_gate = None  # N 振动门随之重置
        state.gated_N = 0.0
        state.gated_N_abs = 0.0  # 防短测量未定标时残留上一段 N
        state.measure_active = True
    return {"status": "ok", "T_start": req.T_start, "current_T": state.current_T}


@router.post("/stop")
async def stop():
    with state.measure_lock:
        state.measure_active = False
    if state.fringe_counter is None:
        return {"status": "no_data"}
    _, N_per = state.fringe_counter.total_N
    N_avg = state.gated_N_abs          # N 振动门 (惯性覆盖) 后的稳定值
    return {"status": "ok", "N": N_avg, "N_per": N_per,
            "n_frames": state.fringe_counter.n_frames}


@router.get("/summary")
async def summary():
    if state.fringe_counter is None:
        return {"measuring": False}
    _, N_per = state.fringe_counter.total_N
    N_avg = state.gated_N_abs          # N 振动门 (惯性覆盖) 后的稳定值
    dL = delta_L_from_N(N_avg)
    return {"measuring": state.measure_active, "N": round(N_avg, 2),
            "N_per": N_per, "delta_L_um": round(dL * 1e6, 3),
            "n_frames": state.fringe_counter.n_frames,
            "quality": round(state.current_quality, 3),
            "center_source": state.center_source, "r0_source": state.r0_source,
            "current_T": round(state.current_T, 2)}
