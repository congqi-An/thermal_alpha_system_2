"""VLM 多模态助手 REST API — 辅助光路/教学问答/条纹诊断"""
import asyncio
from fastapi import APIRouter
from pydantic import BaseModel

from .. import state

router = APIRouter(prefix="/api/vlm", tags=["vlm"])


class AskReq(BaseModel):
    prompt: str
    use_frame: bool = True       # True 用当前视频帧, False 纯文本
    max_new: int = 400


@router.get("/status")
async def status():
    return {"ready": state.vlm is not None and state.vlm.ready}


@router.post("/ask")
async def ask(req: AskReq):
    if state.vlm is None or not state.vlm.ready:
        return {"ok": False, "msg": "VLM 未加载"}
    frame = None
    if req.use_frame:
        with state._frame_lock:
            f = state.latest_frame
        if f is not None:
            frame = f.copy()
    # VLM 推理慢(10-48s), 用线程不阻塞 event loop
    text, dt = await asyncio.to_thread(state.vlm.ask, frame, req.prompt, req.max_new)
    return {"ok": True, "text": text, "time": round(dt, 1)}
