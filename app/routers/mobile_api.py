"""Mobile API — 手机前端专用: 支持上传图片的 Agent 对话

主要能力:
  POST /api/mobile/chat          带图/相机帧 + 文本 → Agent 回复 (历史+工具+RAG)
  POST /api/mobile/clear_history 清空移动端聊天历史

对话历史与桌面端共享，桌面/手机双端可在实验流程中看到同一段对话：
guide 讲解、告警和双方对话均互通。

实验指导、状态和报告复用现有 /api/agent/start|stop|status|report。
"""
import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

from .. import state

router = APIRouter(prefix="/api/mobile", tags=["mobile"])


class MobileChatReq(BaseModel):
    prompt: str
    image_base64: Optional[str] = None   # 用户上传图片 (data URI 或裸 base64)
    use_frame: bool = False              # 附带当前相机帧 (上传图片时不生效)
    web_search: bool = False             # 🌐 联网搜索开关
    file_content: Optional[str] = None   # 用户上传文件提取的文本
    file_name: Optional[str] = None


@router.post("/chat/stream")
async def chat_stream(req: MobileChatReq):
    """移动端流式对话 (SSE)，复用 Agent 对话链并共享桌面端历史。"""
    if state.agent is None:
        async def err_gen():
            yield f"data: {json.dumps({'t': 'error', 'd': 'Agent 未初始化'}, ensure_ascii=False)}\n\n"
        return StreamingResponse(err_gen(), media_type="text/event-stream")

    # 移动端 image_base64 可能是完整 data URI, 剥前缀为裸 base64 (chat_stream 会再包装)
    b64 = req.image_base64
    if b64 and b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]

    async def gen():
        try:
            async for ev in state.agent.chat_stream(
                    req.prompt, use_frame=req.use_frame,
                    image_b64=b64, history=state.agent._chat_history,
                    web_search=req.web_search,
                    file_content=req.file_content,
                    file_name=req.file_name):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'t': 'error', 'd': str(e)[:200]}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/chat")
async def chat(req: MobileChatReq):
    """移动端对话 (非流式): 直接走 agent_core.chat, 共享桌面端聊天历史.
    注意: 前端目前只用 /chat/stream, 保留本路由供调试/脚本使用."""
    if state.agent is None:
        return {"ok": False, "msg": "Agent 未初始化 (检查 Ollama 是否运行)"}

    # 移动端 image_base64 可能是完整 data URI, 剥前缀为裸 base64
    b64 = req.image_base64
    if b64 and b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]

    return await state.agent.chat(
        req.prompt, use_frame=req.use_frame,
        image_b64=b64,
        web_search=req.web_search,
        file_content=req.file_content,
        file_name=req.file_name)


@router.post("/clear_history")
async def clear_history():
    """清空共享聊天历史 (双端同时生效)."""
    if state.agent:
        state.agent.clear_chat_history()
    return {"ok": True, "msg": "对话历史已清空"}
