"""Agent REST API — 9 阶段实验指导 + 对话 + 报告 控制接口"""
import base64
import io
import json

from fastapi import APIRouter, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

from .. import state

router = APIRouter(prefix="/api/agent", tags=["agent"])


# ─── 请求模型 ─────────────────────────────────────────────────────

class ChatReq(BaseModel):
    prompt: str
    use_frame: bool = True
    image_base64: Optional[str] = None   # 前端快照 (优先于 use_frame, 所见即所析)
    web_search: bool = False             # 🌐 联网搜索开关
    file_content: Optional[str] = None   # 用户上传文件提取的文本
    file_name: Optional[str] = None

class GuideNextReq(BaseModel):
    force: bool = False    # True=跳过本阶段未达标提醒, 直接进入下一阶段

class AiConfigReq(BaseModel):
    temperature: Optional[float] = None      # None=不改 (部分更新, 前端开关只发 thinking)
    max_tokens: Optional[int] = None
    thinking: Optional[bool] = None          # 深度思考开关 (DeepSeek 风格前端开关)


# ─── 接口 ─────────────────────────────────────────────────────────

@router.get("/health")
async def health():
    """检查 LLM 服务连通性"""
    if state.agent is None:
        return {"ok": False, "msg": "Agent 未初始化"}
    return await state.agent.health()


@router.post("/start")
async def start():
    """开始九阶段分步实验指导。"""
    if state.agent is None:
        return {"ok": False, "msg": "Agent 未初始化"}
    return state.agent.start_guide()


@router.post("/stop")
async def stop():
    """停止实验指导 / 中断 Agent"""
    if state.agent is None:
        return {"ok": False, "msg": "Agent 未初始化"}
    if state.agent.guide.active:
        return state.agent.stop_guide()
    state.agent.stop()
    return {"ok": True, "msg": "Agent 已停止"}


# ─── 分步实验指导 API ───────────────────────────────

@router.get("/guide")
async def guide_status():
    """获取实验指导进度 (全部步骤状态 + 当前步详情)"""
    if state.agent is None:
        return {"ok": False, "msg": "Agent 未初始化"}
    return {"ok": True, **state.agent.guide_status()}


@router.post("/guide/next")
async def guide_next(req: Optional[GuideNextReq] = None):
    """进入下一阶段 (软门控: 未达标返回 warn; force=True 跳过提醒)"""
    if state.agent is None:
        return {"ok": False, "msg": "Agent 未初始化"}
    return state.agent.guide_next(force=(req.force if req else False))


@router.post("/guide/prev")
async def guide_prev():
    """返回上一阶段"""
    if state.agent is None:
        return {"ok": False, "msg": "Agent 未初始化"}
    return state.agent.guide_prev()


@router.post("/guide/briefing")
async def guide_briefing():
    """SSE: agent 为当前阶段生成讲解 (think/tok/done), 前端流式渲染进聊天"""
    if state.agent is None:
        async def err_gen():
            yield f"data: {json.dumps({'t': 'error', 'd': 'Agent 未初始化'}, ensure_ascii=False)}\n\n"
        return StreamingResponse(err_gen(), media_type="text/event-stream")

    async def gen():
        try:
            async for ev in state.agent.guide_briefing_stream():
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'t': 'error', 'd': str(e)[:200]}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/guide/verify")
async def guide_verify():
    """运行当前阶段验证: state 读实时状态; vision 调 VLM 判定 → {passed, detail, vision}"""
    if state.agent is None:
        return {"ok": False, "msg": "Agent 未初始化"}
    return await state.agent.guide_verify()


@router.get("/status")
async def status():
    """Agent 当前状态"""
    if state.agent is None:
        return {"phase": "not_initialized"}
    return state.agent.status()


@router.get("/log")
async def log(last_n: int = 50):
    """获取决策日志"""
    if state.agent is None:
        return {"log": []}
    return {"log": state.agent.get_log(last_n)}


@router.post("/chat")
async def chat(req: ChatReq):
    """手动对话模式 (单次问答, 不触发自主流程)"""
    if state.agent is None:
        return {"ok": False, "msg": "Agent 未初始化"}
    result = await state.agent.chat(req.prompt, use_frame=req.use_frame,
                                    image_b64=req.image_base64,
                                    web_search=req.web_search,
                                    file_content=req.file_content,
                                    file_name=req.file_name)
    return result


@router.post("/chat/stream")
async def chat_stream(req: ChatReq):
    """流式对话 (SSE): 思考/工具/回答逐段推送, 前端实时渲染"""
    if state.agent is None:
        async def err_gen():
            yield f"data: {json.dumps({'t': 'error', 'd': 'Agent 未初始化'}, ensure_ascii=False)}\n\n"
        return StreamingResponse(err_gen(), media_type="text/event-stream")

    async def gen():
        try:
            async for ev in state.agent.chat_stream(
                    req.prompt, use_frame=req.use_frame,
                    image_b64=req.image_base64,
                    web_search=req.web_search,
                    file_content=req.file_content,
                    file_name=req.file_name):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'t': 'error', 'd': str(e)[:200]}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/report")
async def report():
    """获取 Agent 生成的实验报告"""
    if state.agent is None:
        return {"report": None}
    r = state.agent.get_report()
    return {"report": r, "has_report": r is not None}


@router.post("/report/generate")
async def report_generate():
    """第九阶段: 生成实验报告 PDF (实测数据 + LLM 结论 + VLM 评价 + 图表)"""
    if state.agent is None:
        return {"ok": False, "msg": "Agent 未初始化"}
    return await state.agent.generate_report()


@router.post("/report/capture-coarse")
async def report_capture_coarse():
    """粗调光路拍照: 自动从相机取当前帧 → 存档 coarse_photo.png + VLM 分析光点重合
    多次点击覆盖前次 (只取最新一张进报告)"""
    from ..agent.llm_client import LLMClient
    from ..agent.prompts import COARSE_PROMPT
    if state.agent is None:
        return {"ok": False, "msg": "Agent 未初始化"}
    with state._frame_lock:
        frame = state.latest_frame
    if frame is None:
        return {"ok": False, "msg": "相机无画面，请先启动取流"}
    state.agent._save_report_frame(frame, "coarse_photo.png")
    # VLM 分析光点重合 (失败不阻断存档)
    analysis = ""
    try:
        b64 = LLMClient.frame_to_base64(frame)
        if b64:
            analysis = (await state.agent.llm.ask_with_image(COARSE_PROMPT, b64) or "").strip()
    except Exception as e:
        print(f"  [Report] 粗调 VLM 分析失败: {e}")
    return {"ok": True, "asset": "coarse_photo.png",
            "analysis": analysis or "（VLM 分析不可用）"}


@router.post("/report/asset")
async def report_asset(file: UploadFile = File(...)):
    """上传实验过程照片 (粗调光点图等) → measurements/report_assets/，供报告嵌入"""
    from ..config import MEASURE_SAVE_DIR
    if not file.filename:
        return {"ok": False, "msg": "无文件名"}
    data = await file.read()
    if len(data) > 25_000_000:
        return {"ok": False, "msg": "文件过大 (>25MB)"}
    ext = (file.filename.rsplit(".", 1)[-1].lower()
           if "." in file.filename else "png")
    if ext not in ("png", "jpg", "jpeg", "bmp", "webp"):
        return {"ok": False, "msg": f"不支持图片格式: {ext}"}
    d = MEASURE_SAVE_DIR / "report_assets"
    d.mkdir(parents=True, exist_ok=True)
    name = "coarse_photo.png"     # 固定名: 报告引用粗调光点图
    (d / name).write_bytes(data)
    return {"ok": True, "asset": name, "path": str(d / name)}


@router.post("/clear_history")
async def clear_history():
    """清空对话历史"""
    if state.agent is None:
        return {"ok": False, "msg": "Agent 未初始化"}
    state.agent.clear_chat_history()
    return {"ok": True, "msg": "对话历史已清空"}


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """上传文件 → 供对话使用 (聊天区 "+" 按钮):
      图像 → {type:'image', base64}
      文本类 (txt/md/csv/json/log/…) → {type:'text', content}
      PDF/DOCX → 提取文本 → {type:'text', content}
      其他 → {ok:false}
    """
    name = file.filename or "file"
    data = await file.read()
    if len(data) > 25_000_000:
        return {"ok": False, "msg": "文件过大 (>25MB)"}
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    mime = file.content_type or ""

    # 图像 → base64 (供 VLM 视觉分析)
    if mime.startswith("image/") or ext in ("png", "jpg", "jpeg", "bmp", "gif", "webp", "tif", "tiff"):
        return {"ok": True, "type": "image", "name": name,
                "base64": base64.b64encode(data).decode("ascii")}

    # 文本类 → 直接读
    if ext in ("txt", "md", "csv", "json", "log", "py", "tex", "cfg", "yaml", "yml"):
        try:
            return {"ok": True, "type": "text", "name": name,
                    "content": data.decode("utf-8", errors="replace")[:4000]}
        except Exception:
            pass

    # PDF → pypdf 提取
    if ext == "pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(data))
            text = "\n".join((p.extract_text() or "") for p in reader.pages[:20])
            return {"ok": True, "type": "text", "name": name,
                    "content": (text or "(PDF 无可提取文本)")[:4000]}
        except Exception as e:
            return {"ok": False, "msg": f"PDF 提取失败: {e}"}

    # DOCX → python-docx 提取
    if ext == "docx":
        try:
            from docx import Document
            doc = Document(io.BytesIO(data))
            text = "\n".join(p.text for p in doc.paragraphs)
            return {"ok": True, "type": "text", "name": name,
                    "content": (text or "(DOCX 无可提取文本)")[:4000]}
        except Exception as e:
            return {"ok": False, "msg": f"DOCX 提取失败: {e}"}

    return {"ok": False, "msg": f"不支持的文件类型: {ext or mime}"}


@router.post("/config")
async def update_config(req: AiConfigReq):
    """实时更新 Agent 推理参数 (部分更新: 只改非 None 字段)"""
    if state.agent is None:
        return {"ok": False, "msg": "Agent 未初始化"}
    state.agent.update_config(
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        thinking=req.thinking,
    )
    return {"ok": True, "msg": "AI 参数已更新", **state.agent.config_snapshot()}

# ─── 实验叙事 API ─────────────────────────────────────────────────

@router.get("/narrative")
async def get_narrative(last_n: int = 20):
    """获取实验叙事日志（最近 N 条解说/观察）"""
    if state.agent is None:
        return {"ok": False, "msg": "Agent 未初始化", "narrative": []}
    return {"ok": True, "narrative": state.agent.get_narrative(last_n)}


@router.get("/phase")
async def get_phase():
    """获取当前实验阶段 + 阶段转换历史"""
    if state.agent is None:
        return {"ok": False, "msg": "Agent 未初始化", "phase": "idle"}
    return {"ok": True, **state.agent.get_experiment_phase()}


@router.websocket("/ws/narration")
async def ws_narration(ws: WebSocket):
    """WebSocket 实时叙事推送：Agent 每生成一条解说即推送到所有连接的客户端"""
    await ws.accept()
    if state.agent is None:
        await ws.send_json({"type": "error", "text": "Agent 未初始化"})
        await ws.close()
        return
    state.agent._narration_ws_clients.append(ws)
    try:
        while True:
            # 保持连接，接收客户端 ping（无需处理）
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if ws in state.agent._narration_ws_clients:
            state.agent._narration_ws_clients.remove(ws)
