"""Agent 引擎 — 9 阶段实验指导、对话工具与实验报告。"""
import asyncio
import json
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from .. import state
from ..config import LLM_SERVER_URL, LLM_MODEL_NAME
from .guide import ExperimentGuide
from .llm_client import LLMClient
from .prompts import CHAT_PROMPT
from .tools import TOOL_DEFS, ToolExecutor


class AgentPhase(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    DONE = "done"
    STOPPED = "stopped"


class ExperimentPhase(str, Enum):
    """实验进展阶段（区别于 AgentPhase 的 Agent 自身运行状态）"""
    IDLE = "idle"               # 未开始
    SETUP = "setup"             # 检查硬件、调参
    HEATING = "heating"         # 升温中
    MEASURING = "measuring"     # 条纹计数中
    STABILIZING = "stabilizing" # 等稳定
    ANALYZING = "analyzing"     # 数据分析
    COMPLETE = "complete"       # 完成


@dataclass
class LogEntry:
    """决策日志条目"""
    time: str
    role: str          # "tool" / "system"
    content: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    tool_result: Optional[str] = None

    def to_dict(self):
        d = {"time": self.time, "role": self.role}
        if self.content:
            d["content"] = self.content[:500]
        if self.tool_name:
            d["tool"] = self.tool_name
            d["args"] = self.tool_args
        if self.tool_result:
            d["result"] = self.tool_result[:300]
        return d


class ExperimentAgent:
    """9 阶段实验指导 + 对话工具 + 反应层安全 Agent"""

    def __init__(self, llm_url: str = None):
        url = llm_url or LLM_SERVER_URL
        self.llm = LLMClient(base_url=url, model_name=LLM_MODEL_NAME)
        self.executor = ToolExecutor(self.llm)

        # 分步实验指导
        self.guide = ExperimentGuide()

        # 运行状态
        self.phase: AgentPhase = AgentPhase.IDLE
        self.goal: str = ""
        self.log: list[LogEntry] = []
        self.report: Optional[str] = None
        # 对话历史 (chat 模式记忆)
        self._chat_history: list[dict] = []
        self._chat_max_turns: int = 20  # 保留最近 20 轮对话
        # 可调推理参数 (前端设置面板实时修改)
        self._temperature: float = 0.5
        self._max_tokens: int = 2048
        # 深度思考开关 (前端聊天区可切, 类似 DeepSeek): 同步到 llm.think_default
        self._thinking: bool = True

        # ── 实验叙事状态 ──
        self.exp_phase: ExperimentPhase = ExperimentPhase.IDLE
        self._phase_history: list[tuple] = []  # (phase, timestamp_str)
        self._narrative_log: list[dict] = []   # [{time, text, type}]
        self._narration_ws_clients: list = []  # WebSocket 客户端列表

    # ═══════════════════════════════════════════════════════════
    # 公开接口
    # ═══════════════════════════════════════════════════════════

    def stop(self):
        """停止实验指导。"""
        if self.guide.active:
            self.guide.stop()
        self.phase = AgentPhase.STOPPED
        self._log("system", content="Agent 被用户中断")

    # ═══ 分步实验指导 ═══

    def start_guide(self) -> dict:
        """开始实验 = 启动分阶段指导 (agent 驱动)"""
        if self.guide.active:
            return {"ok": False, "msg": "实验指导已在进行中"}
        payload = self.guide.start()
        self.phase = AgentPhase.RUNNING
        self.goal = "分阶段指导学生完成热膨胀系数测量实验"
        self._log("system", content="实验指导已启动 (9 阶段流程)")
        self._push_narration(
            "实验开始！我会分阶段带你完成：安装样品→调光路→相机成像→升温扫描→数据处理。",
            "guide")
        return {"ok": True, "msg": "实验指导已启动", "step": payload}

    def stop_guide(self) -> dict:
        """结束实验指导"""
        was_active = self.guide.active
        self.guide.stop()
        if self.phase == AgentPhase.RUNNING:
            self.phase = AgentPhase.STOPPED
        if was_active:
            self._log("system", content="实验指导已结束")
        return {"ok": True, "msg": "实验指导已结束"}

    def guide_status(self) -> dict:
        return self.guide.status()

    def guide_next(self, force: bool = False) -> dict:
        """进入下一阶段 (软门控: 未达标返回 warn, force 可跳过)"""
        r = self.guide.next(force=force)
        if r.get("ok"):
            if r.get("finished"):
                self.phase = AgentPhase.DONE
                self._push_narration("全部流程完成！实验收尾后可以让 Agent 生成实验总结。", "guide")
            else:
                self._push_narration(
                    f"进入第{r.get('step_no')}阶段：{r.get('title')}", "guide")
        return r

    def guide_prev(self) -> dict:
        return self.guide.prev()

    def _append_guide_message(self, text: str):
        """把讲解或验证结果写入桌面端与移动端共享的聊天历史。"""
        msg = {"role": "assistant", "content": text}
        self._chat_history.append(msg)

    async def guide_briefing_stream(self):
        """为当前阶段生成 agent 讲解, SSE 流式 (think/tok/done).

        组装: 阶段简报提示 + RAG 知识 + 实时状态 + (vision 阶段 attach 当前帧)。
        完成后把讲解文本存 guide._briefing_text 并 append 进双端聊天历史。
        """
        from .prompts import GUIDE_BRIEFING_PROMPT
        from .llm_client import LLMClient
        guide = self.guide
        if not guide.active:
            yield {"t": "error", "d": "实验指导未进行"}
            return
        spec = guide.phase_spec(guide.idx)

        # RAG 参考
        rag_ctx = ""
        if spec.get("rag_query"):
            try:
                from ..rag import rag_store
                if rag_store.ready:
                    rag_ctx = await asyncio.to_thread(
                        rag_store.build_context, spec["rag_query"], 3, 1500)
            except Exception:
                pass

        # 实时状态
        scan_r = state.scan_runner
        scan_st = ("已完成" if (scan_r and getattr(scan_r, "phase", "") == "done")
                   else ("运行中" if state.scan_active else "未启动"))
        cam_st = "取流中" if (state.camera and state.camera.is_grabbing) else "未取流"
        n_now = abs(state.fringe_counter.total_N[0]) if state.fringe_counter else 0
        state_str = (f"温度 {state.current_T:.1f}°C | 质量 Q {state.current_quality:.2f} | "
                     f"N {n_now:.1f} | 相机 {cam_st} | 扫描 {scan_st}")

        content = GUIDE_BRIEFING_PROMPT.format(
            phase=f"第{guide.idx + 1}/{guide.total}阶段《{spec['title']}》",
            objective=spec["objective"],
            actions="\n".join(f"- {a}" for a in spec["actions"]),
            checkpoint=spec["checkpoint"],
            state=state_str,
            reference=rag_ctx or "(无)",
        )

        # 消息 (vision 阶段 attach 当前帧)
        parts = []
        if spec.get("vision"):
            with state._frame_lock:
                frame = state.latest_frame
            if frame is not None:
                b64 = LLMClient.frame_to_base64(frame)
                if b64:
                    parts.append({"type": "image_url",
                                  "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        parts.append({"type": "text", "text": content})
        messages = [
            {"role": "system", "content": "你是实验分步指导搭档，结合画面与知识库讲解当前阶段。"},
            {"role": "user", "content": parts},
        ]

        buf = ""
        try:
            # think=False: qwen3-vl 深度思考会吃掉 token 预算导致内容为空 (同反应层告警)
            # 讲解场景不向用户展示思考过程: 丢弃 think 事件, 只转发 tok/error/done
            async for ev in self.llm.chat_stream(messages, temperature=0.4,
                                                 max_tokens=800, think=False):
                if ev["t"] == "think":
                    continue
                if ev["t"] == "tok":
                    buf += ev["d"]
                yield ev
        except Exception as e:
            yield {"t": "error", "d": f"讲解生成失败: {e}"}
            return
        text = buf.strip()
        guide._briefing_text = text
        if text:
            self._append_guide_message(f"🎓 第{guide.idx + 1}阶段《{spec['title']}》指导：\n{text}")
        yield {"t": "done", "tool_trace": [], "usage": {}}

    async def guide_verify(self) -> dict:
        """运行当前阶段验证: state 直接读, vision 调一次 VLM."""
        from .prompts import VISION_PROMPT
        from .llm_client import LLMClient
        guide = self.guide
        if not guide.active:
            return {"ok": False, "msg": "实验指导未进行"}
        spec = guide.phase_spec(guide.idx)
        vision = None
        if spec.get("verify", {}).get("type") == "vision":
            with state._frame_lock:
                frame = state.latest_frame
            if frame is None:
                return {"ok": True, "passed": None,
                        "detail": "相机无画面，请先启动取流", "vision": None}
            b64 = LLMClient.frame_to_base64(frame)
            vision = await self.llm.ask_with_image(VISION_PROMPT, b64)
            t = vision or ""
            bad_kw = ["椭圆", "斜条纹", "模糊", "消失", "过曝", "无条纹", "没有条纹",
                      "断裂", "全白", "全黑", "无法识别"]
            good_kw = ["居中", "中心", "正中", "对称"]
            passed = any(k in t for k in good_kw) and not any(k in t for k in bad_kw)
            guide._vision_check = {
                "passed": bool(passed),
                "detail": "AI 视觉判定：" + (t.strip()[:150]),
                "key": "vision",
            }
            # 每次检验都存档条纹帧供报告 (多次点击取最后一次, 覆盖前次)
            if frame is not None:
                self._save_report_frame(frame, "fringe_fine.png")
        chk = guide.check_passed()
        return {"ok": True, "passed": chk["passed"], "detail": chk["detail"],
                "vision": vision}

    def _save_report_frame(self, frame, name: str):
        """实验过程画面存档 (报告图表: 细调条纹帧/过程截图) 到 measurements/report_assets/"""
        try:
            import cv2
            from ..config import MEASURE_SAVE_DIR
            d = MEASURE_SAVE_DIR / "report_assets"
            d.mkdir(parents=True, exist_ok=True)
            bgr = (cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                   if frame.ndim == 3 and frame.shape[2] == 3 else frame)
            cv2.imwrite(str(d / name), bgr)
            print(f"  [Report] 画面存档: {d / name}")
        except Exception as e:
            print(f"  [Report] 画面存档失败 {name}: {e}")

    async def generate_report(self) -> dict:
        """生成实验报告 PDF: 实测数据 + LLM 结论 + VLM 评价 + reportlab 排版 (LaTeX 公式/图表)"""
        from pathlib import Path
        from ..config import (LASER_WAVELENGTH_NM, SAMPLE_LENGTH_MM, REFERENCE_ALPHA,
                              CAMERA_EXPOSURE_US, CAMERA_GAIN_DB, CAMERA_FPS,
                              MEASURE_SAVE_DIR)
        from .prompts import CONCLUSION_PROMPT, VISION_PROMPT
        from .report_pdf import build_report_pdf, latex_available
        from .llm_client import LLMClient

        sr = state.scan_runner
        if sr is None or not getattr(sr, "segments", None):
            return {"ok": False, "msg": "尚无扫描数据，请先完成扫描再生成报告"}
        res = sr.result()
        segments = res["segments"]

        mode_name = {"temp": "温度步进", "fringe": "N 步进",
                     "free": "即时计数"}.get(sr.mode, sr.mode)
        params = [
            ("激光波长 λ", f"{LASER_WAVELENGTH_NM} nm"),
            ("样品长度 L₀", f"{SAMPLE_LENGTH_MM:.0f} mm"),
            ("参考 α (H62)", f"{REFERENCE_ALPHA}×10⁻⁶/K"),
            ("扫描模式", mode_name),
            ("扫描范围", f"{sr.T_start:.0f} → {sr.T_end:.0f} °C"),
            ("步长", (f"{sr.step} °C" if sr.mode != "fringe"
                      else f"{sr.n_step} 条/段")),
            ("相机曝光", f"{CAMERA_EXPOSURE_US} μs"),
            ("相机增益/帧率", f"{CAMERA_GAIN_DB} dB / {CAMERA_FPS} fps"),
        ]
        steps = [(h["title"], h.get("time"))
                 for h in self.guide.history] if self.guide.history else []

        # ── VLM 评价: 光路细调验证记录 + 实时对当前帧评价 ──
        vlm_history = []
        vc = self.guide._vision_check
        if vc and vc.get("detail"):
            vlm_history.append({"time": datetime.now().strftime("%H:%M:%S"),
                                "scene": "光路细调验证",
                                "quality_cnn": round(state.current_quality, 3),
                                "analysis": vc.get("detail", "")[:150]})
        with state._frame_lock:
            frame = state.latest_frame
        vlm_note = ""
        if frame is not None:
            try:
                b64 = LLMClient.frame_to_base64(frame)
                if b64:
                    va = await self.llm.ask_with_image(VISION_PROMPT, b64)
                    if va:
                        vlm_history.append({"time": datetime.now().strftime("%H:%M:%S"),
                                            "scene": "报告时条纹质量",
                                            "quality_cnn": round(state.current_quality, 3),
                                            "analysis": (va or "")[:150]})
                        vlm_note = (va or "")[:200]
            except Exception as e:
                print(f"  [Report] VLM 评价失败: {e}")

        # ── LLM 生成结论段 (对比参考值 + AI 置信度评价 + 误差来源) ──
        seg_txt = "\n".join(
            f"段{rec['seg']}: {rec['T1']}→{rec['T2']}°C, N={rec['N']:.2f}, "
            f"ΔT={rec['dT']}°C, α={rec['alpha']}" for rec in segments)
        data_txt = (f"共 {len(segments)} 段\n{seg_txt}\n"
                    f"α 均值 = {res['alpha_avg']} ± {res['alpha_std']}×10⁻⁶/K, "
                    f"相对误差 {res['error_pct']}% (参考值 {res['reference_alpha']})")
        conclusion = ""
        try:
            resp = await self.llm.chat(
                [{"role": "user",
                  "content": CONCLUSION_PROMPT.format(data=data_txt,
                                                      vlm=vlm_note or "(无)")}],
                temperature=0.4, max_tokens=500, think=False)
            conclusion = (resp.content or "").strip()
        except Exception as e:
            print(f"  [Report] 结论生成失败: {e}")
        if not conclusion:
            conclusion = (f"实测 α 均值为 {res['alpha_avg']} ± {res['alpha_std']}×10⁻⁶/K，"
                          f"与黄铜参考值 20.8×10⁻⁶/K 的相对误差为 {res['error_pct']}%。"
                          f"主要误差来源为温度测量热滞后与条纹计数不确定性。")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = MEASURE_SAVE_DIR / "reports" / f"report_{ts}.pdf"

        # 图表资产: 粗调光点照片 + 细调条纹帧
        assets = Path(MEASURE_SAVE_DIR) / "report_assets"
        coarse = (str(assets / "coarse_photo.png")
                  if (assets / "coarse_photo.png").exists() else None)
        fine = (str(assets / "fringe_fine.png")
                if (assets / "fringe_fine.png").exists() else None)
        # φ(s) / 通道分散度: 最近段相位解调统计
        error_stats = None
        try:
            if state.fringe_counter is not None:
                error_stats = state.fringe_counter.get_error_stats()
        except Exception:
            pass

        ctx = {
            "title": "基于计算机视觉和轻量VLM的干涉法测量热膨胀实验报告",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "segments": segments,
            "alpha_avg": res["alpha_avg"], "alpha_std": res["alpha_std"],
            "error_pct": res["error_pct"],
            "params": params, "steps": steps,
            "vlm_history": vlm_history,
            "conclusion": conclusion,
            "n_history": list(state.n_history),
            "temp_history": list(state.temp_history),
            "t_scan_start": getattr(sr, "t_scan_start", None),
            "coarse_photo": coarse, "fine_frame": fine,
            "error_stats": error_stats,
            "disp_history": list(getattr(state, "disp_history", [])),
            "latex_degraded": not latex_available(),
        }
        try:
            # 超时兜底: PDF 排版(图表/LaTeX)最坏 120s, 超时返回 ok:False
            info = await asyncio.wait_for(
                asyncio.to_thread(build_report_pdf, ctx, pdf_path), timeout=120)
        except asyncio.TimeoutError:
            return {"ok": False, "msg": "报告生成超时 (PDF 排版耗时过长)"}
        except FileNotFoundError as e:
            return {"ok": False, "msg": str(e)}
        except Exception as e:
            print(f"  [Report] PDF 构建失败: {e}")
            return {"ok": False, "msg": f"报告生成失败: {e}"}

        self.report = conclusion
        size_kb = round(pdf_path.stat().st_size / 1024, 1)
        url = f"/measurements/reports/{pdf_path.name}"
        self._log("system", content=f"实验报告已生成: {pdf_path.name} "
                                    f"(LaTeX={'真' if info.get('latex_used') else '降级'})")
        self._append_guide_message(
            f"📄 实验报告已生成：{pdf_path.name}（{size_kb} KB），可在聊天窗口打开或下载。")
        return {"ok": True, "pdf_url": url, "pdf_name": pdf_path.name,
                "size_kb": size_kb, "latex_used": info.get("latex_used", False)}

    async def _prepare_chat(self, prompt: str, use_frame: bool,
                            image_b64: Optional[str],
                            history: Optional[list] = None,
                            web_search: bool = False,
                            file_content: Optional[str] = None,
                            file_name: Optional[str] = None):
        """对话准备: 构建消息/历史/路由/RAG/联网搜索/文件/系统提示词
        history: 自定义对话历史（缺省使用桌面端与移动端共享的历史）
        web_search: 🌐 联网搜索开关 (Bing, 结果注入参考资料)
        file_content/file_name: 用户上传文件提取的文本 (注入参考资料)
        返回 (messages, chat_sys, content, needs_tools)
        """
        # 构建当前用户消息
        content = []
        if image_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
            })
        elif use_frame:
            with state._frame_lock:
                frame = state.latest_frame
            if frame is not None:
                b64 = LLMClient.frame_to_base64(frame)
                if b64:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                    })
        content.append({"type": "text", "text": prompt})
        history = self._chat_history if history is None else history

        # 历史中旧图片替换为占位文本: 避免每轮重发历史 base64,
        # 多轮带图对话后请求体可达数 MB, 上下文很快耗尽
        self._strip_history_images(history)

        # 追加到对话历史
        history.append({"role": "user", "content": content})

        # 裁剪历史 (in-place, 保持传入的 history 引用有效, 移动端全局列表也能被裁剪)
        max_msgs = self._chat_max_turns * 2
        if len(history) > max_msgs:
            del history[:len(history) - max_msgs]

        # 对话路由: 动词门控判断工具诉求; 知识诉求触发 RAG (两者可同时成立)
        from ..rag.store import route_query
        needs_tools, needs_rag = route_query(prompt)

        # 附带图片时聚焦图像分析: 不注入 RAG (避免参考资料把回答带向背理论),
        # 并提示模型先描述图中实际画面再回答
        img_attached = any(isinstance(p, dict) and p.get("type") == "image_url"
                           for p in content)
        if img_attached:
            needs_rag = False
            content[-1]["text"] += "\n(请先仔细观察图中实际画面，描述你看到的内容，再回答问题)"

        # RAG 检索增强: 知识问答类请求触发 (操作纯指令不触发, 避免无谓嵌入开销)
        rag_context = ""
        if needs_rag:
            try:
                from ..rag import rag_store
                if rag_store.ready:
                    rag_context = await asyncio.to_thread(
                        rag_store.build_context, prompt, 3, 1800)
            except Exception:
                pass

        # 注入实验上下文到对话提示词
        ctx_parts = [f"实验阶段: {self.exp_phase.value}"]
        if self.guide.active:
            cur = self.guide.current_phase_payload()
            ctx_parts.append(
                f"实验指导进行中: 第{cur['step_no']}/{cur['total']}阶段 《{cur['title']}》")
        ctx_parts.append(f"温度: {state.current_T:.1f}°C")
        ctx_parts.append(f"条纹质量: {state.current_quality:.2f}")
        if state.fringe_counter:
            ctx_parts.append(f"条纹计数 N: {abs(state.fringe_counter.total_N[0]):.2f}")
        if self._narrative_log:
            ctx_parts.append(f"最近观察: {self._narrative_log[-1]['text']}")
        context_str = " | ".join(ctx_parts)
        chat_sys = CHAT_PROMPT.replace("{context}", context_str)

        # 参考资料 (RAG 知识库 + 🌐 联网搜索 + 用户上传文件), 仅当轮注入不污染历史
        ref_parts: list[str] = []
        if rag_context:
            ref_parts.append(rag_context)
        if web_search:
            try:
                from ..search import build_context
                web_ctx = await asyncio.to_thread(build_context, prompt)
                if web_ctx:
                    ref_parts.append(web_ctx)
            except Exception:
                pass
        if file_content:
            ref_parts.append(f"【用户上传文件: {file_name or '附件'}】\n{file_content[:4000]}")
        ref_context = "\n\n".join(ref_parts)

        if ref_context:
            messages = [{"role": "system", "content": chat_sys}]
            messages += history[:-1]  # 历史(不含当轮)
            # 参考资料注入时保留当轮图片 part (修复: 旧实现重构纯文本 prompt 丢弃图片,
            # 带图 + 联网搜索/文件 时图片不发给模型)
            last = history[-1] if history else None
            img_parts = []
            if last and isinstance(last.get("content"), list):
                img_parts = [p for p in last["content"]
                             if isinstance(p, dict) and p.get("type") == "image_url"]
            new_text = f"参考资料:\n{ref_context}\n\n请结合资料回答(不相关可忽略):\n{prompt}"
            user_content = (img_parts + [{"type": "text", "text": new_text}]
                            if img_parts else new_text)
            messages.append({"role": "user", "content": user_content})
        else:
            messages = [{"role": "system", "content": chat_sys}] + history

        return messages, chat_sys, content, needs_tools

    async def chat(self, prompt: str, use_frame: bool = True,
                   image_b64: Optional[str] = None,
                   web_search: bool = False,
                   file_content: Optional[str] = None,
                   file_name: Optional[str] = None) -> dict:
        """手动对话模式 (带对话历史记忆 + 工具调用能力)

        image_b64: 前端点击"附带画面"时拍的快照 base64 — 优先使用,
        保证对话里展示的图与发给模型分析的图完全一致。
        web_search: 🌐 联网搜索开关.
        file_content/file_name: 用户上传文件提取的文本.
        """
        messages, chat_sys, content, needs_tools = await self._prepare_chat(
            prompt, use_frame, image_b64,
            web_search=web_search, file_content=file_content, file_name=file_name)

        # 调用 LLM
        resp = await self.llm.chat(messages, tools=TOOL_DEFS if needs_tools else None,
                                   temperature=self._temperature,
                                   max_tokens=self._max_tokens)

        # 收集深度思考内容与工具调用轨迹 (前端可折叠展示)
        thinking_parts = [resp.thinking] if resp.thinking else []
        tool_trace: list[dict] = []

        # 伪工具调用兼容: 模型在长上下文下把 function calling 退化成文本输出时,
        # 先用干净上下文重试一次; 仍失败则正则解析后直接执行工具
        if (not resp.has_tool_calls and needs_tools
                and self._looks_like_pseudo_tool(resp.content)):
            retry_msgs = [{"role": "system", "content": chat_sys},
                          {"role": "user", "content": content}]
            resp_retry = await self.llm.chat(retry_msgs, tools=TOOL_DEFS,
                                             temperature=self._temperature,
                                             max_tokens=self._max_tokens)
            if resp_retry.thinking:
                thinking_parts.append(resp_retry.thinking)
            if resp_retry.has_tool_calls:
                resp = resp_retry
            else:
                parsed = self._parse_pseudo_tool(resp_retry.content or resp.content or "")
                if parsed:
                    name, args = parsed
                    result_str = await self.executor.execute(name, args)
                    tool_trace.append({"name": name, "args": args,
                                       "result": result_str[:300]})
                    reply = f"✅ 已执行 {name}({json.dumps(args, ensure_ascii=False)})\n结果: {result_str[:200]}"
                    self._chat_history.append({"role": "assistant", "content": reply})
                    return {"ok": True, "text": reply,
                            "thinking": "\n\n".join(thinking_parts) or None,
                            "tool_trace": tool_trace, "usage": resp.usage}

        # 如果模型返回工具调用, 执行并获取最终回复
        if resp.has_tool_calls:
            # 记录 assistant 工具调用消息
            assistant_msg = {"role": "assistant", "content": resp.content or ""}
            assistant_msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)}}
                for tc in resp.tool_calls
            ]
            self._chat_history.append(assistant_msg)

            # 执行工具
            tool_results_text = []
            for tc in resp.tool_calls:
                result_str = await self.executor.execute(tc.name, tc.arguments)
                self._chat_history.append({
                    "role": "tool", "tool_call_id": tc.id, "content": result_str
                })
                tool_results_text.append(f"[{tc.name}] {result_str[:200]}")
                tool_trace.append({"name": tc.name, "args": tc.arguments,
                                   "result": result_str[:300]})

            # 再次调用 LLM 获取最终回复 (带工具结果)
            messages2 = [{"role": "system", "content": chat_sys}] + self._chat_history
            resp2 = await self.llm.chat(messages2, temperature=0.5)
            if resp2.thinking:
                thinking_parts.append(resp2.thinking)
            reply = resp2.content or "\n".join(tool_results_text)
        else:
            reply = resp.content or "无回复"

        # 记录 AI 回复到历史
        self._chat_history.append({"role": "assistant", "content": reply})

        return {"ok": True, "text": reply,
                "thinking": "\n\n".join(thinking_parts) or None,
                "tool_trace": tool_trace, "usage": resp.usage}

    async def chat_stream(self, prompt: str, use_frame: bool = True,
                          image_b64: Optional[str] = None,
                          history: Optional[list] = None,
                          web_search: bool = False,
                          file_content: Optional[str] = None,
                          file_name: Optional[str] = None):
        """流式对话 (async generator), 事件协议 (SSE 转发给前端):
          {"t":"think","d":str}     思考增量
          {"t":"tok","d":str}       回答增量
          {"t":"tool","name","args"}        工具开始执行
          {"t":"tresult","name","result"}   工具执行结果
          {"t":"done","tool_trace","usage"} 全部完成
          {"t":"error","d":str}     错误
        history: 自定义对话历史（缺省使用桌面端与移动端共享的历史）
        web_search: 🌐 联网搜索开关.
        file_content/file_name: 用户上传文件提取的文本.
        """
        history = self._chat_history if history is None else history
        messages, chat_sys, content, needs_tools = await self._prepare_chat(
            prompt, use_frame, image_b64, history,
            web_search=web_search, file_content=file_content, file_name=file_name)

        tool_trace: list[dict] = []
        content_buf: list[str] = []
        tool_calls = []
        usage = {}

        # ── 第一轮 ──
        # 注意: qwen3-vl + 当前 Ollama 版本下, 流式 + tools 组合会返回空响应
        # (思考后既不输出内容也不调工具)。因此工具轮用非流式(可靠),
        # 仅无工具的纯问答轮走流式。
        if needs_tools:
            resp = await self.llm.chat(
                messages, tools=TOOL_DEFS,
                temperature=self._temperature, max_tokens=self._max_tokens)
            if resp.thinking:
                yield {"t": "think", "d": resp.thinking}
            if resp.content:
                content_buf.append(resp.content)
                yield {"t": "tok", "d": resp.content}
            if resp.finish_reason == "error":
                # 修复: 错误分支统一收尾 (原逻辑仅处理有内容情形, 无内容时
                # 会漏到末尾 append "无回复" 进历史误导用户)
                if not resp.content:
                    yield {"t": "tok", "d": "❌ LLM 调用失败（无响应）"}
                yield {"t": "done", "tool_trace": [], "usage": {}}
                return
            tool_calls = resp.tool_calls
            usage = resp.usage
        else:
            async for ev in self.llm.chat_stream(
                    messages, temperature=self._temperature,
                    max_tokens=self._max_tokens):
                if ev["t"] in ("think", "tok"):
                    if ev["t"] == "tok":
                        content_buf.append(ev["d"])
                    yield ev
                elif ev["t"] == "finish":
                    usage = ev["usage"]
                elif ev["t"] == "error":
                    yield {"t": "tok", "d": f"❌ {ev['d']}"}
                    yield {"t": "done", "tool_trace": [], "usage": {}}
                    return

        full_content = "".join(content_buf)

        # ── 伪工具调用兜底: 正则解析后直接执行 ──
        if (not tool_calls and needs_tools
                and self._looks_like_pseudo_tool(full_content)):
            parsed = self._parse_pseudo_tool(full_content)
            if parsed:
                name, args = parsed
                yield {"t": "tool", "name": name, "args": args}
                result_str = await self.executor.execute(name, args)
                tool_trace.append({"name": name, "args": args,
                                   "result": result_str[:300]})
                yield {"t": "tresult", "name": name, "result": result_str[:300]}
                reply = f"✅ 已执行 {name}，结果：{result_str[:200]}"
                history.append({"role": "assistant", "content": reply})
                yield {"t": "tok", "d": reply}
                yield {"t": "done", "tool_trace": tool_trace, "usage": usage}
                return

        # ── 工具执行 + 第二轮流式汇总 ──
        if tool_calls:
            assistant_msg = {"role": "assistant", "content": full_content}
            assistant_msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.name,
                              "arguments": json.dumps(tc.arguments, ensure_ascii=False)}}
                for tc in tool_calls]
            history.append(assistant_msg)

            tool_results_text = []
            for tc in tool_calls:
                yield {"t": "tool", "name": tc.name, "args": tc.arguments}
                result_str = await self.executor.execute(tc.name, tc.arguments)
                history.append({
                    "role": "tool", "tool_call_id": tc.id, "content": result_str})
                tool_results_text.append(f"[{tc.name}] {result_str[:200]}")
                tool_trace.append({"name": tc.name, "args": tc.arguments,
                                   "result": result_str[:300]})
                yield {"t": "tresult", "name": tc.name, "result": result_str[:300]}

            # 第二轮: 带工具结果流式生成最终回复
            messages2 = [{"role": "system", "content": chat_sys}] + history
            content_buf2: list[str] = []
            async for ev in self.llm.chat_stream(messages2, temperature=0.5):
                if ev["t"] in ("think", "tok"):
                    if ev["t"] == "tok":
                        content_buf2.append(ev["d"])
                    yield ev
                elif ev["t"] == "finish":
                    usage = ev["usage"] or usage
                elif ev["t"] == "error":
                    break
            reply = "".join(content_buf2) or "\n".join(tool_results_text)
        else:
            reply = full_content or "无回复"

        history.append({"role": "assistant", "content": reply})
        yield {"t": "done", "tool_trace": tool_trace, "usage": usage}

    def _strip_history_images(self, history: Optional[list] = None):
        """把对话历史里的图片内容替换为占位文本 (只保留当轮新图).
        history 可传入自定义历史。"""
        history = self._chat_history if history is None else history
        for msg in history:
            c = msg.get("content")
            if isinstance(c, list):
                new_c = []
                for part in c:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        new_c.append({"type": "text", "text": "[历史图像帧已省略]"})
                    else:
                        new_c.append(part)
                msg["content"] = new_c

    # ═══ 伪工具调用兼容 ═══
    # MiniCPM-V 在多轮历史/长上下文下会把 function calling 退化成文本形式输出
    # (如 <function=set_temperature><parameter=target_c><=30.0> ...)。
    # 若不处理: 工具不会执行、用户只看到一段乱码文本 ("无回复"现象的根因)。
    # 对策: 1) 检测到伪格式 → 用干净上下文重试一次;
    #       2) 重试仍输出伪格式 → 正则解析后直接执行。

    _PSEUDO_TOOL_FN_RE = re.compile(r"<function\s*=\s*(\w+)\s*>", re.I)
    # 旧版末尾 `>?` 可选 + 惰性量词会让值匹配空串/截断 (伪工具兜底从未真正取值)。
    # 改为贪婪 [^<>]+ 值 ≥1 字符; 分隔符兼容 `<=` 与单个 `=`
    _PSEUDO_PARAM_RE = re.compile(
        r"<parameter\s*=\s*(\w+)\s*>\s*(?:<=|=)?\s*([^<>]+)\s*>?", re.I)

    def _looks_like_pseudo_tool(self, text: str) -> bool:
        return bool(text) and ("<function=" in text or "<tool_call>" in text)

    def _parse_pseudo_tool(self, text: str) -> Optional[tuple]:
        """从伪格式文本解析 (工具名, 参数dict), 失败返回 None。
        修复: 旧版 search 只匹配第一个 <parameter>, 多参数工具 (如
        set_camera_params) 的其余参数被静默丢弃; 现用 finditer 收集全部。"""
        fn = self._PSEUDO_TOOL_FN_RE.search(text)
        if not fn:
            return None
        name = fn.group(1)
        args: dict = {}
        for pm in self._PSEUDO_PARAM_RE.finditer(text):
            key, raw = pm.group(1), pm.group(2).strip()
            try:
                args[key] = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                args[key] = raw
        if not args:
            return None
        return name, args

    def clear_chat_history(self):
        """清空对话历史 (前端清空时调用)"""
        self._chat_history = []

    def update_config(self, temperature: float = None, max_tokens: int = None,
                      thinking: bool = None):
        """实时更新对话推理参数。"""
        if temperature is not None:
            self._temperature = max(0.0, min(1.0, temperature))
        if max_tokens is not None:
            self._max_tokens = max(128, min(8192, max_tokens))
        if thinking is not None:
            self._thinking = bool(thinking)
            self.llm.think_default = self._thinking
        print(f"  [Agent] 参数更新: temp={self._temperature}, "
              f"max_tok={self._max_tokens}, thinking={self._thinking}")

    def config_snapshot(self) -> dict:
        """当前对话推理参数快照。"""
        return {
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "thinking": self._thinking,
        }

    def status(self) -> dict:
        """Agent 当前状态。"""
        return {
            "phase": self.phase.value,
            "goal": self.goal,
            "has_report": self.report is not None,
            "log_tail": [e.to_dict() for e in self.log[-5:]],
            "thinking": self._thinking,
        }

    def get_log(self, last_n: int = 50) -> list:
        return [e.to_dict() for e in self.log[-last_n:]]

    def get_report(self) -> Optional[str]:
        return self.report

    async def health(self) -> dict:
        ok = await self.llm.health_check()
        return {"llm_online": ok, "url": self.llm.base_url, "model": self.llm.model_name}

    # ═══════════════════════════════════════════════════════════
    # 实验叙事
    # ═══════════════════════════════════════════════════════════

    def _push_narration(self, text: str, ntype: str = "system"):
        """推入叙事日志 + 异步推送 WebSocket 客户端"""
        entry = {"time": datetime.now().strftime("%H:%M:%S"),
                 "text": text, "type": ntype}
        self._narrative_log.append(entry)
        if len(self._narrative_log) > 50:
            self._narrative_log = self._narrative_log[-50:]
        # 异步推送，不阻塞当前请求。
        for ws in list(self._narration_ws_clients):
            try:
                import asyncio
                asyncio.get_event_loop().create_task(
                    ws.send_json({"type": "narration", **entry}))
            except Exception:
                self._narration_ws_clients.remove(ws)

    def _detect_phase_transition(self):
        """根据系统状态推断实验阶段，阶段变化时生成解说"""
        old = self.exp_phase
        if state.scan_active and state.scan_runner:
            sr = state.scan_runner
            if "heat" in str(sr.phase).lower() or "升温" in str(getattr(sr, 'message', '')):
                new = ExperimentPhase.HEATING
            elif "stab" in str(sr.phase).lower() or "稳定" in str(getattr(sr, 'message', '')):
                new = ExperimentPhase.STABILIZING
            else:
                new = ExperimentPhase.MEASURING
        elif state.measure_active:
            new = ExperimentPhase.MEASURING
        elif state.scan_runner and state.scan_runner.phase == "done":
            new = ExperimentPhase.COMPLETE
        elif self.phase == AgentPhase.RUNNING:
            new = ExperimentPhase.SETUP
        else:
            new = ExperimentPhase.IDLE

        if new != old:
            self.exp_phase = new
            ts = datetime.now().strftime("%H:%M:%S")
            self._phase_history.append((new.value, ts))
            if len(self._phase_history) > 20:
                self._phase_history = self._phase_history[-20:]
            # 阶段转换自动生成解说
            transitions = {
                ExperimentPhase.SETUP: "开始检查硬件和条纹状态，准备实验。",
                ExperimentPhase.HEATING: "温度开始上升，等待条纹响应。",
                ExperimentPhase.MEASURING: "条纹在吞吐，正在计数——样品在膨胀。",
                ExperimentPhase.STABILIZING: "温度接近目标，条纹在减速，快稳定了。",
                ExperimentPhase.COMPLETE: "扫描完成，数据收集齐全。",
            }
            if new in transitions:
                self._push_narration(transitions[new], "phase")

    def get_narrative(self, last_n: int = 20) -> list:
        """获取叙事日志"""
        return self._narrative_log[-last_n:]

    def get_experiment_phase(self) -> dict:
        """获取实验阶段信息"""
        return {
            "phase": self.exp_phase.value,
            "history": self._phase_history[-10:],
            "narrative_count": len(self._narrative_log),
        }

    def _log(self, role: str, content: str = None, tool_name: str = None,
             tool_args: dict = None, tool_result: str = None):
        """记录决策日志"""
        entry = LogEntry(
            time=datetime.now().strftime("%H:%M:%S"),
            role=role,
            content=content,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=tool_result,
        )
        self.log.append(entry)
        # 控制台关键输出
        if tool_name:
            print(f"  [Agent] {tool_name}({tool_args or ''})")
        elif content:
            preview = content[:80].replace('\n', ' ')
            print(f"  [Agent] {preview}...")
