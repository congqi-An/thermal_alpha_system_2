"""Agent 工具定义与执行器 — 包装系统现有能力为 OpenAI function calling 格式"""
import json

from .. import state
from ..config import (LASER_WAVELENGTH_NM, SAMPLE_LENGTH_MM, REFERENCE_ALPHA,
                      QUALITY_THRESHOLD)
from .llm_client import LLMClient

# ═══════════════════════════════════════════════════════════════════
# 工具定义 (OpenAI function calling 格式)
# ═══════════════════════════════════════════════════════════════════

TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "start_guide",
            "description": "【实验引导】启动分阶段实验指导: agent 按 9 个阶段带领用户完成全流程实验(开题安全→安装样品→调光路→相机成像→升温扫描→数据处理→报告)。用户说'开始实验'时优先调用本工具。",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_system_status",
            "description": "获取整个实验系统的当前状态, 包括相机、温控、测量、条纹质量等所有信息",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_camera_frame",
            "description": "获取当前相机画面的亮度/对比度统计与尺寸信息 (画面本身每轮已随态势附图, 无需重复获取)",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_camera_params",
            "description": "调节相机参数(曝光/增益/Gamma/帧率), 用于优化条纹成像质量",
            "parameters": {
                "type": "object",
                "properties": {
                    "exposure_us": {"type": "integer", "description": "曝光时间(微秒), 范围100-500000"},
                    "gain_db": {"type": "number", "description": "增益(dB), 范围0-20"},
                    "gamma": {"type": "number", "description": "Gamma值, 范围0.1-2.0, 小于1增亮暗部"},
                    "fps": {"type": "number", "description": "帧率, 范围1-120"},
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "camera_start_stop",
            "description": "开始或停止相机取流",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["start", "stop"], "description": "start开始取流, stop停止取流"}
                },
                "required": ["action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_temperature",
            "description": "读取当前温度控制器数据: PV(当前温度)、SV(设定温度)、MV(输出%)",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_temperature",
            "description": "设定温控器目标温度SV并启动PID加热(安全范围20~60°C)",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_c": {"type": "number", "description": "目标温度(摄氏度), 范围20~60"}
                },
                "required": ["target_c"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "start_measurement",
            "description": "开始条纹计数测量(重置计数器并开始累计)",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "stop_measurement",
            "description": "停止条纹计数测量并返回累计条纹数N",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_measurement",
            "description": "获取当前测量摘要(条纹数N、伸长量、帧数等), 不中断测量",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "configure_scan",
            "description": "配置自动温度扫描参数(起始温度、终止温度、步长、稳定时间). 返回 segments 为每段 [T1,T2] 区间, 向用户转述时直接用该值, 勿自行推算",
            "parameters": {
                "type": "object",
                "properties": {
                    "T_start": {"type": "number", "description": "起始温度°C, 默认30"},
                    "T_end": {"type": "number", "description": "终止温度°C, 默认50"},
                    "step": {"type": "number", "description": "每段温升步长°C, 默认5"},
                    "stabilize_s": {"type": "number", "description": "每段稳定等待时间(秒), 默认3"},
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "start_scan",
            "description": "启动自动温度扫描(多段升温+逐段条纹计数+计算α)",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "stop_scan",
            "description": "停止正在进行的自动温度扫描",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_scan_status",
            "description": "获取扫描进度(当前段、阶段、已完成段数据、当前温度)",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_scan_result",
            "description": "获取扫描最终结果(α均值、标准差、误差、各段明细)",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_fringe_image",
            "description": "用视觉模型分析当前条纹图像, 评估质量并给出调参建议",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wait_seconds",
            "description": "等待指定秒数(用于调参后等待效果生效、或等待温度稳定/扫描完成)",
            "parameters": {
                "type": "object",
                "properties": {
                    "seconds": {"type": "number", "description": "等待秒数, 最大300. 扫描期间建议120-180"}
                },
                "required": ["seconds"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_experiment_narrative",
            "description": "获取实验叙事上下文: 当前阶段、阶段历史、最近观察记录、关键指标时间线。用于了解实验进展全貌。",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "explain_phenomenon",
            "description": "生成面向用户的物理解说: 解释当前观察到的现象(条纹吞吐/静止/异常)背后的物理原因。输出通俗有温度。",
            "parameters": {
                "type": "object",
                "properties": {
                    "phenomenon": {"type": "string", "description": "要解释的现象描述, 如'条纹在快速吞吐'或'条纹突然静止'"}
                },
                "required": ["phenomenon"]
            }
        }
    },
]


# ═══════════════════════════════════════════════════════════════════
# 工具执行器
# ═══════════════════════════════════════════════════════════════════

class ToolExecutor:
    """工具执行器, 持有 LLM 客户端引用(用于视觉分析)"""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def execute(self, name: str, args: dict) -> str:
        """执行工具, 返回 JSON 字符串结果"""
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return json.dumps({"error": f"未知工具: {name}"}, ensure_ascii=False)
        try:
            result = await handler(args)
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as e:
            return json.dumps({"error": f"{name} 执行异常: {e}"}, ensure_ascii=False)

    # ─── 系统状态 ─────────────────────────────────────────────

    async def _tool_get_system_status(self, args: dict) -> dict:
        cam = state.camera
        grabbing = cam is not None and cam.is_grabbing
        return {
            "camera": {
                "connected": cam is not None and cam.is_open,
                "grabbing": grabbing,
                # 中文状态串供 LLM 直接引用, 避免把"已连接"误描述为"正在取流"
                "状态": "正在取流" if grabbing else ("已连接但未取流(需 camera_start_stop 启动)" if cam is not None and cam.is_open else "离线"),
                "resolution": f"{cam.width}x{cam.height}" if cam else "N/A",
            },
            "temperature": {
                "online": state.temp_online,
                "PV": round(state.current_T, 2),
                "SV": round(state.current_sv, 2),
                "MV": round(state.current_mv, 1),
            },
            "measurement": {
                "active": state.measure_active,
                "quality": round(state.current_quality, 3),
                "quality_threshold": QUALITY_THRESHOLD,
                "center": state.current_center,
                "r0s": state.current_r0s,
            },
            "scan": {
                "active": state.scan_active,
                "phase": state.scan_runner.phase if state.scan_runner else "N/A",
                "progress": f"{state.scan_runner.current_seg}/{state.scan_runner.total_seg}"
                           if state.scan_runner else "N/A",
            },
            "models": {
                "frame_analysis_net": state.frame_analysis_net is not None,
                "center_r0_net": state.center_r0_net is not None,
            },
            "params": {
                "laser_nm": LASER_WAVELENGTH_NM,
                "sample_mm": SAMPLE_LENGTH_MM,
                "reference_alpha": f"{REFERENCE_ALPHA}\u00d710\u207b\u2076/K (黄铜H62线膨胀系数参考值, 非角度)",
            }
        }

    # ─── 相机 ─────────────────────────────────────────────────

    async def _tool_get_camera_frame(self, args: dict) -> dict:
        with state._frame_lock:
            frame = state.latest_frame
        if frame is None:
            return {"error": "无可用画面(相机未取流)"}
        # 不返回 base64 (无消费方, 且会灌爆对话历史/日志);
        # 战术层每轮已随态势附图, 这里只给统计信息供文本推理
        import numpy as np
        g = frame.mean(axis=2) if frame.ndim == 3 else frame
        return {
            "shape": list(frame.shape),
            "brightness_mean": round(float(g.mean()), 1),
            "brightness_p99": round(float(np.percentile(g, 99)), 1),
            "contrast_std": round(float(g.std()), 1),
            "quality_cnn": round(state.current_quality, 3),
        }

    async def _tool_set_camera_params(self, args: dict) -> dict:
        cam = state.camera
        if cam is None:
            return {"error": "相机未初始化"}
        results = {}
        # LLM 输入不可信, 钳制到相机安全范围 (与 set_temperature 的钳制风格一致)
        if "exposure_us" in args:
            exp = int(min(max(int(args["exposure_us"]), 50), 500000))
            cam.set_exposure(exp)
            results["exposure_us"] = exp
        if "gain_db" in args:
            gain = float(min(max(float(args["gain_db"]), 0.0), 30.0))
            cam.set_gain(gain)
            results["gain_db"] = gain
        if "gamma" in args:
            gamma = float(min(max(float(args["gamma"]), 0.05), 5.0))
            cam.set_gamma(gamma)
            results["gamma"] = gamma
        if "fps" in args:
            fps = float(min(max(float(args["fps"]), 1.0), 120.0))
            cam.set_framerate(fps)
            results["fps"] = fps
        return {"ok": True, "set": results}

    async def _tool_camera_start_stop(self, args: dict) -> dict:
        cam = state.camera
        if cam is None:
            return {"error": "相机未初始化"}
        # 未真正打开 → 明确报错而非抛原始异常 (与 /api/camera/start 修复一致)
        if not getattr(cam, "is_open", False):
            return {"error": "相机未连接/未打开，无法取流。请检查相机供电与 GigE 连接，或提示用户重连。"}
        # 大小写归一: "START"/"Start" 不再误入 stop 分支 (修复)
        action = str(args.get("action", "start")).strip().lower()
        if action == "stop":
            try:
                cam.stop()
            except Exception as e:
                return {"error": f"停止取流失败: {e}"}
            return {"ok": True, "grabbing": bool(cam.is_grabbing)}
        try:
            cam.start()
        except Exception as e:
            return {"error": f"开始取流失败: {e}"}
        return {"ok": True, "grabbing": bool(cam.is_grabbing)}

    # ─── 温控 ─────────────────────────────────────────────────

    async def _tool_get_temperature(self, args: dict) -> dict:
        return {
            "online": state.temp_online,
            "PV": round(state.current_T, 2),
            "SV": round(state.current_sv, 2),
            "MV": round(state.current_mv, 1),
            "history_len": len(state.temp_history),
        }

    async def _tool_set_temperature(self, args: dict) -> dict:
        # LLM 输入不可信: 缺参/非法/越界一律拒绝, 绝不带默认值执行 (0°C 事故修复)
        if "target_c" not in args:
            return {"error": "缺少必要参数 target_c (目标温度 °C)"}
        try:
            target = float(args["target_c"])
        except (TypeError, ValueError):
            return {"error": f"target_c 非法: {args['target_c']}"}
        if not (20.0 <= target <= 60.0):
            return {"error": "安全限制: 目标温度须在 20~60°C 之间", "requested": target}
        if state.temp_ctrl is None or not state.temp_ctrl.online:
            return {"error": "温控器离线, 无法设定温度"}
        state.temp_ctrl.regulate_to(target)
        return {"ok": True, "target_c": target, "msg": f"已设定SV={target}°C, PID加热中"}

    # ─── 测量 ─────────────────────────────────────────────────

    async def _tool_start_measurement(self, args: dict) -> dict:
        # 前置检查: 相机或视频模拟源任一在线即可 (否则 N 恒为 0 误导用户)
        if state.video_source is None and (
                state.camera is None or not state.camera.is_grabbing):
            return {"error": "相机 OFFLINE(未取流), 禁止启动测量. 先 camera_start_stop(start)"}
        from ..measurement.phase_fringe_counter import FringeCounter
        with state.measure_lock:
            if state.measure_active:
                return {"status": "already_measuring"}
            if state.fringe_counter is None:
                state.fringe_counter = FringeCounter()
            state.fringe_counter.reset()
            state.n_gate = None  # N 振动门随之重置
            state.gated_N = 0.0
            state.gated_N_abs = 0.0
            state.measure_active = True
        return {"ok": True, "msg": "测量已开始, 条纹计数中", "T": round(state.current_T, 2)}

    async def _tool_stop_measurement(self, args: dict) -> dict:
        with state.measure_lock:
            state.measure_active = False
        if state.fringe_counter is None:
            return {"error": "无测量数据"}
        _, N_per = state.fringe_counter.total_N
        N_avg = state.gated_N_abs   # N 振动门 (惯性覆盖) 后
        return {
            "ok": True, "N_avg": round(N_avg, 2), "N_per_r0": N_per,
            "n_frames": state.fringe_counter.n_frames,
        }

    async def _tool_get_measurement(self, args: dict) -> dict:
        if state.fringe_counter is None:
            return {"measuring": False}
        from ..measurement.alpha_calc import delta_L_from_N
        _, N_per = state.fringe_counter.total_N
        N_avg = state.gated_N_abs   # N 振动门 (惯性覆盖) 后
        dL = delta_L_from_N(N_avg)
        return {
            "measuring": state.measure_active,
            "N_avg": round(N_avg, 2),
            "delta_L_um": round(dL * 1e6, 3),
            "n_frames": state.fringe_counter.n_frames,
            "T": round(state.current_T, 2),
            "quality": round(state.current_quality, 3),
        }

    # ─── 扫描 ─────────────────────────────────────────────────

    async def _tool_configure_scan(self, args: dict) -> dict:
        if state.scan_runner is None:
            return {"error": "scan_runner 未初始化"}
        result = state.scan_runner.configure(
            T_start=args.get("T_start"),
            T_end=args.get("T_end"),
            step=args.get("step"),
            stabilize_s=args.get("stabilize_s"),
        )
        if result.get("error"):
            return result
        # 配置有效且扫描未在运行 → 自动级联启动扫描 (LLM 一轮只返回一个工具,
        # 用户说"开始扫描/扫描"时 configure 后若不启动会停在原地等待)
        if not getattr(state.scan_runner, "_running", False):
            if state.video_source is None and (
                    state.camera is None or not state.camera.is_grabbing):
                result["scan_started"] = False
                result["scan_hint"] = "帧源离线, 未自动启动 (需先取流或加载视频)"
            else:
                ok = state.scan_runner.start()
                result["scan_started"] = bool(ok)
                if ok:
                    result["msg"] = "扫描已自动启动"
        return result

    async def _tool_start_scan(self, args: dict) -> dict:
        if state.scan_runner is None:
            return {"error": "scan_runner 未初始化"}
        # 前置检查: 相机或视频模拟源任一在线即可 (视频模拟时 camera 可能未取流)
        if state.video_source is None and (
                state.camera is None or not state.camera.is_grabbing):
            return {"error": "相机 OFFLINE(未取流), 禁止启动扫描. 先 camera_start_stop(start)"}
        ok = state.scan_runner.start()
        if ok:
            return {"ok": True, "msg": "扫描已启动",
                    "segments": state.scan_runner.total_seg}
        return {"error": "扫描启动失败(可能已在运行)"}

    async def _tool_stop_scan(self, args: dict) -> dict:
        if state.scan_runner is None:
            return {"error": "scan_runner 未初始化"}
        state.scan_runner.stop()
        return {"ok": True, "msg": "扫描已停止"}

    async def _tool_get_scan_status(self, args: dict) -> dict:
        if state.scan_runner is None:
            return {"active": False}
        full = state.scan_runner.status()
        # 压缩输出: 只保留摘要 + 最近2段明细, 避免上下文膨胀
        segments = full.get("segments", [])
        compressed = {
            "active": full.get("active"),
            "phase": full.get("phase"),
            "message": full.get("message"),
            "progress": f"{full.get('current_seg', 0)}/{full.get('total_seg', 0)}",
            "current_T": full.get("current_T"),
            "alpha_avg": full.get("alpha_avg"),
            "completed_segments": len(segments),
        }
        # 只附带最近 2 段明细
        if segments:
            compressed["recent_segments"] = segments[-2:]
        return compressed

    async def _tool_get_scan_result(self, args: dict) -> dict:
        if state.scan_runner is None:
            return {"error": "无扫描数据"}
        return state.scan_runner.result()

    # ─── 视觉分析 ─────────────────────────────────────────────

    async def _tool_analyze_fringe_image(self, args: dict) -> dict:
        """获取当前帧 + 调 LLM 视觉分析"""
        from .prompts import VISION_PROMPT
        with state._frame_lock:
            frame = state.latest_frame
        if frame is None:
            return {"error": "无可用画面"}
        b64 = LLMClient.frame_to_base64(frame)
        if b64 is None:
            return {"error": "帧编码失败"}
        analysis = await self.llm.ask_with_image(VISION_PROMPT, b64)
        return {"analysis": analysis, "quality_score": state.current_quality}

    # ─── 等待 ─────────────────────────────────────────────────

    async def _tool_wait_seconds(self, args: dict) -> dict:
        import asyncio, math
        # LLM 输入不可信: 非数值/负数/NaN 一律回退 1s (负值会令 asyncio.sleep 抛 ValueError)
        try:
            seconds = float(args.get("seconds", 1))
        except (TypeError, ValueError):
            seconds = 1.0
        if not math.isfinite(seconds) or seconds < 0:
            seconds = 1.0
        seconds = min(seconds, 300.0)
        await asyncio.sleep(seconds)
        return {"ok": True, "waited_s": seconds}

    # ─── 实验叙事 ─────────────────────────────────────────────

    async def _tool_get_experiment_narrative(self, args: dict) -> dict:
        """获取实验叙事上下文"""
        agent = state.agent
        if agent is None:
            return {"error": "Agent 未初始化"}
        phase_info = agent.get_experiment_phase()
        narrative = agent.get_narrative(10)
        metrics = {
            "temperature": round(state.current_T, 1),
            "quality": round(state.current_quality, 3),
            "N": abs(state.fringe_counter.total_N[0]) if state.fringe_counter else 0,
            "scan_active": state.scan_active,
            "scan_progress": (f"{state.scan_runner.current_seg}/{state.scan_runner.total_seg}"
                             if state.scan_runner and state.scan_active else "N/A"),
        }
        return {"phase": phase_info, "recent_narrative": narrative, "metrics": metrics}

    async def _tool_explain_phenomenon(self, args: dict) -> dict:
        """生成面向用户的物理解说"""
        phenomenon = args.get("phenomenon", "条纹变化")
        context = (
            f"当前温度: {state.current_T:.1f}\u00b0C, "
            f"条纹质量: {state.current_quality:.2f}, "
            f"N: {abs(state.fringe_counter.total_N[0]):.1f}条"
            if state.fringe_counter else f"当前温度: {state.current_T:.1f}\u00b0C"
        )
        prompt = (
            f"用1-2句通俗有温度的话解释这个实验现象的物理原因:\n"
            f"现象: {phenomenon}\n"
            f"上下文: {context}\n"
            f"原理: 迈克尔逊干涉, 条纹吞吐=样品膨胀推动反射镜, 每条=\u03bb/2\u2248316nm。"
        )
        resp = await self.llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.7, max_tokens=150)
        explanation = resp.content if resp.content else f"现象: {phenomenon}"
        if state.agent:
            state.agent._push_narration(explanation, "explain")
        return {"explanation": explanation}

    # ─── 分阶段实验指导 ───────────────────────────────────────

    async def _tool_start_guide(self, args: dict) -> dict:
        """启动分阶段实验指导 (agent 驱动全流程)"""
        if state.agent is None:
            return {"error": "Agent 未初始化"}
        r = state.agent.start_guide()
        if r.get("ok"):
            step = r.get("step", {})
            return {
                "ok": True,
                "msg": "实验指导已启动，当前阶段：" + step.get("title", ""),
                "step_no": step.get("step_no"),
                "total": step.get("total"),
                "phase_title": step.get("title"),
                "objective": step.get("objective", ""),
                "actions": step.get("actions", []),
                "checkpoint": step.get("checkpoint", ""),
                "tips": step.get("tips", ""),
            }
        return r
