"""分阶段实验指导 — agent 驱动的全流程引导

设计:
- 9 个阶段覆盖全流程: 开题安全→样品安装→光路粗调(光点)→光路细调(圆环,VLM验证)→
  相机成像→初始温度→自动扫描测量→数据处理误差→报告关机
- 每阶段含: 目标 / 操作 / 通过标准 / 讲解提示(rag_query) / 是否需视觉(vision) / 验证规范(verify)
- 验证分三类:
  * state : 读系统实时状态 (temp_online / camera_grabbing / quality / scan_done / scan_alpha / heat_off)
  * vision: 需 VLM 分析当前帧 (环心/对称/椭圆), 结果存 _vision_check
  * none  : 认知/手动, 学生确认即可
- next() 软门控: 本阶段 state 门未过 → 返回 warn(提醒但不阻止), force 可跳过
- 讲解文本由 agent 生成后存 _briefing_text, 前端轮询/刷新展示
"""
import time
from typing import Optional

from .. import state

# 指导阶段对条纹质量 Q 的要求 (比系统最低阈值 0.4 更严, 实验要求)
GUIDE_QUALITY_THRESHOLD = 0.75

# ═══════════════════════════════════════════════════════════
# 实验指导流程 (依据 docx《干涉法测量固体的线膨胀系数》+ kb_optical_path.md)
# ═══════════════════════════════════════════════════════════

GUIDE_PHASES = [
    {
        "title": "开题与安全",
        "objective": "了解实验原理与目标，牢记安全规则",
        "actions": [
            "牢记安全：眼睛不可直视激光束（包括扩束后的发散光）",
            "了解原理：α = N·λ/(2·L₁·ΔT)，每吞吐一条 ≈ 膨胀 λ/2 ≈ 316nm",
            "回顾全流程：安装样品 → 调光路 → 相机成像 → 升温扫描计数 → 数据处理",
        ],
        "checkpoint": "理解原理，明确安全与全流程",
        "verify": {"type": "none"},
        "vision": False,
        "rag_query": "迈克尔逊干涉 线膨胀系数 原理 安全",
        "tips": "黄铜 H62 参考 α ≈ 20.8×10⁻⁶/K；λ=632.8nm（氦氖）",
    },
    {
        "title": "样品安装与供电",
        "objective": "安装黄铜试件、连接动镜与测温探头，通电就绪",
        "actions": [
            "用游标卡尺测量并记录试件长度 L₁（标称 150mm）",
            "将试件放入温控炉：测温孔与炉侧面 PT100 探头圆孔对准，轻放（防砸碎底端石英玻璃垫）",
            "将带螺纹的反射镜3(动镜)与试件连接，不可拧得过紧（石英玻璃管易碎）",
            "PT100 探头经炉侧圆孔插入试件测温孔，插头接仪器后面板 PT100 接口",
            "接通电源：点亮氦氖激光器（半导体激光器接右侧 5V 电源），开启温控表",
        ],
        "checkpoint": "试件与 PT100 安装到位，激光器与温控表已通电",
        "verify": {"type": "state", "key": "temp_online"},
        "vision": False,
        "rag_query": "试件安装 PT100 反射镜3 温控炉 线膨胀",
        "tips": "更换试件时：先拧下反射镜3，再用 M4 螺钉取出试件",
    },
    {
        "title": "光路粗调·光点重合",
        "objective": "不加扩束镜，让两臂两个最亮光点重合",
        "actions": [
            "先移开扩束镜（此时激光是约 1mm 的细平行光束）",
            "调节激光器出射方向、反射镜1、反射镜2 和分束镜的方位",
            "在毛玻璃屏上找到两组光点，使其中两个最亮的光点完全重合",
            "拍摄光点照片：点『📷 拍摄粗调图』，系统自动从相机取当前帧存档并做 VLM 分析光点重合（进报告图 1）；如相机对激光饱和看不清，也可手机拍摄后手动上传",
        ],
        "checkpoint": "两组光点中最亮的两个完全重合（两臂主反射光束共轴）",
        "verify": {"type": "none"},
        "vision": False,   # 相机对未扩束激光会饱和 (kb §3.3), 手动目视
        "rag_query": "光路粗调 光点重合 不加扩束镜",
        "tips": "这一步看不到任何干涉条纹是完全正常的——没有扩束镜就没有扩展光源",
    },
    {
        "title": "光路细调·干涉圆环",
        "objective": "加扩束镜获得干涉圆环，调环心居中对称",
        "actions": [
            "将扩束镜放入光路：半导体激光器用扩束镜调节架，氦氖激光器用磁性镜架吸附在出光口",
            "仔细调节扩束镜位置 → 毛玻璃屏上出现干涉条纹（同心圆环）",
            "微调反射镜1、反射镜2 的倾斜螺丝，把干涉环调到便于观察的位置",
            "启动相机取流，让 AI 检查环心是否居中、环纹是否对称",
        ],
        "checkpoint": "明暗相间的同心圆环，环心居中、左右上下对称",
        "verify": {"type": "vision"},
        "vision": True,
        "rag_query": "扩束镜 干涉圆环 环心 对称 椭圆 等倾",
        "tips": "椭圆环说明两镜不平行；加扩束镜仍无圆环 → 回阶段3重新对光点",
    },
    {
        "title": "相机成像优化",
        "objective": "条纹成像清晰，质量分 Q 达标",
        "actions": [
            "观察质量分 Q（0~1）；Q 偏低时调曝光/Gamma（可对 Agent 说\"帮我优化相机参数\"）",
            "画面要求：亮暗分明、不过曝、环纹清晰",
            "环境杂散光强时会压低对比度，可遮光",
        ],
        "checkpoint": f"相机取流中，质量分 Q ≥ {GUIDE_QUALITY_THRESHOLD}",
        "verify": {"type": "state", "key": "quality"},
        "vision": False,
        "rag_query": "相机曝光 Gamma 质量分 条纹",
        "tips": "调参约 3 秒生效；画面全白=曝光过长或激光未扩束直射",
    },
    {
        "title": "初始温度基准",
        "objective": "确认温控在线、温度稳定，记录初始温度 t₁",
        "actions": [
            "确认温控器在线，PV 读数稳定",
            "记录初始温度 t₁（当前 PV 值）",
        ],
        "checkpoint": "温控在线、初始温度已记录",
        "verify": {"type": "state", "key": "temp_online"},
        "vision": False,
        "rag_query": "温控 PT100 PV 初始温度",
        "tips": "温度标签仅作参考记录，α 计算用扫描段起止的实测 PV",
    },
    {
        "title": "自动扫描与测量",
        "objective": "配置并启动自动温度扫描，全程监控条纹计数",
        "actions": [
            "配置扫描：推荐 30→50°C、步长 5°C，点击\"应用配置\"",
            "启动扫描（或对 Agent 说\"开始扫描\"）",
            "系统自动：进段确认 → 开始计数 → 逐段升温 → 段末结算 α",
            "监控：条纹有节奏吞吐、N 随温度增长；质量骤降/条纹消失/N 不动时立即询问 Agent",
        ],
        "checkpoint": "扫描完成，各段 N/ΔT/α 已结算",
        "verify": {"type": "state", "key": "scan_done"},
        "vision": False,
        "rag_query": "温度扫描 配置 条纹计数 吞吐",
        "tips": "相机未取流时扫描会被拒绝；温度在设定点附近回摆属正常，带符号计数自动抵消往复",
    },
    {
        "title": "数据处理与误差",
        "objective": "读取 α、对比参考值、分析误差来源",
        "actions": [
            "查看段表：各段 T1/T2/N/α，α 均值 ± 标准差",
            "与黄铜参考值 20.8×10⁻⁶/K 对比，计算相对误差",
            "分析误差来源：温度测量(主导)/条纹计数/试件与动镜接触传动",
            "可对 Agent 说\"帮我分析误差\"或\"为什么 α 偏大\"",
        ],
        "checkpoint": "拿到 α 与误差，能说明主要误差来源",
        "verify": {"type": "state", "key": "scan_alpha"},
        "vision": False,
        "rag_query": "误差分析 温度测量 条纹计数 线膨胀系数",
        "tips": "仪器系统误差标称 <3%；误差超此范围优先检查 ΔT 来源与段起止温度",
    },
    {
        "title": "报告与关机",
        "objective": "生成实验报告，安全关机收尾",
        "actions": [
            "点击\"生成实验报告\"，让 Agent 生成完整报告（数据+误差+物理解释）",
            "停止加热，设定温度调到室温以下，等待温控炉冷却",
            "冷却后关闭温控电源与激光器电源，光学元件防尘归位",
            "更换试件时先拧下反射镜3",
        ],
        "checkpoint": "报告已生成，加热已关闭",
        "verify": {"type": "state", "key": "heat_off"},
        "vision": False,
        "rag_query": "关机 安全 报告 收尾",
        "tips": "动镜石英玻璃管与炉内石英玻璃垫均为易碎件，操作轻缓",
    },
]

# auto_check 键 → 实时状态判定
def _eval_checks() -> dict:
    # 帧源在线: 相机取流 或 视频模拟源已加载 (视频模式 camera 可能未取流)
    cam_grab = state.camera is not None and state.camera.is_grabbing
    frame_online = bool(cam_grab or state.video_source is not None)
    scan_r = state.scan_runner
    progress_ok = False
    if scan_r is not None and state.scan_active:
        progress_ok = getattr(scan_r, "current_seg", 0) > 0
    has_segments = bool(getattr(scan_r, "segments", None)) if scan_r else False
    # free 模式无自然完成 (phase 保持 'stopped' 防前端误判"扫描完成"),
    # 但已结算出段即视为完成, 供指导阶段 7/8 自动验证放行
    scan_done = bool(scan_r is not None and (
        getattr(scan_r, "phase", "") == "done"
        or (getattr(scan_r, "mode", "") == "free" and has_segments)))
    return {
        "temp_online": bool(state.temp_online),
        "camera_grabbing": bool(cam_grab),
        "quality": bool(frame_online and state.current_quality >= GUIDE_QUALITY_THRESHOLD),
        "scan_active": bool(state.scan_active),
        "scan_progress": bool(progress_ok or scan_done),
        "scan_done": bool(scan_done),
        "scan_alpha": bool(scan_done and has_segments),
        "heat_off": bool(round(state.current_mv, 1) == 0.0),
    }


_CHECK_DESC = {
    "temp_online": "温控器在线",
    "camera_grabbing": "相机取流中",
    "quality": f"条纹质量 Q ≥ {GUIDE_QUALITY_THRESHOLD}",
    "scan_done": "扫描已完成",
    "scan_alpha": "已获得各段 α",
    "heat_off": "加热已关闭",
}


class ExperimentGuide:
    """分阶段实验指导状态机 (agent 驱动)"""

    def __init__(self):
        self.active: bool = False
        self.finished: bool = False
        self.idx: int = 0
        self.started_at: float = 0.0
        self.history: list[dict] = []      # [{step, title, time}]
        self._vision_check: Optional[dict] = None   # 当前阶段 VLM 验证结果
        self._briefing_text: str = ""               # 当前阶段 agent 讲解文本

    @staticmethod
    def phase_spec(idx: int) -> dict:
        return GUIDE_PHASES[idx]

    @property
    def total(self) -> int:
        return len(GUIDE_PHASES)

    # ─── 生命周期 ────────────────────────────────────────────

    def start(self) -> dict:
        self.active = True
        self.finished = False
        self.idx = 0
        self.started_at = time.time()
        self.history = []
        self._vision_check = None
        self._briefing_text = ""
        return self.current_phase_payload()

    def stop(self):
        self.active = False

    # ─── 验证 ────────────────────────────────────────────────

    def check_passed(self) -> dict:
        """评估当前阶段验证结果 → {passed, detail, key}
        passed: True=达标 / False=未达标 / None=无需系统验证或尚未验证"""
        spec = self.phase_spec(self.idx)
        vtype = spec.get("verify", {}).get("type")
        if vtype == "none":
            return {"passed": None, "detail": "本阶段手动确认即可", "key": None}
        if vtype == "vision":
            if self._vision_check is None:
                return {"passed": None, "detail": "尚未 AI 视觉验证", "key": "vision"}
            return dict(self._vision_check)
        key = spec.get("verify", {}).get("key")
        checks = _eval_checks()
        passed = bool(checks.get(key))
        return {"passed": passed, "detail": _CHECK_DESC.get(key, key), "key": key}

    # ─── 步进 (软门控: 提醒但不阻止) ──────────────────────────

    def next(self, force: bool = False) -> dict:
        if not self.active:
            return {"ok": False, "msg": "指导未开始"}
        if not force:
            chk = self.check_passed()
            if chk["passed"] is False:
                return {
                    "ok": False, "warn": True,
                    "msg": f"本阶段尚未达标：{chk['detail']}。可以继续下一步，或先完成该阶段。",
                    "checks": chk,
                }
        phase = self.phase_spec(self.idx)
        self.history.append({"step": self.idx + 1, "title": phase["title"],
                             "time": time.strftime("%H:%M:%S")})
        if self.idx + 1 >= self.total:
            self.finished = True
            self.active = False
            return {"ok": True, "finished": True,
                    "msg": "🎉 实验全部流程完成！可以请 Agent 生成实验总结。"}
        self.idx += 1
        self._vision_check = None
        self._briefing_text = ""
        return {"ok": True, "finished": False, **self.current_phase_payload()}

    def prev(self) -> dict:
        if not self.active or self.idx == 0:
            return {"ok": False, "msg": "已是第一阶段"}
        self.idx -= 1
        self._vision_check = None
        self._briefing_text = ""
        return {"ok": True, **self.current_phase_payload()}

    # ─── 状态输出 ────────────────────────────────────────────

    def current_phase_payload(self) -> dict:
        spec = self.phase_spec(self.idx)
        chk = self.check_passed()
        return {
            "step_no": self.idx + 1,
            "total": self.total,
            "title": spec["title"],
            "objective": spec["objective"],
            "actions": spec["actions"],
            "checkpoint": spec["checkpoint"],
            "tips": spec.get("tips", ""),
            "verify": {"type": spec.get("verify", {}).get("type"),
                       "passed": chk["passed"],
                       "detail": chk["detail"]},
            "briefing": self._briefing_text,
            "rag_query": spec.get("rag_query", ""),
        }

    def status(self) -> dict:
        checks = _eval_checks()
        phases_brief = []
        for i, s in enumerate(GUIDE_PHASES):
            st = "done" if (i < self.idx or self.finished) else \
                 ("current" if (self.active and i == self.idx) else "pending")
            phases_brief.append({"no": i + 1, "title": s["title"], "state": st})
        d = {
            "active": self.active,
            "finished": self.finished,
            "step_no": self.idx + 1,
            "total": self.total,
            "elapsed_s": round(time.time() - self.started_at, 1) if self.started_at else 0,
            "phases": phases_brief,
            "steps": phases_brief,          # 兼容旧前端字段名
            "checks": checks,
            "history": self.history[-10:],
        }
        if self.active:
            d["current"] = self.current_phase_payload()
        return d
