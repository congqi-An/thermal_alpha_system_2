# 基于计算机视觉和轻量 VLM 的干涉法热膨胀测量系统

这是一个面向迈克尔逊干涉实验的本地 Web 系统。系统使用工业相机采集同心干涉环，以 CNN 完成光心、采样半径和画面质量感知，再用空间相位算法计算带方向的条纹变化量；温控、分段扫描、实验指导、问答和 PDF 报告在同一个 FastAPI 服务中协同运行。

本 README 只描述当前代码中的有效链路。历史实验数据、模型权重、知识库资料和比赛报告均作为项目资产保留，不属于运行时源码。

## 测量原理

样品热膨胀推动迈克尔逊干涉仪动镜。动镜每移动半个波长，视场中产生一次完整条纹吞吐：

```text
ΔL = Nλ / 2
α  = Nλ / (2L₀ΔT)
```

当前默认参数：

| 参数 | 默认值 |
|---|---:|
| He-Ne 激光波长 λ | 632.8 nm |
| 黄铜 H62 样品长度 L₀ | 150 mm |
| 参考线膨胀系数 | 20.8 × 10⁻⁶/K |
| 工业相机帧率 | 20 fps |
| 标准扫描范围 | 30 → 50 ℃ |
| 标准扫描步长 | 5 ℃ |
| 系统安全温度上限 | 60 ℃ |

## 当前真实运行链路

```text
海康相机 / 本地实验视频
        │ RGB 帧
        ▼
FrameAnalysisNet（三头 CNN）
  ├─ 光心 (cx, cy)
  ├─ ROI 半径 r0 × 3
  └─ 画面质量 q
        │
        ▼
PhaseFringeCounter
  红通道 → s=r² 极坐标展开 → FFT 窄带解析
  → 时间解缠绕 → K 通道 Theil-Sen 外推 → 带符号 N
        │ 64 帧热身后开始输出
        ▼
NRateGate（抑制异常跳变）
        │ gated_N_abs
        ├─ 手动测量：N、ΔL 实时显示
        └─ ScanRunner：按段计算 α、统计均值/标准差并写 CSV
```

关键事实：

- `app/main.py` 是唯一服务入口，启动硬件、模型、取帧线程、温度同步线程和反应层。
- FrameAnalysisNet 是首选感知模型。加载失败时才依次使用独立的 QualityNet、CenterR0Net、CenterNet/CV 回退链。
- `center_mode` 只有 `cnn` 和 `cv` 两种；前端切换直接决定光心与 ROI 的来源。
- PhaseFringeCounter 不是旧式时序峰谷计数。它从空间条纹解调相位，能够保留方向并抵消升温回摆造成的反向吞吐。
- 计数器每次 `reset()` 后需要 64 帧热身定标；热身期 N 保持为 0。
- 对外显示和扫描结算使用 NRateGate 后的 `gated_N_abs`，不是计数器未经约束的瞬时值。
- temp 扫描按配置温度段计算 ΔT；fringe/free 模式在温控在线时按段内实测 PV 计算 ΔT。

## 三种扫描模式

| 模式 | 行为 | ΔT 口径 |
|---|---|---|
| `temp` | 按温度区间逐段升温，段末结算 | 配置的 T2 − T1 |
| `fringe` | 连续升温，每累计指定条纹数结算一段 | 实测 PV 起止差 |
| `free` | 不主动控制温度，停止时结算一次 | 温控在线时用实测 PV；离线视频用配置温差 |

启动测量或扫描前，必须已经开始相机取流，或已经加载本地视频模拟源。temp/fringe 扫描结束后会关闭加热；free 模式不会主动操作加热。

## Agent 的实际职责

当前 Agent 由三部分组成，不包含旧的自主战术/战略循环：

- 九阶段实验指导：安全说明、样品安装、光路粗调、光路细调、相机成像、初始温度、自动扫描、数据处理、报告与关机。
- 对话与工具调用：读取状态、控制相机和温控、启停测量/扫描、分析当前画面、查询 RAG、生成解释。
- 反应层：`SafetyGuard` 每 2 秒检查温度和条纹状态。超过 60 ℃ 时直接降温并停止扫描；低质量与条纹丢失事件会进入聊天和测量 WebSocket 提醒。

桌面端和移动端共享同一段聊天历史。对话支持当前相机帧、图片、文本文件以及可选的 PDF/DOCX 文本提取。

本地 AI 服务使用 Ollama：

- `qwen3-vl:4b`：对话、视觉分析和工具调用。
- `bge-m3`：RAG 文本嵌入与检索。

## 项目结构

```text
app/
  main.py                     FastAPI 入口与后台运行链
  config.py                   硬件、模型、扫描和物理常量
  state.py                    进程内共享状态
  camera/
    hik_camera.py             海康工业相机封装
    video_source.py           本地视频模拟帧源
    MvImport/                 海康 MVS Python 接口
  controller/
    modbus_temp.py            LU-926U 温控器通信
  measurement/
    frame_analysis_net.py     光心/r0/质量三头 CNN
    phase_fringe_counter.py   空间相位条纹计数
    n_rate_gate.py            N 跳变约束
    alpha_calc.py             ΔL、α 与统计计算
    center_*.py               感知模型回退实现
    quality_net.py            质量模型回退实现
  scan/
    scan_runner.py            temp/fringe/free 扫描状态机
  agent/
    agent_core.py             指导、聊天、工具、叙事和报告协调
    guide.py                  九阶段指导状态机
    safety.py                 规则安全检查
    tools.py                  Agent 工具定义与执行
    report_pdf.py             PDF 实验报告
  rag/
    store.py                  本地向量索引加载与检索
  routers/                    REST、SSE 与 WebSocket 接口
static/                       桌面端与移动端页面
checkpoints/                  当前运行链使用的模型权重
docs/                         RAG 知识源、索引和外部参考资料
measurements/                 实测 CSV、报告和报告素材
competition/report/           比赛报告、图表与构建脚本
```

顶层保留的 `autotune_pid.py` 用于温控器 PID 自整定。

## 环境与依赖

建议使用 Windows、Python 3.10+。真实硬件运行还需要：

- 海康 MVS SDK 与 MV-CS050-60GC 相机；项目内已包含 `MvImport` Python 接口。
- ANTHONE LU-926U 温控器，通过 RS485/USB 连接；默认 `COM7`、9600 baud、从站 1。
- Ollama；默认服务地址为 `http://localhost:11434`。
- CUDA GPU 为可选项；没有 CUDA 时 CNN 使用 CPU。

安装 Python 依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

准备 Ollama 模型：

```powershell
ollama pull qwen3-vl:4b
ollama pull bge-m3
```

相机、温控或 Ollama 不在线时，服务仍会尽量启动，并在控制台标出降级状态。硬件端口、模型名和实验参数统一在 `app/config.py` 修改。

## 启动

在项目目录执行：

```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080
```

访问：

- 桌面端：`http://localhost:8080/`
- 移动端：`http://localhost:8080/mobile/`
- OpenAPI：`http://localhost:8080/docs`

也可以直接运行：

```powershell
python -m app.main
```

## 无硬件视频模拟

页面或 `/api/camera/video/load` 可加载本机视频。视频源一旦加载，会优先于工业相机进入同一取帧和测量链，并自动开始计数；暂停、回卷和卸载由 `/api/camera/video/ctrl` 控制。

## 主要接口

| 前缀 | 用途 |
|---|---|
| `/api/camera` | 相机取流、参数、截图、中心模式、质量门、视频源 |
| `/api/temp` | PV/SV/MV、加热、参数和重连 |
| `/api/measure` | 手动测量启停与摘要 |
| `/api/scan` | 扫描配置、启停、状态和结果 |
| `/api/agent` | 指导、聊天、上传、报告、叙事和模型设置 |
| `/api/mobile` | 移动端共享聊天接口 |
| `/api/vlm` | VLM 状态与单次视觉问答 |
| `/ws/video` | JPEG 实时画面 |
| `/ws/measure` | 实时 N、质量、温度和告警 |
| `/ws/scan` | 扫描进度与段结果 |
| `/api/agent/ws/narration` | Agent 实验叙事 |

精确请求体与响应结构以运行后的 `/docs` 为准。

## 数据与报告

- 扫描记录写入 `measurements/scan_YYYYMMDD_HHMMSS.csv`。
- PDF 报告写入 `measurements/reports/`。
- 报告使用的粗调、细调画面位于 `measurements/report_assets/`。
- `measurements/` 中已有文件是真实测量与调试记录，清理代码时不应批量删除。
- `competition/report/` 是独立的比赛材料，不参与 Web 服务运行。

报告中的 α 均值和标准差只统计 `valid=True` 的扫描段。PT100 读数与样品真实平均温度之间可能存在热滞后，目前未单独建立定量补偿模型。

## RAG 知识库

源文档位于 `docs/*.md`，预构建索引位于 `docs/rag_index/`。修改知识源后，在 Ollama 和 `bge-m3` 可用时重新构建：

```powershell
python docs/build_index.py
```

外部论文、实验讲义和提取文本位于 `docs/external/`，它们是溯源资料，不进入应用源码链。

## 开发边界

这轮整理保留了实际运行链、回退模型、真实测量数据、报告素材、RAG 资料和 PID 自整定工具；已移除未接入运行链的探索模型、旧自主 Agent 循环、废弃接口、一次性验证/标注/旧分析脚本、缓存、临时目录、日志和未使用权重。

后续功能修改应从 `app/main.py` 的启动与取帧链、`app/measurement/phase_fringe_counter.py` 的计数输出，以及 `app/scan/scan_runner.py` 的结算口径开始核对，避免把历史说明或离线脚本误认为线上行为。
