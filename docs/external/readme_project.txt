# 来源: 项目 README
# 提取字符数: 15438, 中文字符: 3078

# 基于机器视觉和轻量多模态的热膨胀智能实验系统

> 迈克尔逊干涉法 · He-Ne 激光 · 黄铜 H62 热膨胀系数 α 全自动测量
> 机器视觉 CNN 帧分析 + 轻量多模态 VLM Agent + RAG 知识增强

---

## 一、项目概述

本系统将传统迈克尔逊干涉法测热膨胀系数实验进行智能化改造，融合**机器视觉**（轻量 CNN 实时帧分析）和**轻量多模态大模型**（MiniCPM-V 4.6, 1.2B 参数）两大 AI 技术，实现从条纹采集、质量评估、自动计数到实验报告生成的全流程自动化。

### 核心测量原理

黄铜样品（L₀=150mm）受热膨胀 → 顶推动镜 M1 → 等效空气膜厚度 h 变化 → 等倾同心圆条纹吞吐 → 计数条纹变化 N → 计算线膨胀系数：

$$\alpha = \frac{N \cdot \lambda}{2 L_0 \cdot \Delta T}$$

其中 λ=632.8nm（He-Ne 激光），每吞吐 1 个条纹对应 M1 位移 λ/2=316.4nm。

### 核心成果

| 成果 | 指标 | 说明 |
|------|------|------|
| 帧分析三合一 CNN | 43fps, 500K params | 光心+ROI+质量单次 forward, 实况微调后光心中位 65px |
| CV 峰谷对称精化光心 | 中位 22px (33帧人工真值) | 去光斑包络 + 镜像对称寻优, 与 CNN 双轨仲裁 |
| 判定模式切换 | auto / cnn / cv 三档 | 顶栏按钮一键切换, 实时生效 |
| 条纹自动计数 | N=22 (Video_28.7-30.7) | 过零法 + 信号质量门控 |
| 三模式扫描 | 温度步进 / N步进 / 即时计数 | 进段 3s 确认, N-dT 对偶测量 |
| 视频模拟源 | 无硬件回放实验视频 | 走与实拍完全相同的分析链路 |
| 三层决策 Agent | MiniCPM-V 4.6 (115 t/s) | 反应层安全联锁 / 战术层实时决策 / 战略层规划 |
| RAG 知识增强 | bge-m3 嵌入, 70 片 | 干涉理论 + CV/CNN + 设备手册 |
| 诚实误差预算 | 三层合成 CI | MC 计数噪声 + 方法分歧 + ΔT 系统差诊断 |
| PINN 数据采集线 | 激励-弛豫协议 | 平衡端点锚 α, 反演 (α,τ,k) 可辨识数据集 |

参考值：黄铜 H62 α = 20.8×10⁻⁶/K。

---

## 二、实验原理

### 2.1 迈克尔逊干涉仪光路

分振幅双光束干涉装置：
- **分束器 G1**：后表面镀半透半反膜，与光轴成 45°
- **补偿板 G2**：补偿两臂玻璃光程
- **动镜 M1**：由黄铜样品热膨胀顶推（本系统核心改造点）
- **定镜 M2**：固定反射镜

M2 经 G1 成虚像 M2'，干涉等效为 M1 与 M2' 间的空气薄膜（间距 h）。调节 M1 ∥ M2' 时产生等倾干涉同心圆环。

### 2.2 等倾干涉光强分布

光程差：$\Delta = 2h\cos\theta$（θ 为倾角）

光强分布：$I(\theta) = 4I_0\cos^2\left(\frac{2\pi h\cos\theta}{\lambda}\right)$

亮纹条件：$2h\cos\theta = k\lambda$

### 2.3 条纹吞吐测长

h 每变化 λ/2，中心吞吐 1 个条纹：$\Delta h = N \cdot \lambda/2$

任意固定半径 r₀ 处光强同步完成 N 个明暗周期（本系统采用此法，抗中心漂移）。

### 2.4 热膨胀系数

样品伸长直接推动 M1：$\Delta h = \Delta L = \alpha L_0 \Delta T$

联立得：$\alpha = \frac{N\lambda}{2L_0\Delta T}$

数值验证：α=20.8×10⁻⁶/K, ΔT=2°C → N≈19.7 条（与实测 N=17-22 一致）。

---

## 三、系统架构

### 3.1 硬件平台

| 设备 | 型号 | 接口 | 用途 |
|------|------|------|------|
| 激光器 | He-Ne 632.8nm | — | 相干光源 |
| 干涉仪 | 迈克尔逊 | — | 动镜被样品顶推 |
| 工业相机 | 海康 MV-CS050-60GC | GigE (2448×2048) | 采集干涉条纹 |
| 温控器 | ANTHONE LU-926U | RS485→USB (COM7@9600) | PID 加热 + PT100 测温 |
| 样品 | 黄铜 H62 | L₀=150mm | 热膨胀对象 |
| GPU | RTX 4060 Laptop 8GB | CUDA 13.0 | CNN 推理 + Ollama VLM |

### 3.2 软件架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Web 服务 (:8080)                       │
├──────────┬──────────┬──────────┬──────────┬─────────────────────┤
│camera_api│temp_api  │measure   │scan_api  │agent_api + vlm_api  │
│config_api│          │_api      │          │ws (video/measure)   │
├──────────┴──────────┴──────────┴──────────┴─────────────────────┤
│                        app/state.py (全局共享)                    │
├───────────────┬────────────────┬────────────────────────────────┤
│  measurement/ │   agent/       │  ai/ + rag/                    │
│  帧分析+计数  │   三层决策     │  VLM + RAG 知识增强            │
├───────────────┼────────────────┼────────────────────────────────┤
│  camera/      │  controller/   │  scan/                         │
│  海康 GigE    │  Modbus RTU    │  温度扫描                      │
└───────────────┴────────────────┴────────────────────────────────┘
         ↕                              ↕
    海康相机硬件                   LU-926U 温控器
```

### 3.3 目录结构

```
thermal_alpha_system/
├── app/
│   ├── main.py                    # FastAPI 启动入口
│   ├── config.py                  # 全局参数配置
│   ├── state.py                   # 全局共享状态
│   ├── camera/
│   │   ├── hik_camera.py          # 海康 GigE 相机封装
│   │   ├── video_source.py        # ★ 视频模拟源 (回放实验视频当帧源)
│   │   └── MvImport/              # MVS SDK Python 绑定
│   ├── controller/
│   │   └── modbus_temp.py         # LU-926U Modbus RTU 驱动
│   ├── measurement/
│   │   ├── frame_analysis_net.py  # ★ FrameAnalysisNet 三头CNN
│   │   ├── center_r0_net.py       # CenterR0Net (回退)
│   │   ├── center_cnn.py          # CenterNet 光心 (回退)
│   │   ├── center_tracker.py      # ★ CV 峰谷一体投票 + 镜像对称精化
│   │   ├── quality_net.py         # QualityNet (回退)
│   │   ├── fringe_counter.py      # ★ 时序过零计数 + 质量门控
│   │   ├── fringe_count_net.py    # FringeCountNet (Agent 工具备用)
│   │   ├── thermo_pinn.py         # 双节点热 ODE 反演 (α,τ,k)
│   │   ├── alpha_calc.py          # α 计算 + MC 置信区间
│   │   └── error_budget.py        # 三层误差预算
│   ├── agent/
│   │   ├── agent_core.py          # ★ 三层决策引擎
│   │   ├── event_bus.py           # 优先级事件总线
│   │   ├── safety.py              # 反应层安全联锁
│   │   ├── vision_loop.py         # 视觉监控循环
│   │   ├── llm_client.py          # Ollama API 客户端
│   │   ├── tools.py               # 16 个硬件工具
│   │   └── prompts.py             # 分层提示词
│   ├── rag/
│   │   └── store.py               # ★ RAG 向量检索 (bge-m3 + numpy)
│   ├── ai/
│   │   └── vlm.py                 # VLM 问答 (Ollama)
│   ├── scan/
│   │   └── scan_runner.py         # ★ 三模式扫描 (温度步进/N步进/即时计数)
│   └── routers/                   # REST + WebSocket 路由
├── static/                        # 前端 (科研 HUD 风格)
├── checkpoints/                   # CNN 模型权重
├── captures/                      # 实况训练帧 + labels.json (CNN 微调)
├── measurements/                  # 扫描 CSV / 采集数据集
├── docs/                          # RAG 知识库 + 理论文档
│   ├── kb_interferometry.md       # 干涉理论整合文档
│   ├── kb_cv_cnn.md               # CV/CNN 理论整合文档
│   ├── rag_index/                 # 向量索引 (70片×1024维)
│   └── external/                  # 外部参考资料
├── ramp_record.py / _gui.py       # 满功率分温度段采集 (CLI / PyQt6)
├── pinn_collect.py / _gui.py      # PINN 激励-弛豫协议采集 (CLI / PyQt6)
├── train_center_r0.py             # 光心/r0 CNN 训练 (视频+实况帧混合)
├── train_thermo_pinn.py           # 热反演 PINN 训练
├── label_captures.py              # 实况帧手动标注工具
├── diag_cnn.py / eval_live.py     # CNN 诊断 / 人工真值批量评估
├── video_detector.py / simulate_video.py  # 离线视频验证
└── requirements.txt
```

---

## 四、机器视觉：CNN 帧分析

### 4.1 FrameAnalysisNet 三头网络

单次前向推理同时输出三个关键量：

```
输入: (B, 1, 256, 256) 灰度帧
│
├─ 主 Encoder: Conv(1→32→64→128→256), stride=2×4
│   ├─ head_center: GAP → FC(256,64) → FC(64,2) → Sigmoid → (cx, cy)
│   └─ head_r0:     GAP → FC(256,64) → FC(64,3) → Sigmoid → (r0_1, r0_2, r0_3)
│
└─ QualityNet: MobileNetV3 InvertedResidual ×4 → GAP → FC → Sigmoid → quality
```

| 指标 | 数值 |
|------|------|
| 参数量 | ~500K |
| 推理速度 | <2ms/帧 (GPU), 43fps |
| 输入 | 256×256×1 灰度 |
| 输出 | 光心(cx,cy) + 3个r0 + 质量分 |

### 4.2 光心/ROI 判定: CNN-CV 双轨仲裁 + 模式切换

实况域偏移问题 (训练集光路与实况不同) 促使架构从"CNN 独断"演化为"双轨仲裁":

```
判定模式 (顶栏按钮切换, POST /api/camera/center_mode):
  auto (默认) ─ 每 30 帧 CNN-CV 交叉仲裁:
      峰谷对称精化成功 (sym≥0.55) → 用精化 CV (中位 22px, 全场最准)
      精化被拒 (无强对称结构)      → 用 CNN (中位 65px, 全天候稳定)
      大分歧                       → 历史轨迹仲裁 (谁近信谁)
      r0 头 raw 退化               → 径向 std 法替代 (r_max=560)
  cnn ─ 纯 CNN: center/r0 头直出, 退化仅警告不切换
  cv  ─ 纯 CV: 峰谷对称法光心 + 径向 std 法 r0, CNN 仅留质量头
```

峰谷对称精化关键: 先减去大核高斯背景去除激光光斑包络 (否则寻优锁到光斑中心),
再在纯条纹振荡结构上镜像对称寻优。模型权重回退链:

```
FrameAnalysisNet (三头, 优先) → CenterR0Net+QualityNet → CenterNet+CV投票
```

实况微调: 前端"采集训练帧"按钮存 captures/ → label_captures.py 人工标注
(椭圆条纹长短轴两点平均) → train_center_r0.py 视频+实况帧混合重训
(带标签平移/缩放增强防回归塌缩) → 自动 merge 生成新 frame_analysis_net.pt。

### 4.3 条纹计数算法 (FringeCounter)

```
海康相机 grab → 红通道 (He-Ne 632.8nm)
    ↓
FrameAnalysisNet → 光心(cx,cy) + 质量q
    ↓ (q < 0.4 剔除)
四方向径向线平均 → 一维剖面 line[0:600]  (r0 可用范围 [100, 560]px)
    ↓
百分位包络归一化 (percentile=85, size=41)
    ↓
测量开始: CNN 头帧锁定 r0, 整段固定
    ↓
I(r0, t) 时序采样 (每5帧取1帧)
    ↓
高斯去趋势 (σ=25) → 分离低频漂移
    ↓
find_peaks (distance=12, prominence=0.8×std)
同时计峰和计谷, N = max(峰数, 谷数)
    ↓
信号质量门控: std < 0.02 的通道剔除
    ↓
多通道有效均值 → 最终 N
```

### 4.4 误差预算 (三层合成)

| 层级 | 内容 | 方法 |
|------|------|------|
| 层1 计数噪声 | N 的随机不确定度 | MC 扰动 500×3r0 → α 95%CI |
| 层2 方法分歧 | 数峰 vs 数谷的差异 | 方法学不确定度 |
| 层3 ΔT 系统差 | 温控 PV ≠ 样品真实温度 | N+α_ref 反演真实 ΔT |

---

## 五、轻量多模态：VLM Agent

### 5.1 模型选择

| 项目 | 选择 | 理由 |
|------|------|------|
| 模型 | MiniCPM-V 4.6 (1.2B) | 8GB 显存可 BF16 直跑, 视觉+文本 |
| 推理引擎 | Ollama (llama.cpp) | Windows 原生, 115 t/s 文本, 44 t/s 图片 |
| 嵌入模型 | bge-m3 (1.2GB) | 中文向量效果好, RAG 用 |
| API | OpenAI 兼容 /v1/chat/completions | 标准 function calling |

### 5.2 三层决策架构

```
┌─────────────────────────────────────────────┐
│ 战略层 (Strategic)                           │
│   实验规划 / 报告生成 / 复杂异常处理         │
│   触发: 扫描完成 / 异常升级 / 用户指令       │
├─────────────────────────────────────────────┤
│ 战术层 (Tactical)                            │
│   每 3-5s LLM 快速决策, 带视觉帧            │
│   输入: 态势快照 + 图片 + 事件队列           │
│   输出: function calling 调 16 个工具        │
├─────────────────────────────────────────────┤
│ 反应层 (Reactive) — 无 LLM, 纯规则          │
│   T > 80°C → 硬停加热 (<1ms)                │
│   质量突降 / 光心漂移 → 事件推送             │
└─────────────────────────────────────────────┘
```

### 5.3 Agent 工具集 (16个)

| 类别 | 工具 | 功能 |
|------|------|------|
| 相机 | get_camera_frame | 获取当前帧 base64 |
| | set_camera_params | 调曝光/增益/Gamma/帧率 |
| | camera_start_stop | 启停取流 |
| 温控 | get_temperature | 读 PV/SV/MV |
| | set_temperature | 设定目标温度 (安全限 80°C) |
| 测量 | start_measurement | 开始条纹计数 |
| | stop_measurement | 停止并返回 N |
| | get_measurement | 实时读取计数 |
| 扫描 | configure_scan | 配置温度扫描参数 |
| | start_scan / stop_scan | 启停扫描 |
| | get_scan_status / get_scan_result | 查状态/取结果 |
| 视觉 | analyze_fringe_image | VLM 深度分析条纹 |
| 系统 | get_system_status | 全系统状态 JSON |
| | wait_seconds | 等待 (扫描期间) |

### 5.4 RAG 知识增强

| 组件 | 实现 |
|------|------|
| 嵌入模型 | Ollama bge-m3 (1024维) |
| 向量检索 | numpy 余弦相似度 (<1ms) |
| 知识源 | 干涉理论 + CV/CNN + LU-926U 手册 + README |
| 索引规模 | 70 片 × 1024 维 |
| 触发策略 | 仅知识问答触发, 操作指令跳过 (避免无谓开销) |
| 上下文限制 | 最多 1200 字符注入 |

### 5.5 对话能力

- 多轮记忆（保留最近 20 轮）
- 工具调用（检查状态、设参数、启停设备）
- RAG 增强（物理公式、设备操作、CNN 原理）
- 视觉分析（附带当前相机画面）
- 历史图片自动替换为占位符（防上下文爆炸）

---

## 六、数据流 (实时链路)

```
帧源 (优先级: 视频模拟源 > 海康相机实拍)
  ├─ 海康相机 grab (20fps, 2448×2048)
  └─ VideoFileSource (按视频原生 fps 节奏回放, 无硬件模拟)
    │
    ▼ 红通道提取 (He-Ne 632.8nm)
    │
    ▼ 下采样 256×256
    │
    ▼ FrameAnalysisNet (GPU, <2ms)
    ├─ 光心 (cx,cy) → 5帧滑动平均 → 实时追踪
    ├─ ROI (r0×3) → 显示用, 每30帧更新
    └─ 质量 q → q<0.4 剔除
    │
    ▼ 测量模式: 头帧锁定 r0
    │
    ▼ FringeCounter (每5帧采样)
    ├─ 四方向径向线 → 包络归一化
    ├─ 高斯去趋势 → find_peaks
    └─ 质量门控 → N
    │
    ▼ α = Nλ/(2L₀ΔT)
    │
LU-926U Modbus → PV/SV → ΔT (段起止实测)
    │
ScanRunner 三模式:
  temp   温度步进: 升至段起点 T1 保持 3s 确认进段 → 开计数 → SV=T2 等稳 → 段末 α
  fringe N步进:   连续升温, 每累计 n_step 条结算一段 (N 固定, dT 实测 — 对偶测量)
  free   即时计数: 不控温, 启动即计, 停止时结算单段 (dT<0.05°C 时 α 标无效)
    │
Agent (Ollama): 视觉监控 + 战术决策 + 战略规划
```

---

## 七、安装与启动

### 7.1 环境要求

- Python 3.10+ (conda 环境 `ocv`)
- PyTorch 2.10+ with CUDA
- Ollama (Windows 原生安装)
- 海康 MVS SDK (相机驱动)

### 7.2 安装

```bash
# 1. conda 环境
conda activate ocv

# 2. Python 依赖
cd thermal_alpha_system
pip install -r requirements.txt

# 3. Ollama + 模型
ollama pull openbmb/minicpm-v4.6   # VLM (1.6GB)
ollama pull bge-m3                  # 嵌入模型 (1.2GB)

# 4. 海康 MVS
#    安装 MVS, 防火墙放行 UDP 3956 + 49152-65535

# 5. 温控串口
#    USB转RS485, 装 CH340/Prolific 驱动, 确认 COM 号
```

### 7.3 启动

```bash
# 确认 Ollama 运行中
ollama list  # 应看到 minicpm-v4.6 和 bge-m3

# 启动服务
conda activate ocv
cd e:\pycharm\PythonProject2\thermal_alpha_system
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

浏览器打开 `http://localhost:8080`。

### 7.4 无硬件验证 (视频模拟)

不插相机/温控也能跑完整分析链路:

- **Web 内置 (推荐)**: 相机面板→"视频模拟"区→输入视频路径→加载并播放。
  画面/光心/Q/信号震荡图实时显示, 配合测量面板或扫描"即时计数"模式数条纹
- 命令行: `python video_detector.py` (GUI) / `python simulate_video.py` (链路验证)

### 7.5 数据采集程序 (独立于 web, 相机独占需先停 uvicorn)

| 程序 | 用途 |
|------|------|
| `python ramp_record_gui.py` | 满功率升温分温度段采集: 进段 3s 确认, 段视频+N(t)/T(t) 统一时间标签落盘 |
| `python pinn_collect_gui.py` | PINN 反演数据: 基线→加热激励→自由弛豫至平衡 (α 锚点), 支持多功率批量 |

采集输出均在 `measurements/`, 含时间重采样 + 分相位多尺度事件提取 + 采样诊断。

---

## 八、配置参数 (app/config.py)

### 物理参数

| 参数 | 值 | 说明 |
|------|-----|------|
| LASER_WAVELENGTH_NM | 632.8 | He-Ne 波长 nm |
| SAMPLE_LENGTH_MM | 150.0 | 黄铜样品长度 |
| REFERENCE_ALPHA | 20.8 | H62 参考值 ×10⁻⁶/K |

### 相机

| 参数 | 值 | 说明 |
|------|-----|------|
| CAMERA_EXPOSURE_US | 20000 | 曝光 μs |
| CAMERA_GAIN_DB | 0.0 | 增益 |
| CAMERA_FPS | 20.0 | 帧率 |
| CAMERA_GAMMA | 0.7 | Gamma 校正 |

### 温控

| 参数 | 值 | 说明 |
|------|-----|------|
| MODBUS_PORT | COM7 | 串口 |
| MODBUS_BAUD | 9600 | 波特率 |
| SCAN_T_START / T_END | 30 / 50 °C | 扫描范围 |
| SCAN_STEP | 2.0 °C | 步进 |
| SCAN_STABILIZE_S | 60 s | 段内稳定时间 |

### 计数

| 参数 | 值 | 说明 |
|------|-----|------|
| COUNT_R0S | (150,200,250) | 固定径向位置 (回退) |
| COUNT_PROFILE_LEN | 600 | 径向剖面长度 (r0 上限 560px) |
| COUNT_DISTANCE | 12 | 最小峰间距 |
| COUNT_PROMINENCE_RATIO | 0.8 | prominence = 0.8×std |
| COUNT_DETREND_SIGMA | 25 | 去趋势高斯核 |
| QUALITY_THRESHOLD | 0.4 | 质量门控阈值 |

### AI

| 参数 | 值 | 说明 |
|------|-----|------|
| LLM_SERVER_URL | http://localhost:11434 | Ollama 地址 |
| LLM_MODEL_NAME | openbmb/minicpm-v4.6 | 模型标识 |
| AGENT_TEMP_LIMIT | 80.0 °C | 安全温度上限 |
| AGENT_TACTICAL_INTERVAL | 5.0 s | 战术决策间隔 |
| AGENT_VISION_INTERVAL | 3.0 s | 视觉循环间隔 |
| VLM_ENABLED | True | 启用 VLM |

---

## 九、API 参考

### REST 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/camera/info | 相机状态 |
| POST | /api/camera/start / stop | 启停取流 |
| POST | /api/camera/reconnect | 重连相机 |
| POST | /api/camera/exposure / gain / framerate | 调参 |
| POST | /api/camera/capture | 保存当前帧到 captures/ (CNN 微调采集) |
| GET/POST | /api/camera/center_mode | 光心/ROI 判定模式 (auto/cnn/cv) |
| POST | /api/camera/video/load | 加载视频模拟源 (接管帧源) |
| POST | /api/camera/video/ctrl | play/pause/rewind/unload |
| GET | /api/camera/video/status | 视频回放进度 |
| GET | /api/temp/status | 温控状态 {online,pv,sv,mv} |
| POST | /api/temp/set_sv | 设温度 |
| POST | /api/temp/heat | 加热 on/off |
| POST | /api/temp/reconnect | 重连温控 |
| POST | /api/measure/start / stop | 手动测量 |
| GET | /api/measure/summary | 实时 N/ΔL/质量Q/光心来源 |
| POST | /api/scan/config | 扫描配置 {mode: temp/fringe/free, n_step, ...} |
| POST | /api/scan/start / stop | 扫描启停 |
| GET | /api/scan/status / result | 扫描状态/结果 (含 mode) |
| POST | /api/agent/start / stop | Agent 控制 |
| GET | /api/agent/status | Agent 状态 |
| POST | /api/agent/chat | 对话 (带记忆+工具+RAG) |
| POST | /api/agent/intervene | 人工干预 |
| POST | /api/agent/config | 调推理参数 |
| POST | /api/agent/clear_history | 清空对话记忆 |
| GET | /api/agent/health | Ollama 连通检查 |
| POST | /api/vlm/ask | VLM 视觉问答 |

### WebSocket

| 路径 | 推送 |
|------|------|
| /ws/video | JPEG帧 + meta{N, quality, T, r0s} |
| /ws/measure | 实时测量数据流 |
| /ws/scan | 扫描进度 |

---

## 十、系统行为约定

| 行为 | 约定 |
|------|------|
| 硬件前置检查 | 无帧源 (相机未取流且未加载视频) → 测量/扫描拒绝启动 |
| 视频模拟源 | 加载即接管帧源 (卸载才回相机); 暂停时画面定格 |
| 安全联锁 | T > 80°C 反应层硬停 (不经 LLM, <1ms) |
| 扫描进段 | 升温至段起点 T1 并保持 3s 才开计数 (防首段从室温起累计 α 虚高) |
| 扫描ΔT | 温控在线取段起止实测 PV; 离线用标称值; dT<0.05°C 时 α 标无效 |
| 扫描结束 | temp/fringe 模式自动 heat_off; free 模式不碰温控 (不打断手动加热) |
| 计数 r0 锁定 | 每段开始时头帧重新锁定 (来源随判定模式) |
| 质量门控 | 仅质量模型可用时生效, 否则自动豁免 |
| 温控重连 | 串口打开≠在线, 首次成功读 PV 后确认 |
| Agent 硬件感知 | 态势快照明确标注 OFFLINE, 提示词禁止离线操作 |
| 温度标签 | 仅为参考, 不用于理论计算 (N 是直接观测量) |

---

## 十一、技术栈

| 层级 | 技术 |
|------|------|
| 语言 | Python 3.10 |
| 深度学习 | PyTorch 2.10 + CUDA 13.0 |
| 图像处理 | OpenCV 4.13, SciPy |
| Web 框架 | FastAPI + uvicorn |
| 异步 HTTP | httpx |
| 串口通信 | pyserial (Modbus RTU) |
| 相机 SDK | 海康 MVS (GigE) |
| 采集 GUI | PyQt6 (ramp_record_gui / pinn_collect_gui) |
| VLM 推理 | Ollama (llama.cpp, MiniCPM-V 4.6) |
| 嵌入模型 | Ollama bge-m3 |
| RAG 检索 | numpy 余弦相似度 |
| 前端 | 原生 HTML/CSS/JS + KaTeX + marked.js |

---

## 十二、故障排查

| 现象 | 解决 |
|------|------|
| 相机未发现 | 检查网口/同网段/防火墙; 顶栏重连按钮 |
| 温控离线 | 设备管理器确认 COM 号; 顶栏重连 |
| 测量/扫描被拒 | 先启动取流, 或加载视频模拟源 |
| 光心不准 | 切换判定模式对比 (顶栏"判定"按钮); 采集实况帧微调 CNN |
| Agent 不可用 | 确认 Ollama 运行: `ollama list` |
| CNN 未加载 | 检查 checkpoints/frame_analysis_net.pt |
| N=0 | 检查 r0 锁定/信号强度/曝光 |
| α 偏差大 | 温度标签仅参考, 以 N 为直接观测量 |
| 公式不渲染 | 检查 CDN (jsdelivr) 是否可达 |
