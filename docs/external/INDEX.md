# 外部资料索引 (RAG 知识源清单)

> 下载/提取时间: 2026-07-26 | 提取脚本: `docs/extract_pdfs.py` | 下载脚本: `docs/download_refs.py`

## 理论类（迈克尔逊干涉 / 等倾干涉）

| 文件 | 来源 | 内容 | 提取状态 |
|------|------|------|---------|
| ncu_michelson.pdf/.txt | 南昌大学物理实验中心 | 仪器结构、等倾干涉原理、G1/G2/M1/M2 结构说明 | OK 2727字符 |
| hrbeu_michelson.pdf/.txt | 哈尔滨工程大学物理实验中心 | 预习要点、调节方法、干涉条件、环纹特性 | OK 6629字符 |
| ustc_interference.pdf/.txt | 中国科大 崔宏滨《光学》讲义 3-03 | **Δ=2nh·cosθ 严格推导**、半波损失讨论、等厚/等倾对比 | OK 11421字符 |
| nenu_teaching_qa.pdf/.txt | 东北师大物理实验中心 (物理实验期刊) | 迈克耳孙干涉实验常见教学疑难辨析 | OK 10579字符 |
| hanspub_analysis.pdf/.txt | 汉斯出版社《现代物理》论文 | 杨氏/等倾/等厚/迈克尔逊干涉对比分析 | OK 10189字符 |
| htu_michelson.docx/.txt | 河南师范大学 | 迈克尔逊干涉仪的调整和使用（实验指导书体例） | OK 3944字符 |

## 设备类

| 文件 | 来源 | 内容 | 提取状态 |
|------|------|------|---------|
| lu926u_manual.txt | ANTHONE LU-926U 说明书 V1.8 (本机 PDF) | Modbus 寄存器表、PID 参数、操作说明、接线 | OK 22630字符 |

## 项目类

| 文件 | 来源 | 内容 | 提取状态 |
|------|------|------|---------|
| readme_project.txt | thermal_alpha_system/README.md | 项目架构、测量链路、CNN模型、计数算法、误差预算、核心成果 | OK 13125字符 |

## 合成文档

| 文件 | 说明 |
|------|------|
| ../michelson_theory.md | **RAG 主知识源**：基于上述资料合成的严格理论推导（等倾干涉光强分布、条纹吞吐、α 测量公式、误差传播、代码对应表），公式为 KaTeX 兼容 LaTeX 格式 |

## 原始 URL（供重新下载）

- 南昌大学: https://wlsyzx.ncu.edu.cn/docs/2023-07/b354796076af4d1f8fe5c78c3e552ab9.pdf
- 哈工程: https://pec.hrbeu.edu.cn/_upload/article/files/80/09/a673f3bf4df593c7784d4297035c/196eebfe-b546-4953-941b-8b7fc99aeb89.pdf
- 中科大: http://staff.ustc.edu.cn/~chunhua/3-03-1.pdf
- 东北师大: https://wlsy.nenu.edu.cn/2212fdh.pdf
- 汉斯出版社: https://pdf.hanspub.org/mp20220300000_37337708.pdf
- 河南师大: https://www.htu.edu.cn/_upload/article/files/e6/d6/c4ab36a74b7c8bf8a8a9c4f69975/230cb2d9-68e4-4b29-8192-3af7a8777af6.docx

## 数值自洽性验证

α = N·λ/2/(L₀·ΔT) 反算: N = 2×20.8e-6×0.15×2/632.8e-9 ≈ **19.7 条** / 2°C 温升
与系统实测 N≈17-22 量级一致，理论公式与代码实现自洽。
