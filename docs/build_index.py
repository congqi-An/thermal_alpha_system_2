"""构建 RAG 向量索引 v3 — 面向本项目的知识源重组

知识源策略 (本项目内容优先):
- kb_optical_path.md   光路搭建与调节指导 (原始实验讲义 + 扩束镜前后现象)
- kb_interferometry.md 物理原理 (讲义式1-6公式链 + 误差深层推理)
- kb_project.md        本项目系统知识 (README 提炼)
- kb_cv_cnn.md         条纹 CV/CNN 分析理论
- 设备手册 (LU-926U) 不再入库: 温控操作细节已在 kb_project 覆盖,
  旧版手册占 50% 切片导致检索被说明书内容淹没

用法: python build_index.py  (需 Ollama 运行且已 pull bge-m3)
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from app.rag.store import RagStore

SOURCES = [
    (HERE / "kb_optical_path.md", "光路搭建与调节"),
    (HERE / "kb_interferometry.md", "干涉法物理原理"),
    (HERE / "kb_project.md", "本系统项目知识"),
    (HERE / "kb_cv_cnn.md", "条纹CV/CNN分析"),
]

if __name__ == "__main__":
    store = RagStore()
    store.build([(str(p), name) for p, name in SOURCES])

    # 检索自检
    print("\n===== 检索自检 =====")
    test_queries = [
        "光路怎么搭建和调节",
        "不加扩束镜能看到什么现象",
        "加了扩束镜之后会出现什么",
        "两个最亮的光点重合是什么意思",
        "线膨胀系数怎么计算",
        "为什么条纹会吞吐",
        "等倾干涉光强分布公式",
        "这个系统是干什么的",
        "扫描有哪几种模式",
        "为什么测出的α偏大",
        "条纹质量CNN是怎么工作的",
        "温控器离线了怎么办",
    ]
    for q in test_queries:
        results = store.search(q, top_k=2)
        print(f"\nQ: {q}")
        if not results:
            print("  (无命中)")
        for r in results:
            print(f"  [{r['score']:.3f}] {r['source']}: {r['text'][:50]}...")
