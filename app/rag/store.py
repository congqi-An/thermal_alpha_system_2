"""轻量 RAG 存储与检索 — 基于 Ollama bge-m3 嵌入 + numpy 余弦相似度

零重型依赖 (不用 chromadb/faiss): 知识库仅数百切片, numpy 检索 <1ms。
索引持久化: docs/rag_index/{chunks.json, embeddings.npy}
"""
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import httpx

from ..config import LLM_SERVER_URL

EMBED_MODEL = "bge-m3"
INDEX_DIR = Path(__file__).resolve().parent.parent.parent / "docs" / "rag_index"
CHUNK_SIZE = 800       # 每片目标字符数 (加大: 减少总片数, 降低检索开销)
CHUNK_OVERLAP = 120    # 相邻片重叠


class RagStore:
    """向量知识库: 切片 -> 嵌入 -> numpy 余弦检索"""

    def __init__(self, base_url: str = None):
        self.base_url = (base_url or LLM_SERVER_URL).rstrip("/")
        self.chunks: list[dict] = []       # [{text, source, idx}]
        self.embeddings: Optional[np.ndarray] = None  # (N, dim) 已归一化

    # ─── 嵌入 ─────────────────────────────────────────────────

    def embed(self, texts: list[str]) -> np.ndarray:
        """调用 Ollama /api/embed 批量嵌入, 返回归一化向量"""
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(f"{self.base_url}/api/embed",
                               json={"model": EMBED_MODEL, "input": texts})
            resp.raise_for_status()
            embs = np.array(resp.json()["embeddings"], dtype=np.float32)
        # L2 归一化 (余弦相似度 = 点积)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / np.maximum(norms, 1e-8)

    # ─── 切片 ─────────────────────────────────────────────────

    @staticmethod
    def split_text(text: str, source: str) -> list[dict]:
        """标题锚定切片: 每个 ##/### 小节独立成片, 片头拼"文档标题 > 小节标题"
        路径作为语义锚点 (旧版按空行聚合, 标题被并进上一片, 检索时丢失主题)。
        超长小节再按句号硬切, 带重叠。
        """
        lines = text.splitlines()
        doc_title = next((l.lstrip("# ").strip() for l in lines
                          if l.startswith("# ")), source)
        sections: list[tuple[str, list[str]]] = []  # (标题路径, 正文行)
        h2 = h3 = ""
        cur_path, cur_body = doc_title, []
        for line in lines:
            if line.startswith("### "):
                h3 = line[4:].strip()
                if cur_body:
                    sections.append((cur_path, cur_body))
                cur_path = f"{doc_title} > {h2} > {h3}" if h2 else f"{doc_title} > {h3}"
                cur_body = []
            elif line.startswith("## "):
                h2, h3 = line[3:].strip(), ""
                if cur_body:
                    sections.append((cur_path, cur_body))
                cur_path = f"{doc_title} > {h2}"
                cur_body = []
            elif line.strip() in ("---", ""):
                continue
            else:
                cur_body.append(line.rstrip())
        if cur_body:
            sections.append((cur_path, cur_body))

        chunks = []
        for path, body in sections:
            seg = "\n".join(body)
            if len(seg) > CHUNK_SIZE:
                # 超长小节: 按句号切分, 带重叠, 每片仍带标题锚点
                parts, buf = [], ""
                for sent in re.split(r"(?<=[。；])", seg):
                    if len(buf) + len(sent) > CHUNK_SIZE and buf:
                        parts.append(buf)
                        buf = buf[-CHUNK_OVERLAP:] + sent
                    else:
                        buf += sent
                if buf.strip():
                    parts.append(buf)
                chunks.extend(f"【{path}】\n{p}" for p in parts)
            else:
                chunks.append(f"【{path}】\n{seg}")
        return [{"text": c, "source": source, "idx": i}
                for i, c in enumerate(chunks) if len(c.strip()) > 40]

    # ─── 构建索引 ─────────────────────────────────────────────

    def build(self, doc_paths: list[tuple], batch_size: int = 16):
        """doc_paths: [(path, source_name)], 构建并保存索引"""
        all_chunks = []
        for path, source in doc_paths:
            try:
                text = Path(path).read_text(encoding="utf-8")
            except Exception as e:
                print(f"  [RAG] 读取失败 {source}: {e}")
                continue
            cs = self.split_text(text, source)
            all_chunks.extend(cs)
            print(f"  [RAG] {source}: {len(cs)} 片")

        self.chunks = all_chunks
        # 分批嵌入
        embs = []
        for i in range(0, len(all_chunks), batch_size):
            batch = [c["text"] for c in all_chunks[i:i + batch_size]]
            embs.append(self.embed(batch))
            print(f"  [RAG] 嵌入 {min(i + batch_size, len(all_chunks))}/{len(all_chunks)}")
        self.embeddings = np.vstack(embs)

        # 持久化
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        with open(INDEX_DIR / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False)
        np.save(INDEX_DIR / "embeddings.npy", self.embeddings)
        print(f"  [RAG] 索引完成: {len(self.chunks)} 片, dim={self.embeddings.shape[1]}")

    # ─── 加载 ─────────────────────────────────────────────────

    def load(self) -> bool:
        """从磁盘加载索引"""
        try:
            with open(INDEX_DIR / "chunks.json", encoding="utf-8") as f:
                self.chunks = json.load(f)
            self.embeddings = np.load(INDEX_DIR / "embeddings.npy")
            return True
        except Exception:
            return False

    @property
    def ready(self) -> bool:
        return self.embeddings is not None and len(self.chunks) > 0

    # ─── 检索 ─────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 3, min_score: float = 0.4) -> list[dict]:
        """检索 top-k 相关切片, 过滤低分"""
        if not self.ready:
            return []
        q_emb = self.embed([query])[0]              # (dim,)
        scores = self.embeddings @ q_emb            # 余弦相似度
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in top_idx:
            if scores[i] < min_score:
                continue
            results.append({
                "text": self.chunks[i]["text"],
                "source": self.chunks[i]["source"],
                "score": round(float(scores[i]), 3),
            })
        return results

    def build_context(self, query: str, top_k: int = 3, max_chars: int = 2000) -> str:
        """检索并拼接为提示词上下文, 无结果返回空串"""
        results = self.search(query, top_k=top_k)
        if not results:
            return ""
        parts = []
        total = 0
        for r in results:
            seg = f"【{r['source']} (相关度{r['score']})】\n{r['text']}"
            if total + len(seg) > max_chars:
                break
            parts.append(seg)
            total += len(seg)
        return "\n\n".join(parts)


# ─── 对话路由: 操作/状态诉求 vs 知识问答 ────────────────────
# 原则: 动词门控而非名词门控。旧版用"条纹/温度/相机"等名词判断操作指令,
# 导致"条纹为什么吞吐"这类知识问答被误判为操作而永远触发不了 RAG。

_ACTION_WORDS = [          # 硬件操作 / 状态查询诉求 → 需要工具
    "启动", "停止", "开始", "终止", "执行", "设定", "设置", "配置",
    "重连", "取流", "曝光", "增益", "帧率", "gamma",
    "检查", "看一下", "查看", "状态", "多少度", "几度", "多少条",
    "帮我", "请帮我", "升温", "降温", "加热", "关闭", "打开",
    # 高频操作表述 (修复: "把SV调到45"/"设温度到45"/"温度调到50" 曾被漏检
    # → needs_tools=False → 模型拿不到工具, 命令被静默忽略)
    "调到", "调成", "改成", "改为", "设到", "设成", "升到", "降到",
    "提高到", "降低到", "目标温度", "sv", "设", "调", "改",
]

_ACTION_COMBOS = [         # 动词+对象组合: 避免"扫描有哪几种模式"被名词误判
    "开始扫描", "启动扫描", "停止扫描", "执行扫描", "配置扫描", "设置扫描",
    "开始测量", "启动测量", "停止测量", "执行测量",
    "开始实验", "开始计数", "调曝光", "调增益", "调温度",
    # 温度状态查询: "现在温度多少"/"温度是多少" 须触发 get_temperature 工具
    # (修复: 旧词表只有"多少度/几度", "温度多少"漏判为纯知识问答 → 模型拿不到工具无法查实时温度)
    "温度多少", "温度是多少", "现在温度", "当前温度", "目前温度",
]

_KNOWLEDGE_WORDS = [       # 知识/原理/指导诉求 → 触发 RAG
    "为什么", "为啥", "原理", "原因", "解释", "含义", "公式", "推导",
    "光路", "搭建", "调节", "扩束", "干涉", "条纹", "膨胀", "波长",
    "怎么调", "怎么搭", "如何调", "如何搭", "怎么办", "怎么处理",
    "怎么解决", "区别", "影响", "机制", "现象", "物理", "误差",
    "是什么", "什么是", "干什么", "哪些", "怎样", "如何", "怎么",
]


# 纯问候/寒暄: 去掉问候词后无实质诉求 → 不触发 RAG
_GREETINGS = (
    "你好", "您好", "你们好", "hello", "hi", "嗨", "在吗",
    "早上好", "下午好", "晚上好", "谢谢", "感谢", "好的", "嗯",
    "再见", "拜拜", "辛苦了",
)


def _pure_greeting(p: str) -> bool:
    """去掉问候词+标点后几乎无剩余 → 纯寒暄。保留长度≥2 的实质内容
    (如"你好，什么是等倾干涉"会剩"什么是等倾干涉", 不判为寒暄)。"""
    rest = p
    for g in _GREETINGS:
        rest = rest.replace(g, "")
    rest = rest.strip(" \t\n，。,.！!？?～~、")
    return len(rest) < 2


def route_query(prompt: str) -> tuple[bool, bool]:
    """返回 (needs_tools, needs_rag)。两者可同时为真:
    例: "温度多少? 为什么会过冲" 既要工具查状态也要知识解释。
    都不命中时默认走知识问答 (RAG 有 min_score 兜底, 开销毫秒级)。
    纯问候/寒暄 (无操作、无知识诉求) → 两者都不触发, 避免无谓嵌入开销。
    """
    p = prompt.lower()
    needs_tools = (any(w in p for w in _ACTION_WORDS)
                   or any(c in p for c in _ACTION_COMBOS))
    has_knowledge = any(w in p for w in _KNOWLEDGE_WORDS)
    if not needs_tools and not has_knowledge and _pure_greeting(p):
        return False, False
    needs_rag = has_knowledge or not needs_tools
    return needs_tools, needs_rag


# 全局单例 (在 main.py 启动时 load)
rag_store = RagStore()
