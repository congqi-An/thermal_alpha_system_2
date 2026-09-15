"""Ollama 模型生命周期管理 — 启动预热 / 运行保活 / 退出卸载

配合系统环境变量 OLLAMA_MAX_LOADED_MODELS=2 (双模型常驻):
- 服务启动: 后台预热 bge-m3 + qwen3-vl, 首次对话零加载等待
- 运行期间: 每 4 分钟发最小请求刷新 keep_alive, 防空闲自动卸载
- 服务退出: keep_alive=0 立即卸载两个模型, 释放显存
"""
import asyncio

import httpx

from ..config import LLM_SERVER_URL, LLM_MODEL_NAME

EMBED_MODEL = "bge-m3"
KEEP_ALIVE = "30m"       # 单次请求声明的保活时长 (保活循环 4 分钟刷一次)
KEEPALIVE_INTERVAL = 240  # 保活循环间隔 (秒)


async def _ping(client: httpx.AsyncClient, quiet: bool = True):
    """对两个模型各发一个最小请求, 加载/续命"""
    await client.post(
        f"{LLM_SERVER_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": ["warmup"],
              "keep_alive": KEEP_ALIVE})
    await client.post(
        f"{LLM_SERVER_URL}/api/chat",
        json={"model": LLM_MODEL_NAME,
              "messages": [{"role": "user", "content": "hi"}],
              "think": False, "options": {"num_predict": 1},
              "keep_alive": KEEP_ALIVE})


async def warmup():
    """启动预热 (后台任务, 不阻塞服务启动)"""
    try:
        async with httpx.AsyncClient(timeout=180) as client:
            await _ping(client)
        print("  [OK] Ollama 双模型预热完成 (bge-m3 + VLM 常驻显存)")
    except Exception as e:
        print(f"  [WARN] Ollama 预热失败 (首次对话时再加载): {e}")


async def keepalive_loop():
    """保活循环: 定期刷新 keep_alive, 防止空闲超时卸载"""
    while True:
        await asyncio.sleep(KEEPALIVE_INTERVAL)
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                await _ping(client)
        except Exception:
            pass  # Ollama 暂不可用时静默跳过, 下轮重试


async def unload_models():
    """服务退出时卸载模型 (keep_alive=0 立即释放显存)
    注: 嵌入模型用 /api/chat 卸载 (/api/embed 的 keep_alive=0 不生效, 实测验证)"""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            await client.post(f"{LLM_SERVER_URL}/api/chat",
                              json={"model": LLM_MODEL_NAME, "keep_alive": 0})
            await client.post(f"{LLM_SERVER_URL}/api/chat",
                              json={"model": EMBED_MODEL, "keep_alive": 0})
        print("  [OK] Ollama 模型已卸载, 显存已释放")
    except Exception:
        pass  # Ollama 已离线则无需卸载
