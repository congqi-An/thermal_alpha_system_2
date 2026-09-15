"""多模态助手 (Qwen3-VL 4B) — 通过 Ollama API 提供视觉问答

不再本地加载模型(避免占显存/速度慢), 改为调用 Ollama 服务。
接口保持兼容: ask(image, prompt) -> (text, time_s)
"""
import time
import base64
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from ..config import LLM_SERVER_URL, LLM_MODEL_NAME


class VLMAssistant:
    """通过 Ollama API 调用 VLM (Qwen3-VL 4B) 的轻量客户端"""

    def __init__(self):
        self.base_url = LLM_SERVER_URL.rstrip("/")
        self.model_name = LLM_MODEL_NAME
        self._ready = False

    def load(self):
        """检查 Ollama 服务是否可用"""
        import requests
        try:
            r = requests.get(f"{self.base_url}/api/version", timeout=5)
            self._ready = r.status_code == 200
            if self._ready:
                print(f"  [OK] VLM via Ollama ({self.model_name})")
            return self._ready
        except Exception as e:
            print(f"  [WARN] VLM Ollama 连接失败: {e}")
            self._ready = False
            return False

    @property
    def ready(self):
        return self._ready

    def ask(self, image, prompt, max_new=400):
        """视觉问答, 返回 (text, time_s)

        image: numpy RGB uint8 或 None(纯文本)
        Qwen3-vl 走 Ollama 原生 /api/chat (think=False): /v1 兼容端点不支持
        think 参数, 小 token 预算下思考吞预算/答案被截断 (2026-08-04 实测
        "2+2等于" 被截断); 原生端点 think=False 直接回答, 快且不吞 token。
        """
        import requests

        if not self._ready:
            return "VLM 服务不可用 (Ollama 离线)", 0.0

        # 原生消息格式: content 纯文本 + images 数组 (base64)
        images = []
        if image is not None:
            b64 = self._frame_to_b64(image)
            if b64:
                images.append(b64)
        # qwen3-vl 裸模板不响应 think:false → /no_think 方案真正关闭思考
        # (末尾 user 加 /no_think + 预填空思考块 + raw:true, 实测 thinking=0)
        msg = {"role": "user", "content": prompt.rstrip() + " /no_think"}
        if images:
            msg["images"] = images

        payload = {
            "model": self.model_name,
            "messages": [msg, {"role": "assistant",
                               "content": "<think>\n\n</think>\n\n"}],
            "stream": False,
            "raw": True,             # /no_think 方案需绕过模板
            "think": False,          # 轻量问答: 关思考直接答, 不浪费 token
            # num_ctx 必须显式抬高: 默认 4096, 一张图即 ~4000 token,
            # 不带 num_ctx 时上下文被图吃光 → 答案为空 (2026-08-04 实测)
            "options": {"num_predict": max(int(max_new), 1024), "num_ctx": 16384},
        }

        t0 = time.time()
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload, timeout=60
            )
            resp.raise_for_status()
            data = resp.json()
            text = data.get("message", {}).get("content", "")
            dt = time.time() - t0
            return text, dt
        except Exception as e:
            return f"VLM 调用失败: {e}", time.time() - t0

    @staticmethod
    def _frame_to_b64(frame, quality=75, max_side=1280):
        """numpy RGB frame -> base64 JPEG (长边限 max_side, 控制图像 token 开销)"""
        if frame is None:
            return None
        if cv2 is not None:
            f = frame
            h, w = f.shape[:2]
            if max(h, w) > max_side:
                scale = max_side / max(h, w)
                f = cv2.resize(f, (int(w * scale), int(h * scale)))
            # RGB 转 BGR 供 imencode
            bgr = (cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
                   if len(f.shape) == 3 and f.shape[2] == 3 else f)
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ok:
                return base64.b64encode(buf).decode("ascii")
        # fallback: PIL
        try:
            from PIL import Image
            import io
            if isinstance(frame, np.ndarray):
                img = Image.fromarray(frame)
            else:
                img = frame
            if max(img.size) > max_side:
                img.thumbnail((max_side, max_side))
            bio = io.BytesIO()
            img.save(bio, format="JPEG", quality=quality)
            return base64.b64encode(bio.getvalue()).decode("ascii")
        except Exception:
            return None
