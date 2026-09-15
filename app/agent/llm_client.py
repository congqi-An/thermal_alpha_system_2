"""LLM HTTP 客户端 — 对接 Ollama 原生 /api/chat (Qwen3-VL 4B)

qwen3 系列走原生端点 (支持 think 参数, 图片转 images 数组);
非 qwen3 回退 /v1/chat/completions。支持: 纯文本/图片+文本/function calling。
"""
import base64
import json
from dataclasses import dataclass, field
from typing import Optional

import httpx
try:
    import cv2
except ImportError:
    cv2 = None


@dataclass
class ToolCall:
    """LLM 返回的工具调用"""
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    """LLM 响应"""
    content: Optional[str] = None
    thinking: Optional[str] = None   # 深度思考内容 (思考型模型, 前端可折叠展示)
    tool_calls: list = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class LLMClient:
    """异步 LLM 客户端, 对接 Ollama /v1/chat/completions"""

    def __init__(self, base_url: str = "http://localhost:11434",
                 model_name: str = "qwen3-vl:4b", timeout: float = 180.0,
                 think: bool = True):
        # timeout 2026-08-07: 60→180s — qwen3-vl 4B 思考模式 + 工具轮 + 长历史
        # 单次非流式调用实测可达 60-120s, 60s 超时导致工具轮/告警偶发失败
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        # 深度思考开关: True=开(qwen3 走原生 /api/chat think=True), False=关(think=False)
        # 前端可经 /api/agent/config 实时切换 (类似 DeepSeek 深度思考开关)
        self.think_default = think
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout, connect=10.0)
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ─── 核心调用 ───────────────────────────────────────────────

    async def chat(
        self,
        messages: list,
        tools: Optional[list] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        think: Optional[bool] = None,
    ) -> LLMResponse:
        """调用 /v1/chat/completions

        Args:
            messages: OpenAI 格式消息列表
            tools: function calling 工具定义 (可选)
            temperature: 采样温度
            max_tokens: 最大生成 token 数
            think: 深度思考开关 (None → 用 think_default; 前端 /api/agent/config 可切)
        """
        client = await self._get_client()

        # Qwen3 系列走 Ollama 原生 /api/chat: /v1 兼容端点不支持 think 参数,
        # 思考输出全部进 reasoning_content 导致正式 content 为空 ("无回复"根因)
        if "qwen3" in self.model_name:
            return await self._chat_native(client, messages, tools,
                                           temperature, max_tokens,
                                           think if think is not None
                                           else self.think_default)

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        for attempt in range(3):
            try:
                resp = await client.post("/v1/chat/completions", json=payload)
                resp.raise_for_status()
                data = resp.json()
                return self._parse_response(data)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                # TransportError 覆盖 ConnectError/ReadError/RemoteProtocolError 等
                if attempt < 2:
                    await self._sleep(2 ** attempt)
                    continue
                return LLMResponse(
                    content=f"[LLM连接失败] {e}",
                    finish_reason="error"
                )
            except httpx.HTTPStatusError as e:
                return LLMResponse(
                    content=f"[LLM HTTP错误] {e.response.status_code}: {e.response.text[:200]}",
                    finish_reason="error"
                )
            except Exception as e:
                # JSON 解析错/其他异常: 不向上抛 (否则 chat 路由直接 500)
                return LLMResponse(
                    content=f"[LLM响应异常] {type(e).__name__}: {e}",
                    finish_reason="error"
                )

    # ─── Ollama 原生端点 (Qwen3 专用, 支持 think 参数) ─────────────

    @staticmethod
    def _to_native_messages(messages: list) -> list:
        """OpenAI 格式 → Ollama 原生格式 (图片转 images 数组, 工具消息转普通消息)"""
        native = []
        for m in messages:
            role = m.get("role")
            if role == "tool":
                native.append({"role": "user",
                               "content": f"工具执行结果: {m.get('content', '')}"})
                continue
            if role == "assistant" and m.get("tool_calls"):
                # 历史中的工具调用: 转成文本描述 (原生格式不接受 assistant tool_calls 历史)
                parts = [m.get("content") or ""]
                for tc in m["tool_calls"]:
                    fn = tc.get("function", {})
                    parts.append(f"[已调用工具 {fn.get('name')}({fn.get('arguments')})]")
                native.append({"role": "assistant", "content": " ".join(p for p in parts if p)})
                continue
            c = m.get("content")
            if isinstance(c, list):
                texts, images = [], []
                for part in c:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "text":
                        texts.append(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        if url.startswith("data:"):
                            images.append(url.split(",", 1)[1])
                nm = {"role": role, "content": "\n".join(texts)}
                if images:
                    nm["images"] = images
                native.append(nm)
            else:
                native.append({"role": role, "content": c or ""})
        return native

    @staticmethod
    def _apply_no_think(native_msgs: list) -> list:
        """qwen3-vl 裸模板不响应 think:false (Ollama Issue #14798, RENDERER/PARSER 为
        qwen3-vl-thinking 但 TEMPLATE 只是 {{ .Prompt }}), 改用官方 workaround:
          1) 末尾 user 消息文本追加 '/no_think'
          2) 预填空思考块 '<think>\\n\\n</think>' (Qwen3 约定: 空思考块 = 不思考)
          3) raw:true 绕过模板直接送模型
        实测 (2026-08-04): 纯文本/系统提示词/工具调用/图像 全部生效, thinking=0。
        """
        msgs = [dict(m) for m in native_msgs]
        for m in reversed(msgs):
            if m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, str):
                    m["content"] = c.rstrip() + " /no_think"
                break
        msgs.append({"role": "assistant", "content": "<think>\n\n</think>\n\n"})
        return msgs

    async def _chat_native(self, client: httpx.AsyncClient, messages: list,
                           tools: Optional[list], temperature: float,
                           max_tokens: int, think: bool = True) -> LLMResponse:
        """Ollama 原生 /api/chat (图片/工具均支持; think 由深度思考开关控制)"""
        native_msgs = self._to_native_messages(messages)
        # qwen3-vl 裸模板不响应 think:false → 用 /no_think workaround (见 _apply_no_think)
        if not think:
            native_msgs = self._apply_no_think(native_msgs)
        payload = {
            "model": self.model_name,
            "messages": native_msgs,
            "stream": False,
            "think": think,  # 深度思考: 开=true(思考随响应返回前端折叠展示) / 关=false
            # qwen3-vl 默认上下文仅 4096, 图片 token 易超限; 显式抬高
            "options": {"temperature": temperature, "num_predict": max_tokens,
                        "num_ctx": 16384},
        }
        if not think:
            payload["raw"] = True   # 绕过模板, 让 /no_think + 空思考块生效
        if tools:
            payload["tools"] = tools

        for attempt in range(3):
            try:
                resp = await client.post("/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
                msg = data.get("message", {})
                # 工具调用 (原生格式 arguments 已是 dict, 兼容 str)
                tool_calls = []
                for i, tc in enumerate(msg.get("tool_calls") or []):
                    fn = tc.get("function", {})
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {"_raw": args}
                    tool_calls.append(ToolCall(
                        id=tc.get("id", f"native_{i}"),
                        name=fn.get("name", ""),
                        arguments=args,
                    ))
                return LLMResponse(
                    content=msg.get("content") or "",
                    thinking=msg.get("thinking") or msg.get("reasoning_content") or None,
                    tool_calls=tool_calls,
                    finish_reason="tool_calls" if tool_calls else "stop",
                    usage={"prompt_tokens": data.get("prompt_eval_count", 0),
                           "completion_tokens": data.get("eval_count", 0)},
                    raw=data,
                )
            except (httpx.TimeoutException, httpx.TransportError) as e:
                if attempt < 2:
                    await self._sleep(2 ** attempt)
                    continue
                return LLMResponse(content=f"[LLM连接失败] {e}", finish_reason="error")
            except httpx.HTTPStatusError as e:
                return LLMResponse(
                    content=f"[LLM HTTP错误] {e.response.status_code}: {e.response.text[:200]}",
                    finish_reason="error")
            except Exception as e:
                return LLMResponse(
                    content=f"[LLM响应异常] {type(e).__name__}: {e}",
                    finish_reason="error")

    async def chat_stream(self, messages: list, tools: Optional[list] = None,
                          temperature: float = 0.3, max_tokens: int = 2048,
                          think: Optional[bool] = None):
        """流式对话 (async generator), 事件协议:
          {"t":"think","d":str}   思考内容增量
          {"t":"tok","d":str}     正式回答增量
          {"t":"finish","tool_calls":[ToolCall],"usage":{}}  生成结束
          {"t":"error","d":str}   错误
        仅支持 Qwen3 (原生端点); 其他模型回退为非流式单次输出。
        think: 深度思考开关 (None → 用 think_default)。
        """
        use_think = self.think_default if think is None else think
        if "qwen3" not in self.model_name:
            resp = await self.chat(messages, tools, temperature, max_tokens, think=use_think)
            if resp.thinking:
                yield {"t": "think", "d": resp.thinking}
            if resp.content:
                yield {"t": "tok", "d": resp.content}
            yield {"t": "finish", "tool_calls": resp.tool_calls, "usage": resp.usage}
            return

        native_msgs = self._to_native_messages(messages)
        if not use_think:
            native_msgs = self._apply_no_think(native_msgs)
        payload = {
            "model": self.model_name,
            "messages": native_msgs,
            "stream": True,
            "think": use_think,
            "options": {"temperature": temperature, "num_predict": max_tokens,
                        "num_ctx": 16384},
        }
        if not use_think:
            payload["raw"] = True   # /no_think 方案需绕过模板
        if tools:
            payload["tools"] = tools

        client = await self._get_client()
        try:
            async with client.stream("POST", "/api/chat", json=payload,
                                     timeout=httpx.Timeout(600.0, connect=10.0)) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if d.get("error"):
                        yield {"t": "error", "d": str(d["error"])[:200]}
                        return
                    msg = d.get("message", {})
                    th = msg.get("thinking") or ""
                    ct = msg.get("content") or ""
                    if th:
                        yield {"t": "think", "d": th}
                    if ct:
                        yield {"t": "tok", "d": ct}
                    if d.get("done"):
                        tool_calls = []
                        for i, tc in enumerate(msg.get("tool_calls") or []):
                            fn = tc.get("function", {})
                            args = fn.get("arguments", {})
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except json.JSONDecodeError:
                                    args = {"_raw": args}
                            tool_calls.append(ToolCall(
                                id=tc.get("id", f"native_{i}"),
                                name=fn.get("name", ""),
                                arguments=args))
                        yield {"t": "finish", "tool_calls": tool_calls,
                               "usage": {"prompt_tokens": d.get("prompt_eval_count", 0),
                                          "completion_tokens": d.get("eval_count", 0)}}
                        return
            yield {"t": "error", "d": "流提前结束(未收到 done)"}
        except (httpx.TimeoutException, httpx.TransportError) as e:
            yield {"t": "error", "d": f"[LLM连接失败] {e}"}
        except httpx.HTTPStatusError as e:
            yield {"t": "error", "d": f"[LLM HTTP错误] {e.response.status_code}"}
        except Exception as e:
            yield {"t": "error", "d": f"[流式异常] {type(e).__name__}: {e}"}

    def _parse_response(self, data: dict) -> LLMResponse:
        """解析 OpenAI 格式响应"""
        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        finish = choice.get("finish_reason", "stop")

        # 解析 tool_calls
        tool_calls = []
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            args_str = fn.get("arguments", "{}")
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except json.JSONDecodeError:
                args = {"_raw": args_str}
            tool_calls.append(ToolCall(
                id=tc.get("id", ""),
                name=fn.get("name", ""),
                arguments=args,
            ))

        return LLMResponse(
            # 兜底: 部分模型把输出全放进 reasoning_content 时从思考内容恢复
            content=msg.get("content") or msg.get("reasoning_content") or "",
            thinking=msg.get("reasoning_content") or None,
            tool_calls=tool_calls,
            finish_reason=finish,
            usage=data.get("usage", {}),
            raw=data,
        )

    # ─── 便捷方法 ───────────────────────────────────────────────

    async def ask_text(self, prompt: str, system: str = "") -> str:
        """纯文本问答, 返回文本"""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = await self.chat(messages)
        return resp.content or ""

    async def ask_with_image(
        self, prompt: str, image_b64: str, system: str = ""
    ) -> str:
        """图片+文本问答"""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": prompt},
            ]
        })
        resp = await self.chat(messages)
        return resp.content or ""

    async def health_check(self) -> bool:
        """检查 Ollama 服务是否在线"""
        try:
            client = await self._get_client()
            resp = await client.get("/api/version", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    # ─── 工具方法 ───────────────────────────────────────────────

    @staticmethod
    def frame_to_base64(frame, quality: int = 75,
                        max_side: int = 1280) -> Optional[str]:
        """numpy RGB frame → base64 JPEG string (长边限制 max_side,
        控制 VLM 图像 token 开销)

        state.latest_frame 为 RGB; cv2.imencode 期望 BGR, 先转色 (2026-08-04 修复:
        原直接 imencode 致红蓝互换, He-Ne 红条纹在 VLM 图中显示为蓝色)。
        """
        if frame is None:
            return None
        if cv2 is not None:
            f = frame
            h, w = f.shape[:2]
            if max(h, w) > max_side:
                scale = max_side / max(h, w)
                f = cv2.resize(f, (int(w * scale), int(h * scale)))
            bgr = (cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
                   if len(f.shape) == 3 and f.shape[2] == 3 else f)
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ok:
                return base64.b64encode(buf).decode("ascii")
        # fallback: PIL (直接接受 RGB)
        try:
            from PIL import Image
            import io
            img = Image.fromarray(frame)
            bio = io.BytesIO()
            img.save(bio, format="JPEG", quality=quality)
            return base64.b64encode(bio.getvalue()).decode("ascii")
        except Exception:
            return None

    @staticmethod
    async def _sleep(seconds: float):
        import asyncio
        await asyncio.sleep(seconds)
