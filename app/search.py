"""联网搜索 — Bing (cn.bing.com, 国内可达)

DuckDuckGo lite 在国内超时; 百度有反爬 302; cn.bing.com 的 HTML 搜索结果
含 b_algo 标记可稳定解析 (2026-08-04 实测: 中文物理类查询结果相关)。
供 Agent 深度思考开关旁的 🌐 联网搜索开关使用 (与 RAG 并列的第二知识源)。
"""
import html as _html
import re

import httpx

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36")
_TIMEOUT = 12.0


def web_search(query: str, top_n: int = 5) -> list[dict]:
    """Bing 联网搜索, 返回 [{title, url, snippet}]. 失败/空返回 []."""
    try:
        r = httpx.get("https://cn.bing.com/search",
                      params={"q": query}, timeout=_TIMEOUT,
                      follow_redirects=True, headers={"User-Agent": _UA})
        if r.status_code != 200:
            return []
        blocks = re.split(r'<li class="b_algo"', r.text)
        results: list[dict] = []
        for b in blocks[1:top_n + 1]:
            m = re.search(r'<h2[^>]*><a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', b, re.S)
            if not m:
                continue
            url = m.group(1)
            title = _html.unescape(re.sub(r'<[^>]+>', '', m.group(2))).strip()
            if not title:
                continue
            s = re.search(r'<p[^>]*>(.*?)</p>', b, re.S)
            snip = _html.unescape(re.sub(r'<[^>]+>', '', s.group(1))).strip() if s else ""
            results.append({"title": title, "url": url, "snippet": snip[:200]})
        return results
    except Exception:
        return []


def build_context(query: str, top_n: int = 5, max_chars: int = 1500) -> str:
    """搜索结果拼成上下文文本 (供提示词注入), 无结果返回空串."""
    results = web_search(query, top_n)
    if not results:
        return ""
    parts = [f"【联网搜索结果】关键词: {query}"]
    total = len(parts[0])
    for i, r in enumerate(results, 1):
        seg = f"{i}. {r['title']}\n   URL: {r['url']}\n   {r['snippet']}"
        if total + len(seg) > max_chars:
            break
        parts.append(seg)
        total += len(seg)
    return "\n\n".join(parts)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    import json
    q = "迈克尔逊干涉仪 测热膨胀系数 原理"
    print(f"查询: {q}")
    print(json.dumps(web_search(q), ensure_ascii=False, indent=1)[:800])
